"""Rank useful checks using current observations, without claiming a root cause."""

from datetime import timedelta

from incident_common.hypothesis_models import CauseAssessment, EvidenceNote
from incident_common.models import Record, ToolName
from incident_common.monitoring_models import get_monitoring_window
from pydantic import ValidationError

from incident_client.memory import InvestigationMemory

_NEARBY_FAILURE_WINDOW = timedelta(minutes=5)
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _records(memory: InvestigationMemory, tool: ToolName) -> list[Record]:
    result = memory.results.get(tool)
    return result.records if result else []


def _note(record: Record, observation: str) -> EvidenceNote:
    # Cite the record actually returned by the tool, not an unseen child reference.
    return EvidenceNote(reference=record.reference, observation=observation)


def _unchecked(memory: InvestigationMemory, tool: ToolName) -> bool:
    return (
        tool not in memory.results
        and not memory.attempts.get(tool)
        and not any(entry.tool == tool for entry in memory.audit)
    )


def _settlement_records(memory: InvestigationMemory, tool: ToolName) -> list[Record]:
    return [
        record for record in _records(memory, tool) if record.facts.get("service") == "settlement"
    ]


def assess_causes(memory: InvestigationMemory) -> list[CauseAssessment]:
    """Return ranked, cited checks. A rank means useful to check, not probability.

    The five-minute comparison is a small teaching rule. It compares sampled
    transaction observations in the same service; it cannot prove a full outage.
    A tool already attempted is never suggested again by this helper.
    """
    logs = [
        record
        for record in _settlement_records(memory, ToolName.LOGS)
        if record.facts.get("http_status") in (502, 503)
    ]
    traces = [
        record
        for record in _settlement_records(memory, ToolName.TRACES)
        if record.facts.get("successful_response") is False
    ]
    successful_traces = [
        record
        for record in _settlement_records(memory, ToolName.TRACES)
        if record.facts.get("successful_response") is True
    ]
    monitors = _settlement_records(memory, ToolName.MONITORING)
    elevated = [record for record in monitors if record.facts.get("elevated_errors") is True]
    if not logs and not traces and not elevated:
        return []

    failure_notes = [
        _note(record, f"Current settlement calls returned HTTP {record.facts['http_status']}.")
        for record in logs
    ] + [
        _note(record, "The current settlement trace has no successful response.")
        for record in traces
    ]
    broad_support = list(failure_notes)
    broad_opposing: list[EvidenceNote] = []
    specific_support = list(failure_notes)
    specific_opposing: list[EvidenceNote] = []
    broad_missing = ["Current downstream evidence explaining why the settlement calls failed."]
    specific_missing = ["Evidence explaining what is different about this transaction."]
    broad_status = "needs_evidence"
    broad_priority = "medium"
    specific_status = "needs_evidence"
    specific_priority = "medium"
    has_comparison = False
    clustered_failures = False
    isolated_sample = False

    for record in monitors:
        facts = record.facts
        if facts.get("elevated_errors") is True:
            note = _note(record, "Monitoring shows elevated settlement-service errors.")
            broad_support.append(note)
            specific_opposing.append(note)
            broad_status = "possible"
        elif facts.get("elevated_errors") is False:
            broad_opposing.append(
                _note(record, "Monitoring does not show elevated settlement-service errors.")
            )
        if facts.get("application_healthy") is True:
            broad_support.append(_note(record, "The application health check is healthy."))
        try:
            window = get_monitoring_window(facts)
        except ValidationError:
            broad_missing.append("A valid time window for the monitoring comparison.")
            continue
        if window is None or window.service != "settlement":
            continue
        target = memory.results[ToolName.MONITORING].transaction_id
        current_failures = [
            item
            for item in window.observations
            if item.transaction_id == target
            and item.outcome == "failed"
            and item.http_status in {502, 503}
        ]
        if not current_failures:
            continue
        has_comparison = True
        nearby = [
            item
            for item in window.observations
            if item.transaction_id != target
            and any(
                abs(item.observed_at - current.observed_at) <= _NEARBY_FAILURE_WINDOW
                for current in current_failures
            )
        ]
        failed = [
            item for item in nearby if item.outcome == "failed" and item.http_status in {502, 503}
        ]
        succeeded = [item for item in nearby if item.outcome == "succeeded"]
        other_failed = sorted({item.transaction_id for item in failed})
        if len(other_failed) >= 2:
            clustered_failures = True
            times = ", ".join(
                f"{item.transaction_id} at {item.observed_at.isoformat()}" for item in failed
            )
            observation = (
                f"The sampled settlement window {window.window_start.isoformat()} to "
                f"{window.window_end.isoformat()} shows {len(other_failed)} other transactions "
                f"with HTTP 502/503 within five minutes of {target}: {times}. "
                "This supports a shared service problem as a possibility; "
                "it does not prove an outage."
            )
            note = _note(record, observation)
            broad_support.append(note)
            specific_opposing.append(note)
            broad_status, broad_priority = "possible", "high"
            specific_status, specific_priority = "less_supported", "low"
        elif not other_failed and succeeded:
            isolated_sample = True
            times = ", ".join(
                f"{item.transaction_id} at {item.observed_at.isoformat()}" for item in succeeded
            )
            note = _note(
                record,
                f"In the sampled window, {target} failed while other settlement transactions "
                f"succeeded within five minutes: {times}. No other HTTP 502/503 was sampled "
                "near that failure; this is not proof that no other transaction failed.",
            )
            specific_support.append(note)
            broad_opposing.append(note)
            specific_status, specific_priority = "possible", "high"
        elif other_failed:
            note = _note(
                record,
                f"One other sampled settlement transaction, {other_failed[0]}, returned HTTP "
                f"502/503 within five minutes of {target}; more coverage is needed.",
            )
            broad_support.append(note)
            broad_status = "possible"

    if not has_comparison:
        broad_missing.append("The timing and results of other transactions in the same service.")
        specific_missing.append("Other transaction results close to the current failure time.")
    if clustered_failures and isolated_sample:
        # Multiple snapshots can disagree; report both instead of selecting a winner.
        broad_status = specific_status = "needs_evidence"
        broad_priority = specific_priority = "high"
        broad_missing.append("Engineer review of the different transaction samples.")
        specific_missing.append("Engineer review of the different transaction samples.")
    elif not elevated and broad_opposing and not clustered_failures:
        broad_status, broad_priority = "less_supported", "low"
    if successful_traces and logs:
        conflict = (
            "Resolve the HTTP errors and the successful current settlement trace "
            "before narrowing the cause."
        )
        broad_status = specific_status = "needs_evidence"
        broad_missing.append(conflict)
        specific_missing.append(conflict)
        for record in successful_traces:
            note = _note(
                record,
                "The current settlement trace reports a successful response, "
                "which needs review against the errors.",
            )
            broad_opposing.append(note)
            specific_opposing.append(note)
    if elevated and any(record.facts.get("elevated_errors") is False for record in monitors):
        broad_status = specific_status = "needs_evidence"
        conflict = (
            "Resolve the different settlement-service health observations "
            "before narrowing the cause."
        )
        broad_missing.append(conflict)
        specific_missing.append(conflict)

    monitoring_unchecked = _unchecked(memory, ToolName.MONITORING)
    monitoring_follow_up = (
        "Use the monitoring already returned; "
        "ask the engineer for current downstream metrics "
        "or a wider transaction sample to explain the shared failures."
        if monitors
        else "The monitoring check was already attempted without usable settlement evidence; "
        "ask the engineer for service health and other transaction results in the same time window."
    )
    trace_follow_up = (
        "Use the trace already returned; ask the engineer to compare the request details "
        "with successful transactions from the same time window."
        if traces or successful_traces
        else "The trace check was already attempted without usable settlement evidence; "
        "ask the engineer for the request path and retry outcome."
    )
    assessments = [
        CauseAssessment(
            cause_id="broad_service_issue",
            explanation="A shared settlement-service problem may affect several transactions.",
            status=broad_status,
            priority=broad_priority,
            supporting_evidence=broad_support,
            opposing_evidence=broad_opposing,
            missing_evidence=broad_missing,
            next_check=(
                "Check settlement-service health and compare other transaction failures "
                "in the same time window."
                if monitoring_unchecked
                else monitoring_follow_up
            ),
            next_tool="get_monitoring" if monitoring_unchecked else None,
        ),
        CauseAssessment(
            cause_id="transaction_specific_issue",
            explanation="Something specific to this transaction may explain its failure.",
            status=specific_status,
            priority=specific_priority,
            supporting_evidence=specific_support,
            opposing_evidence=specific_opposing,
            missing_evidence=specific_missing,
            next_check=(
                "Inspect the current settlement trace for the request path, response, "
                "and retry outcome."
                if _unchecked(memory, ToolName.TRACES)
                else trace_follow_up
            ),
            next_tool="get_trace" if _unchecked(memory, ToolName.TRACES) else None,
        ),
    ]

    deployments = _records(memory, ToolName.DEPLOYMENTS)
    changed = [record for record in deployments if record.facts.get("relevant_change") is True]
    unchanged = [
        record
        for record in deployments
        if record.facts.get("relevant_change") is False
        and record.facts.get("check_completed") is True
    ]
    deployment_follow_up = (
        "Use the deployment evidence already returned; ask the engineer about changes outside "
        "its coverage only if new evidence points to them."
        if changed or unchanged
        else "The deployment check was already attempted without a completed change review; "
        "ask the engineer for the missing change history."
    )
    assessments.append(
        CauseAssessment(
            cause_id="recent_deployment",
            explanation="A related recent change may have contributed to the settlement failure.",
            status="needs_evidence"
            if changed and unchanged
            else "possible"
            if changed
            else "less_supported"
            if unchanged
            else "needs_evidence",
            priority="high" if changed else "low" if unchanged else "medium",
            supporting_evidence=[
                _note(
                    record,
                    "A relevant recent change was found; timing alone does not prove a cause.",
                )
                for record in changed
            ],
            opposing_evidence=[
                _note(
                    record,
                    "The deployment check found no relevant recent change; "
                    "this does not rule it out.",
                )
                for record in unchanged
            ],
            missing_evidence=["Evidence linking a particular change to the observed failures."]
            + (
                ["A completed deployment check covering the failure time."]
                if not changed and not unchanged
                else []
            ),
            next_check=(
                "Check recent application and settlement-service changes against the failure time."
                if _unchecked(memory, ToolName.DEPLOYMENTS)
                else deployment_follow_up
            ),
            next_tool="get_deployments" if _unchecked(memory, ToolName.DEPLOYMENTS) else None,
        )
    )
    histories = _settlement_records(memory, ToolName.SIMILAR) + _settlement_records(
        memory, ToolName.HISTORY
    )
    pool_clues = [
        record
        for record in histories
        if isinstance(record.facts.get("past_root_cause"), str)
        and "pool" in record.facts["past_root_cause"].lower()
        and "connection" in record.facts["past_root_cause"].lower()
        and record.facts.get("root_cause_confirmed") is not False
    ]
    if pool_clues:
        assessments.append(
            CauseAssessment(
                cause_id="connection_pool_issue",
                explanation="A past connection-pool problem is a lead to check in this service.",
                status="needs_evidence",
                priority="medium",
                supporting_evidence=[
                    _note(
                        record,
                        f"Past case {record.facts.get('past_incident_id', 'unknown')} records "
                        f"'{record.facts['past_root_cause']}'. This suggests a check and is not "
                        "evidence of the current cause.",
                    )
                    for record in pool_clues
                ],
                missing_evidence=[
                    "Current connection-pool metrics are not available from the current tools."
                ],
                next_check=(
                    "Ask the engineer for current connection-pool usage, wait times, and errors "
                    "around the settlement failure; "
                    "the existing monitoring tool cannot provide these metrics."
                ),
                next_tool=None,
            )
        )
    return sorted(assessments, key=lambda item: _PRIORITY_ORDER[item.priority])
