"""Show possible causes and the evidence behind the next useful check."""

import re
from textwrap import shorten

from incident_common.hypothesis_models import CauseAssessment
from incident_common.models import AuditEntry, InvestigationReport, Source, ToolName

_TOOL_LABELS = {
    ToolName.TRANSACTION: "Transaction",
    ToolName.LOGS: "Logs",
    ToolName.TRACES: "Trace",
    ToolName.MONITORING: "Monitoring",
    ToolName.DEPLOYMENTS: "Deployments",
    ToolName.HISTORY: "Past incident",
    ToolName.SIMILAR: "Similar incidents",
}
_SOURCE_LABELS = {
    Source.TRANSACTIONS: "transaction status",
    Source.LOGS: "logs",
    Source.TRACES: "trace",
    Source.MONITORING: "service health",
    Source.DEPLOYMENTS: "deployments",
    Source.HISTORY: "past incidents",
}
_CAUSE_LABELS = {
    "broad_service_issue": "Shared service issue",
    "transaction_specific_issue": "Transaction-specific issue",
    "recent_deployment": "Recent change",
    "connection_pool_issue": "Connection-pool issue",
}
_CHECK_LABELS = {
    "get_monitoring": "service health and nearby transaction results",
    "get_trace": "this transaction's trace and retry results",
    "get_deployments": "changes around the failure time",
}


def render_progress(entry: AuditEntry, *, verbose: bool = False) -> str:
    if verbose:
        lines = [
            f"[{entry.call_number}/8] {entry.tool.value}: {entry.status}",
            f"  Why: {entry.reason}",
            f"  Found: {entry.observation}",
        ]
        if entry.decision_references:
            lines.append("  Decision sources: " + ", ".join(entry.decision_references))
        return "\n".join(lines)
    observation = shorten(entry.observation, width=125, placeholder="...")
    if entry.tool in (ToolName.SIMILAR, ToolName.HISTORY) and entry.status == "found":
        ids = list(
            dict.fromkeys(
                ref.rsplit("/", 1)[-1]
                for ref in entry.record_references
                if re.fullmatch(r"INC-[0-9]{5}", ref.rsplit("/", 1)[-1])
            )
        )
        if ids:
            observation = "Past cases: " + ", ".join(ids)
    lines = [f"[{entry.call_number}/8] {_TOOL_LABELS[entry.tool]}: {entry.status}; {observation}"]
    if entry.tool in (ToolName.TRACES, ToolName.MONITORING, ToolName.DEPLOYMENTS):
        lines.append("  Why: " + shorten(entry.reason, width=150, placeholder="..."))
    return "\n".join(lines)


def render_summary(report: InvestigationReport) -> str:
    """Keep uncertainty and stop reasons visible without repeating full evidence lists."""
    stop_reasons = {
        "request_answered": "request answered",
        "planner_error": "AI planning stopped",
        "call_budget": f"{report.call_limit}-call limit reached",
        "no_new_evidence": "two checks added no new evidence",
        "tool_retry_limit": "failed tool reached its retry limit",
        "transaction_not_found": "transaction not found",
        "conflicting_evidence": "evidence conflicts",
        "insufficient_evidence": "more current evidence needed",
    }
    mode = {"with-rag": "with RAG", "without-rag": "without RAG", "legacy": "demo"}.get(
        report.mode, report.mode
    )
    lines = [f"{report.incident_id} / {report.transaction_id} | {mode}"]
    if report.model:
        lines.append(f"Model: {report.model}")
        if report.model_usage.measured_responses:
            usage = report.model_usage
            lines.append(
                f"API tokens reported: {usage.input_tokens} input, {usage.output_tokens} output "
                f"({usage.measured_responses}/{report.model_calls} attempts)."
            )
    elif report.planner == "demo":
        lines.append("Planner: Python demo rules")
    reason = stop_reasons.get(report.stop_reason, report.stop_reason.replace("_", " "))
    lines.append(
        f"Status: {report.status.capitalize()} - {reason}. "
        f"Calls: {report.calls_used}/{report.call_limit}."
    )
    if report.planner_error:
        lines.append("AI issue: " + shorten(report.planner_error, width=160, placeholder="..."))
    if report.retrieval is not None:
        ids = list(dict.fromkeys(match.incident_id for match in report.retrieval.matches))
        lines.append(
            "Past cases: " + ", ".join(ids) + " (leads, not proof)."
            if ids
            else "Past cases: no matching incidents."
        )
    if report.supported_failure_point:
        lines.append(f"Supported failure point: {report.supported_failure_point}.")
    else:
        lines.append("Failure point: Not established.")
    if report.possible_failure_point and not report.supported_failure_point:
        lines.append(f"Possible failure point: {report.possible_failure_point} (needs evidence).")
    if report.cause_assessments:
        lines.append("Possible causes:")
        for cause in report.cause_assessments[:3]:
            lines.append(f"- {_CAUSE_LABELS[cause.cause_id]}: {cause.status.replace('_', ' ')}.")
        if len(report.cause_assessments) > 3:
            lines.append("- More leads are listed in --verbose.")
    if report.missing_sources:
        lines.append(
            "Missing checks: "
            + ", ".join(_SOURCE_LABELS[item] for item in report.missing_sources)
            + "."
        )
    # A completed check is not run again. These are suggestions, not permission to resume.
    suggestion = next(
        (
            cause
            for cause in report.cause_assessments
            if cause.status != "less_supported" and cause.next_tool
        ),
        None,
    )
    if suggestion:
        lines.append("Suggested next check: " + _CHECK_LABELS[suggestion.next_tool] + ".")
    if report.status == "paused":
        lines.append("Next: Ask the engineer whether to continue.")
    elif report.next_steps:
        lines.append("Next: " + shorten(report.next_steps[0], width=150, placeholder="..."))
    lines.append("Root cause: not confirmed. Fixes are manual; saving needs engineer approval.")
    lines.append("Full details: add --verbose; full data: add --json.")
    return "\n".join(lines)


def render_cause_assessments(assessments: list[CauseAssessment]) -> str:
    lines: list[str] = []
    for cause in assessments:
        title = f"Possible cause: {cause.explanation}"
        lines += [title, ""]
        lines.append(
            f"State: {cause.status.replace('_', ' ')}; check priority: {cause.priority}. "
            "Priority means useful to check, not the chance of being the cause."
        )
        for label, evidence in (
            ("Supporting evidence", cause.supporting_evidence),
            ("Evidence against", cause.opposing_evidence),
        ):
            lines += ["", f"{label}:"]
            if evidence:
                lines += [f"- {note.observation} Source: {note.reference}" for note in evidence]
            else:
                lines.append("- None in the collected records.")
        lines += ["", "Missing evidence:"]
        lines += [f"- {item}" for item in cause.missing_evidence] or [
            "- No specific gap listed; engineer review is still needed."
        ]
        lines += ["", f"Next check: {cause.next_check}"]
        if cause.next_tool:
            lines.append(f"Available tool: {cause.next_tool}")
        lines.append("")
    return "\n".join(lines).rstrip()
