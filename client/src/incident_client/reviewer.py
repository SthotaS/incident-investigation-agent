"""Verify structured findings against the original validated tool records."""

from datetime import datetime

from incident_common.hypothesis_models import CauseAssessment, EvidenceNote
from incident_common.models import InvestigationReport, Record, Source
from incident_common.review_models import EvidenceReview, ReviewIssue, ReviewPass


def _failure_records(by_source: dict[Source, list[Record]]) -> tuple[list[Record], list[Record]]:
    logs = [
        r
        for r in by_source.get(Source.LOGS, [])
        if r.facts.get("service") == "settlement" and r.facts.get("http_status") == 503
    ]
    failed = [
        r
        for r in by_source.get(Source.TRACES, [])
        if r.facts.get("service") == "settlement"
        and r.facts.get("successful_response") is False
        and r.facts.get("retries_exhausted") is True
    ]
    return logs, failed


def _deployment_issues(
    cause: CauseAssessment, deployments: list[Record], failure_records: list[Record]
) -> list[ReviewIssue]:
    relevant = [r for r in deployments if r.facts.get("relevant_change") is True]
    cited = {note.reference for note in cause.supporting_evidence}
    cited_deployments = [r for r in relevant if r.reference in cited]
    candidates = cited_deployments or relevant
    if not candidates or not failure_records:
        return [
            ReviewIssue(
                code="missing_timeline",
                detail="Current deployment and failure records are needed to compare timing.",
                severity="revision",
            )
        ]

    first_failure = min(datetime.fromisoformat(r.observed_at) for r in failure_records)
    late = []
    for record in candidates:
        deployed_at = record.facts.get("deployed_at")
        try:
            if not isinstance(deployed_at, str):
                raise ValueError
            time = datetime.fromisoformat(deployed_at)
            if time.utcoffset() is None:
                raise ValueError
        except ValueError:
            return [
                ReviewIssue(
                    code="missing_deployment_time",
                    detail="A relevant deployment lacks a valid time with a time zone.",
                    severity="revision",
                    references=[record.reference],
                )
            ]
        if time > first_failure:
            late.append(record.reference)
    if late:
        return [
            ReviewIssue(
                code="deployment_after_failure",
                detail=f"Deployment occurred after the failure at {first_failure.isoformat()}.",
                severity="revision",
                references=late,
            )
        ]
    if not cited_deployments:
        return [
            ReviewIssue(
                code="missing_deployment_citation",
                detail="The deployment hypothesis needs a current deployment citation.",
                severity="revision",
            )
        ]
    return []


def inspect_evidence(report: InvestigationReport) -> ReviewPass:
    issues: list[ReviewIssue] = []
    records: dict[str, tuple[Source, Record]] = {}
    by_source: dict[Source, list[Record]] = {}

    def issue(code, detail, severity="revision", refs=()):
        issues.append(
            ReviewIssue(code=code, detail=detail, severity=severity, references=list(refs))
        )

    for result in report.evidence_snapshot:
        if result.transaction_id != report.transaction_id:
            issue("wrong_transaction", "Evidence belongs to another transaction.", "escalation")
            continue
        for record in result.records:
            sourced_record = (result.source, record)
            if record.reference in records and records[record.reference] != sourced_record:
                issue(
                    "conflicting_reference",
                    "One reference resolves to different source records.",
                    "escalation",
                    [record.reference],
                )
            records[record.reference] = sourced_record
            by_source.setdefault(result.source, []).append(record)

    transactions = by_source.get(Source.TRANSACTIONS, [])
    statements = [
        (status, record.reference)
        for record in transactions
        if isinstance(status := record.facts.get("status"), str) and status.strip()
    ]
    if len({status for status, _ in statements}) > 1:
        issue(
            "contradictory_transactions",
            "Transaction records disagree about the status.",
            "escalation",
            [ref for _, ref in statements],
        )

    logs, failed = _failure_records(by_source)
    successful = [
        r
        for r in by_source.get(Source.TRACES, [])
        if r.facts.get("service") == "settlement" and r.facts.get("successful_response") is True
    ]
    if logs and successful:
        issue(
            "conflicting_trace",
            "An HTTP 503 log and a successful settlement trace need reconciliation.",
            "escalation",
            [r.reference for r in logs + successful],
        )

    for cause in report.cause_assessments:
        for note in cause.supporting_evidence + cause.opposing_evidence:
            if note.reference not in records:
                issue(
                    "missing_reference",
                    "A cause cites a record absent from the snapshot.",
                    refs=[note.reference],
                )
        if cause.cause_id == "recent_deployment" and cause.status == "possible":
            issues.extend(
                _deployment_issues(cause, by_source.get(Source.DEPLOYMENTS, []), logs + failed)
            )

    if report.supported_failure_point:
        if report.supported_failure_point != "Settlement-service call" or not any(
            a.reference != b.reference for a in logs for b in failed
        ):
            issue("unsupported_failure", "Current logs and failed trace do not support this claim.")
        citations = set(report.failure_point_references)
        evidence = {record.reference for record in logs + failed}
        if not citations <= evidence or not any(
            a.reference != b.reference and {a.reference, b.reference} <= citations
            for a in logs
            for b in failed
        ):
            issue("missing_failure_citation", "Failure point needs verified current citations.")

    decision = "accept"
    if issues:
        decision = "escalate" if any(i.severity == "escalation" for i in issues) else "revise"
    return ReviewPass(decision=decision, issues=issues, checked_references=sorted(records))


def _revise_report(report: InvestigationReport, issues: list[ReviewIssue]) -> None:
    codes = {i.code for i in issues}
    if "unsupported_failure" in codes:
        report.status = "paused"
        report.stop_reason = "insufficient_evidence"
        report.supported_failure_point = None
        report.failure_point_references = []
        report.likely_cause = None
        report.next_steps = ["Ask an engineer to verify the current failure point."]
    elif "missing_failure_citation" in codes:
        by_source: dict[Source, list[Record]] = {}
        for result in report.evidence_snapshot:
            by_source.setdefault(result.source, []).extend(result.records)
        logs, failed = _failure_records(by_source)
        report.failure_point_references = sorted({r.reference for r in logs + failed})

    late = {
        reference
        for item in issues
        if item.code == "deployment_after_failure"
        for reference in item.references
    }
    if late:
        deployments = {
            record.reference
            for result in report.evidence_snapshot
            if result.source == Source.DEPLOYMENTS
            for record in result.records
            if record.facts.get("relevant_change") is True
        }
        reason = "The deployment occurred after the first observed failure."
        for index, cause in enumerate(report.cause_assessments):
            if cause.cause_id != "recent_deployment" or cause.status != "possible":
                continue
            supporting = [note for note in cause.supporting_evidence if note.reference not in late]
            update = {
                "supporting_evidence": supporting,
                "opposing_evidence": cause.opposing_evidence
                + [EvidenceNote(reference=ref, observation=reason) for ref in sorted(late)],
            }
            if not any(note.reference in deployments for note in supporting):
                update.update(
                    status="less_supported",
                    priority="low",
                    explanation=reason,
                    next_check="Check earlier changes only if new evidence points to one.",
                    next_tool=None,
                )
                if cause.next_check in report.next_steps:
                    report.next_steps.remove(cause.next_check)
            report.cause_assessments[index] = cause.model_copy(update=update)
        report.open_possible_causes = [
            cause.explanation
            for cause in report.cause_assessments
            if cause.status in {"possible", "needs_evidence"}
        ]


def review_report(report: InvestigationReport) -> InvestigationReport:
    """Make at most one local correction and then recheck the same evidence."""
    report = report.model_copy(deep=True)
    first = inspect_evidence(report)
    review = EvidenceReview(passes=[first])
    if first.decision == "revise":
        review.revisions_used = 1
        _revise_report(report, first.issues)
        second = inspect_evidence(report)
        if second.decision == "revise":
            second.decision = "escalate"
        review.passes.append(second)
    report.evidence_review = review
    if review.passes[-1].decision == "escalate":
        if report.status == "finished":
            report.stop_reason = "evidence_review_escalated"
        report.status = "paused"
        report.next_steps = [
            "Ask an engineer to resolve these evidence issues before continuing.",
            *(item.detail for item in review.passes[-1].issues),
        ]
        report.supported_failure_point = None
        report.failure_point_references = []
        report.likely_cause = None
        report.cause_assessments = []
        report.open_possible_causes = []
        report.unsupported_explanations = []
    return report
