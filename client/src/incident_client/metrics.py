"""Measured investigation work; a changed next tool is not an efficiency gain."""

from incident_common.models import InvestigationReport, ToolName


def finding_call(report: InvestigationReport) -> int | None:
    """First successful log/trace pair supporting the engine's final checked finding.

    This is evidence readiness, not the time an engineer confirmed a root cause.
    Current tools return each non-error source once. Paused/unsupported/conflicting
    findings have no supported-finding readiness marker.
    """
    if report.supported_failure_point is None or report.stop_reason == "conflicting_evidence":
        return None
    logs, traces = set(), set()
    for entry in report.audit:
        if entry.status != "found":
            continue
        if entry.tool == ToolName.LOGS:
            logs.update(entry.record_references)
        elif entry.tool == ToolName.TRACES:
            traces.update(entry.record_references)
        if any(log != trace for log in logs for trace in traces):
            return entry.call_number
    return None


def work_metrics(report: InvestigationReport) -> dict:
    return {
        "tool_calls": report.calls_used,
        "retrieval_attempts": sum(entry.tool == ToolName.SIMILAR for entry in report.audit),
        "calls_to_supported_finding": finding_call(report),
        "supported_failure_point": report.supported_failure_point,
        "cause_states": {cause.cause_id: cause.status for cause in report.cause_assessments},
        "model_requests": report.model_calls,
    }
