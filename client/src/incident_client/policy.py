"""Tool priorities based on observed evidence, with hard investigation limits."""

from dataclasses import dataclass
from typing import Literal

from incident_common.models import InvestigationRequest, ToolName
from pydantic import BaseModel, ConfigDict, Field

from incident_client.hypotheses import assess_causes
from incident_client.memory import InvestigationMemory


class InvestigationLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tool_calls: int = Field(default=8, ge=1, le=8)
    max_retries_per_tool: int = Field(default=2, ge=0, le=2)
    max_steps_without_evidence: Literal[2] = 2


@dataclass(frozen=True)
class Action:
    tool: ToolName
    reason: str
    evidence_references: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckPriority:
    tool: ToolName
    priority: Literal["high", "medium", "low"]
    reason: str
    evidence_references: tuple[str, ...]
    tests_causes: tuple[str, ...]


def available_tools(request: InvestigationRequest, memory: InvestigationMemory) -> list[ToolName]:
    """Return unused checks; completed/empty checks are not repeated automatically."""
    if ToolName.TRANSACTION not in memory.results:
        return [ToolName.TRANSACTION]
    if request.scope == "status":
        return []
    if ToolName.LOGS not in memory.results:
        return [ToolName.LOGS]
    if request.mode == "with-rag" and ToolName.SIMILAR not in memory.results:
        return [ToolName.SIMILAR]
    remaining = [
        tool
        for tool in (ToolName.TRACES, ToolName.MONITORING, ToolName.DEPLOYMENTS)
        if tool not in memory.results
    ]
    if request.mode == "legacy" and not remaining and ToolName.HISTORY not in memory.results:
        return [ToolName.HISTORY]
    return remaining


def retrieval_query(memory: InvestigationMemory) -> str:
    """Use observed symptoms, never a hidden answer or a desired historical ID."""
    return " ".join(
        record.summary
        for tool in (ToolName.TRANSACTION, ToolName.LOGS)
        if (result := memory.results.get(tool))
        for record in result.records
    )[:2000]


def _references(memory: InvestigationMemory, *tools: ToolName) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            record.reference
            for tool in tools
            if (result := memory.results.get(tool))
            for record in result.records
        )
    )


def prioritize_checks(
    request: InvestigationRequest, memory: InvestigationMemory
) -> list[CheckPriority]:
    """Rank checks that can separate open causes. Ranks are not cause probabilities.

    Current evidence takes priority over an old case. A historical hint may reorder
    otherwise useful checks, but cannot establish a current cause or repeat a checked tool.
    """
    available = available_tools(request, memory)
    setup_reasons = {
        ToolName.TRANSACTION: "Check the transaction status and last known step.",
        ToolName.LOGS: "The transaction is delayed; look for related errors.",
        ToolName.SIMILAR: "Search past incidents using the status and errors seen so far.",
        ToolName.HISTORY: "Compare this pattern with confirmed past incidents.",
    }
    if len(available) == 1 and available[0] in setup_reasons:
        return [CheckPriority(available[0], "high", setup_reasons[available[0]], (), ())]
    downstream_errors = memory.has_facts(ToolName.LOGS, service="settlement", http_status=503)
    failed_trace = memory.has_facts(
        ToolName.TRACES, service="settlement", successful_response=False, retries_exhausted=True
    )
    local_error = memory.has_fact(ToolName.LOGS, "local_exception", True)
    no_related_change = memory.has_facts(
        ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
    )
    rankings = {}
    for tool in available:
        if tool == ToolName.TRACES:
            reason = (
                "HTTP 503s make the settlement call worth checking in the trace. "
                "Separate a failed request path from a wider service problem."
                if downstream_errors
                else "Check where the request stopped before naming a cause."
            )
            rank, refs, causes = (
                60,
                _references(memory, ToolName.LOGS),
                ("transaction_specific_issue", "broad_service_issue"),
            )
            if ToolName.MONITORING in memory.results:
                rank = 95
                monitoring = memory.results[ToolName.MONITORING]
                relevant_records = [
                    record
                    for record in monitoring.records
                    if record.facts.get("service") == "settlement"
                ]
                causes_by_id = {item.cause_id: item for item in assess_causes(memory)}
                broad = causes_by_id.get("broad_service_issue")
                specific = causes_by_id.get("transaction_specific_issue")
                if not relevant_records:
                    reason = (
                        "Monitoring was already checked but returned no relevant settlement "
                        "evidence. Inspect this transaction's trace for its response and retries."
                    )
                elif broad and broad.status == "possible" and broad.priority == "high":
                    reason = (
                        "Use the monitoring already collected: several transactions failed "
                        "close in time. Trace this transaction to check whether its failed "
                        "request matches that shared pattern."
                    )
                elif specific and specific.status == "possible" and specific.priority == "high":
                    reason = (
                        "Use the monitoring already collected: nearby transactions succeeded "
                        "while this one failed. Inspect its trace for a request-specific clue."
                    )
                else:
                    reason = (
                        "Use the monitoring already collected, which does not establish a "
                        "shared failure pattern. Inspect the current trace for the request "
                        "path and retry outcome."
                    )
                refs = _references(memory, ToolName.LOGS, ToolName.MONITORING)
        elif tool == ToolName.MONITORING:
            rank, refs, causes = (
                40,
                _references(memory, ToolName.LOGS, ToolName.TRACES),
                ("broad_service_issue", "transaction_specific_issue"),
            )
            reason = (
                "Compare service health and failures in the same time window to separate "
                "a wider service problem from a transaction-specific problem."
            )
            if downstream_errors and failed_trace:
                rank = 100
                reason = (
                    "Logs and the trace show failed settlement calls. Check service health "
                    "and other transactions in the same time window to narrow the cause."
                )
                if no_related_change:
                    reason = (
                        "The deployment check found no related change, "
                        "weakening the release explanation. "
                        "Check service health and other transactions in the same time window "
                        "to distinguish "
                        "a wider service problem from a transaction-specific failure."
                    )
                    refs = _references(memory, ToolName.LOGS, ToolName.TRACES, ToolName.DEPLOYMENTS)
        elif tool == ToolName.DEPLOYMENTS:
            rank, refs, causes = (
                20,
                _references(memory, ToolName.LOGS, ToolName.MONITORING),
                ("recent_deployment",),
            )
            reason = "Check whether a recent change matches the observed error timing."
            if local_error:
                rank = 110
                reason = "A local error makes recent application changes useful to check first."
        else:
            continue
        rankings[tool] = [rank, reason, refs, causes]

    # A historical hint is useful before stronger current evidence separates the causes.
    history = memory.results.get(ToolName.SIMILAR) if request.mode == "with-rag" else None
    if (
        history
        and not local_error
        and not failed_trace
        and ToolName.MONITORING not in memory.results
    ):
        for record in history.records:
            suggested, service = record.facts.get("useful_check"), record.facts.get("service")
            if service and memory.has_fact(ToolName.LOGS, "service", service):
                for tool in available:
                    if suggested == tool.value and tool in rankings:
                        rankings[tool][0] = 85
                        rankings[tool][1] = (
                            f"Past {record.facts.get('past_incident_id')} has a matching service "
                            f"pattern. {record.facts.get('check_reason')} "
                            "Check current evidence before drawing a conclusion."
                        )
                        rankings[tool][2] = (*_references(memory, ToolName.LOGS), record.reference)
                break
    ordered = sorted(rankings, key=lambda tool: -rankings[tool][0])
    return [
        CheckPriority(
            tool=tool,
            priority="high"
            if rankings[tool][0] >= 80
            else "medium"
            if rankings[tool][0] >= 40
            else "low",
            reason=rankings[tool][1],
            evidence_references=rankings[tool][2],
            tests_causes=rankings[tool][3],
        )
        for tool in ordered
    ]


class DemoPlanner:
    """A repeatable evidence-based planner; API decisions use the same priority context."""

    name = "demo"
    model = None
    calls = 0

    def choose(self, request: InvestigationRequest, memory: InvestigationMemory) -> Action | None:
        checks = prioritize_checks(request, memory)
        if not checks:
            return None
        selected = checks[0]
        return Action(selected.tool, selected.reason, selected.evidence_references)
