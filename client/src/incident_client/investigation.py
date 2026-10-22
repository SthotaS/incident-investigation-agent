"""Bounded demo loop. All calls and retries pass through this one budget check."""

import inspect
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from incident_common.models import (
    TOOL_SOURCES,
    AuditEntry,
    InvestigationReport,
    InvestigationRequest,
    ModelUsage,
    SkippedCheck,
    Source,
    ToolName,
    ToolResult,
)
from pydantic import ValidationError

from incident_client.hypotheses import assess_causes
from incident_client.llm import PlannerError
from incident_client.mcp_client import ToolExecutionError, ToolGateway
from incident_client.memory import InvestigationMemory
from incident_client.policy import (
    Action,
    DemoPlanner,
    InvestigationLimits,
    available_tools,
    retrieval_query,
)


def _source_is_relevant(memory: InvestigationMemory, source: Source) -> bool:
    """A nonempty response is not enough when its facts concern another service."""
    for result in memory.results.values():
        if result.source != source:
            continue
        for record in result.records:
            facts = record.facts
            if source == Source.TRANSACTIONS:
                if isinstance(facts.get("status"), str) and facts["status"].strip():
                    return True
            elif source == Source.LOGS:
                if facts.get("local_exception") is True or (
                    facts.get("service") == "settlement"
                    and isinstance(facts.get("http_status"), int)
                    and not isinstance(facts["http_status"], bool)
                    and 100 <= facts["http_status"] <= 599
                ):
                    return True
            elif source == Source.TRACES:
                if facts.get("service") == "settlement" and isinstance(
                    facts.get("successful_response"), bool
                ):
                    return True
            elif source == Source.MONITORING:
                if facts.get("service") == "settlement" and (
                    isinstance(facts.get("elevated_errors"), bool)
                    or isinstance(facts.get("application_healthy"), bool)
                    or isinstance(facts.get("monitoring_window"), dict)
                ):
                    return True
            elif source == Source.DEPLOYMENTS:
                if facts.get("check_completed") is True and isinstance(
                    facts.get("relevant_change"), bool
                ):
                    return True
            elif source == Source.HISTORY and any(
                isinstance(facts.get(name), str) and facts[name].strip()
                for name in ("past_incident_id", "similar_incident")
            ):
                return True
    return False


class InvestigationEngine:
    def __init__(
        self,
        gateway: ToolGateway,
        limits: InvestigationLimits | None = None,
        planner=None,
    ) -> None:
        self.gateway = gateway
        self.limits = limits or InvestigationLimits()
        self.planner = planner or DemoPlanner()

    async def run(
        self,
        request: InvestigationRequest,
        on_progress: Callable[[AuditEntry | SkippedCheck], None] | None = None,
    ) -> InvestigationReport:
        memory = InvestigationMemory()
        skipped_checks: list[SkippedCheck] = []
        while True:
            if (
                request.mode != "legacy"
                and not request.use_rag
                and request.scope == "full"
                and ToolName.LOGS in memory.results
                and not skipped_checks
            ):
                skipped = SkippedCheck()
                skipped_checks.append(skipped)
                if on_progress:
                    on_progress(skipped)
            transaction = memory.results.get(ToolName.TRANSACTION)
            if transaction and transaction.status == "empty":
                return self._report(
                    request, memory, skipped_checks, "paused", "transaction_not_found"
                )
            if request.scope == "status" and transaction:
                if _source_is_relevant(memory, Source.TRANSACTIONS):
                    return self._report(
                        request, memory, skipped_checks, "finished", "request_answered"
                    )
                return self._report(
                    request, memory, skipped_checks, "paused", "insufficient_evidence"
                )
            if memory.has_facts(
                ToolName.LOGS, http_status=503, service="settlement"
            ) and memory.has_facts(ToolName.TRACES, successful_response=True, service="settlement"):
                return self._report(
                    request, memory, skipped_checks, "paused", "conflicting_evidence"
                )
            if memory.no_new_evidence_count >= self.limits.max_steps_without_evidence:
                return self._report(request, memory, skipped_checks, "paused", "no_new_evidence")

            if len(memory.audit) >= self.limits.max_tool_calls:
                return self._report(request, memory, skipped_checks, "paused", "call_budget")
            action = None
            allowed = available_tools(request, memory)
            retrying = False
            if memory.audit and memory.audit[-1].status == "error":
                failed = memory.audit[-1].tool
                if memory.attempts[failed] <= self.limits.max_retries_per_tool:
                    action = Action(failed, "Retry the failed tool while calls remain.")
                    retrying = True
                else:
                    return self._report(
                        request, memory, skipped_checks, "paused", "tool_retry_limit"
                    )
            else:
                try:
                    choice = self.planner.choose(request, memory)
                    action = await choice if inspect.isawaitable(choice) else choice
                except PlannerError as exc:
                    report = self._report(
                        request, memory, skipped_checks, "paused", "planner_error"
                    )
                    report.planner_error = str(exc)
                    return report
                except Exception:
                    report = self._report(
                        request, memory, skipped_checks, "paused", "planner_error"
                    )
                    report.planner_error = "The planner could not produce a safe next check."
                    return report
            if action is None:
                if allowed:
                    report = self._report(
                        request, memory, skipped_checks, "paused", "planner_error"
                    )
                    report.planner_error = (
                        "The planner stopped before the required checks finished."
                    )
                    return report
                return self._report(request, memory, skipped_checks, "finished", "request_answered")
            known_references = {
                record.reference for result in memory.results.values() for record in result.records
            }
            if (
                not isinstance(action, Action)
                or not isinstance(action.tool, ToolName)
                or (not retrying and action.tool not in allowed)
                or not isinstance(action.reason, str)
                or not action.reason.strip()
                or len(action.reason) > 1000
                or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", action.reason)
                or not isinstance(action.evidence_references, tuple)
                or any(
                    not isinstance(reference, str) or reference not in known_references
                    for reference in action.evidence_references
                )
            ):
                report = self._report(request, memory, skipped_checks, "paused", "planner_error")
                report.planner_error = "The planner selected an invalid check or unknown evidence."
                return report
            if len(memory.audit) >= self.limits.max_tool_calls:
                return self._report(request, memory, skipped_checks, "paused", "call_budget")

            # Reserve a call before execution, including calls that fail or time out.
            call_number = len(memory.audit) + 1
            memory.attempts[action.tool] = memory.attempts.get(action.tool, 0) + 1
            called_at = datetime.now(UTC).isoformat()
            started = monotonic()
            references: list[str] = []
            arguments = {"transaction_id": request.transaction_id}
            if action.tool == ToolName.SIMILAR:
                arguments["query"] = retrieval_query(memory)
                arguments["current_incident_id"] = request.incident_id
            try:
                result = await self.gateway.call_tool(action.tool, **arguments)
                try:
                    if not isinstance(result, ToolResult):
                        raise ValueError("Expected a validated tool result")
                    # Revalidate a copy: model_copy() and nested mutation can bypass validators.
                    result = ToolResult.model_validate(result.model_dump(mode="python"))
                    if (
                        result.source != TOOL_SOURCES[action.tool]
                        or result.transaction_id != request.transaction_id
                    ):
                        raise ValueError("Evidence does not match the source or transaction")
                except (ValidationError, ValueError, TypeError) as exc:
                    raise ToolExecutionError(
                        "The tool returned invalid or mismatched evidence."
                    ) from exc
                memory.results[action.tool] = result
                # New IDs alone do not mean new evidence. Compare the structured facts.
                keys = {
                    (
                        result.source.value,
                        json.dumps(record.facts or {"summary": record.summary}, sort_keys=True),
                    )
                    for record in result.records
                }
                new_evidence = bool(keys - memory.evidence_keys)
                memory.evidence_keys.update(keys)
                status = result.status
                observation = "; ".join(r.summary for r in result.records) or "No matching records."
                references = [r.reference for r in result.records]
            except ToolExecutionError as exc:
                new_evidence = False
                status = "error"
                observation = str(exc)
            except Exception:
                new_evidence = False
                status = "error"
                observation = "The tool could not finish. No evidence was accepted."
            memory.no_new_evidence_count = 0 if new_evidence else memory.no_new_evidence_count + 1
            entry = AuditEntry(
                call_number=call_number,
                called_at=called_at,
                tool=action.tool,
                source=TOOL_SOURCES[action.tool],
                arguments=arguments,
                reason=action.reason,
                decision_references=list(action.evidence_references),
                status=status,
                observation=observation,
                record_references=references,
                new_evidence=new_evidence,
                elapsed_ms=round((monotonic() - started) * 1000, 2),
            )
            memory.audit.append(entry)
            if on_progress:
                on_progress(entry)

    def _report(
        self,
        request: InvestigationRequest,
        memory: InvestigationMemory,
        skipped_checks: list[SkippedCheck],
        status: str,
        stop_reason: str,
    ) -> InvestigationReport:
        # Monitoring and old cases cannot locate a specific request failure on their own.
        logs = memory.results.get(ToolName.LOGS)
        traces = memory.results.get(ToolName.TRACES)
        log_references = (
            {
                record.reference
                for record in logs.records
                if record.facts.get("http_status") == 503
                and record.facts.get("service") == "settlement"
            }
            if logs
            else set()
        )
        trace_references = (
            {
                record.reference
                for record in traces.records
                if record.facts.get("service") == "settlement"
                and record.facts.get("successful_response") is False
                and record.facts.get("retries_exhausted") is True
            }
            if traces
            else set()
        )
        logs_support, traces_support = bool(log_references), bool(trace_references)
        independent_references = any(
            log_reference != trace_reference
            for log_reference in log_references
            for trace_reference in trace_references
        )
        supported = independent_references and stop_reason != "conflicting_evidence"
        possible = logs_support or traces_support
        if status == "finished" and request.scope == "full" and not supported:
            status = "paused"
            stop_reason = "insufficient_evidence"
        unsupported: list[str] = []
        if memory.has_facts(ToolName.MONITORING, service="settlement", application_healthy=True):
            unsupported.append("A failure of the whole application is not supported by monitoring.")
        if memory.has_facts(ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True):
            unsupported.append("A recent deployment problem is not supported; it is not ruled out.")
        next_steps = ["An engineer must confirm root cause and approve any knowledge-base save."]
        if supported:
            next_steps.insert(
                0, "Check settlement-service recovery and whether manual replay is safe."
            )
        assessments = assess_causes(memory)
        broad = next((item for item in assessments if item.cause_id == "broad_service_issue"), None)
        open_causes = [
            item.explanation
            for item in assessments
            if item.status in ("possible", "needs_evidence")
        ]
        if possible and not open_causes:
            open_causes = ["Why the settlement service failed still needs engineer investigation."]
        diagnostic_next = [
            item.next_check
            for item in assessments
            if item.priority == "high" and item.status != "less_supported"
        ]
        for check in reversed(list(dict.fromkeys(diagnostic_next))):
            if check not in next_steps:
                next_steps.insert(0, check)
        if status == "paused":
            next_steps.insert(
                0, "Ask an engineer whether to continue; do not reset the budget here."
            )
        if possible and not supported:
            open_causes.insert(
                0, "Possible settlement-call failure; two-source support is missing."
            )
        return InvestigationReport(
            run_id=str(uuid4()),
            mode=request.mode,
            use_rag=request.use_rag,
            skipped_checks=skipped_checks,
            planner=getattr(self.planner, "name", "demo"),
            model=getattr(self.planner, "model", None),
            model_calls=getattr(self.planner, "calls", 0),
            model_usage=getattr(self.planner, "usage", ModelUsage()),
            model_response_ids=list(getattr(self.planner, "response_ids", [])),
            retrieval=memory.results[ToolName.SIMILAR].retrieval
            if ToolName.SIMILAR in memory.results
            else None,
            historical_context=memory.results[ToolName.SIMILAR].records
            if ToolName.SIMILAR in memory.results
            else [],
            incident_id=request.incident_id,
            transaction_id=request.transaction_id,
            status=status,
            stop_reason=stop_reason,
            calls_used=len(memory.audit),
            call_limit=self.limits.max_tool_calls,
            supported_failure_point="Settlement-service call" if supported else None,
            possible_failure_point="Settlement-service call"
            if possible and not supported
            else None,
            likely_cause=(
                "Possible shared settlement-service issue; root cause is not confirmed."
                if broad
                and broad.status == "possible"
                and possible
                and stop_reason != "conflicting_evidence"
                else None
            ),
            open_possible_causes=open_causes,
            cause_assessments=assessments,
            unsupported_explanations=unsupported,
            missing_sources=[
                s
                for s in Source
                if not (not request.use_rag and request.mode != "legacy" and s == Source.HISTORY)
                and not _source_is_relevant(memory, s)
            ],
            next_steps=next_steps,
            audit=memory.audit,
        )
