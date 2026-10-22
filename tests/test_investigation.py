"""Check the boundaries that keep findings and tool usage honest."""

import pytest
from conftest import FakeGateway, result_for
from incident_client.investigation import InvestigationEngine
from incident_client.policy import Action, InvestigationLimits
from incident_common.models import InvestigationRequest, Source, ToolName
from pydantic import ValidationError


async def test_supported_finding_keeps_engineer_control(results):
    gateway = FakeGateway(results)
    progress = []
    report = await InvestigationEngine(gateway).run(
        InvestigationRequest(), on_progress=progress.append
    )
    assert report.status == "finished"
    assert report.calls_used == 6
    assert report.supported_failure_point == "Settlement-service call"
    assert report.possible_failure_point is None
    assert report.likely_cause and "not confirmed" in report.likely_cause
    assert not report.root_cause_confirmed
    assert not report.knowledge_base_saved
    assert not report.production_changes_made
    assert len(report.unsupported_explanations) == 2
    assert any("engineer" in step.lower() for step in report.next_steps)
    assert progress == report.audit
    assert [entry.call_number for entry in report.audit] == list(range(1, 7))
    assert all(entry.reason and entry.record_references for entry in report.audit)


async def test_status_question_stops_after_one_call(results):
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway).run(
        InvestigationRequest(transaction_id="TX000001", scope="status")
    )
    assert report.status == "finished"
    assert report.stop_reason == "request_answered"
    assert report.calls_used == 1
    assert gateway.calls == [ToolName.TRANSACTION]
    assert report.audit[0].arguments == {"transaction_id": "TX000001"}


async def test_retries_consume_the_same_eight_call_budget(results):
    gateway = FakeGateway(results, fail_first={ToolName.LOGS, ToolName.TRACES, ToolName.MONITORING})
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.status == "paused"
    assert report.stop_reason == "call_budget"
    assert report.calls_used == len(gateway.calls) == 8
    assert sum(entry.status == "error" for entry in report.audit) == 3
    assert ToolName.HISTORY not in gateway.calls
    assert Source.HISTORY in report.missing_sources
    assert any("whether to continue" in step for step in report.next_steps)


async def test_two_consecutive_failures_pause_before_another_retry(results):
    gateway = FakeGateway(results, always_fail={ToolName.LOGS})
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.status == "paused"
    assert report.stop_reason == "no_new_evidence"
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS, ToolName.LOGS]
    assert [entry.status for entry in report.audit] == ["found", "error", "error"]
    assert report.supported_failure_point is None


async def test_empty_search_is_not_retried_and_one_source_stays_possible(results):
    results[ToolName.TRACES] = result_for(ToolName.TRACES, empty=True)
    results[ToolName.MONITORING] = result_for(ToolName.MONITORING, empty=True)
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.stop_reason == "no_new_evidence"
    assert gateway.calls == [
        ToolName.TRANSACTION,
        ToolName.LOGS,
        ToolName.TRACES,
        ToolName.MONITORING,
    ]
    assert [entry.status for entry in report.audit][-2:] == ["empty", "empty"]
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert report.open_possible_causes
    assert not report.unsupported_explanations


async def test_one_missing_source_needs_guidance_even_after_other_checks_finish(results):
    results[ToolName.TRACES] = result_for(ToolName.TRACES, empty=True)
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.calls_used == 6
    assert report.status == "paused"
    assert report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert any("whether to continue" in step for step in report.next_steps)


async def test_completed_tool_cannot_be_repeated_to_generate_new_record_ids(results):
    class ChangingIdsGateway(FakeGateway):
        async def call_tool(self, tool, transaction_id):
            result = await super().call_tool(tool, transaction_id)
            # A new record identifier alone does not provide a new finding.
            records = [
                record.model_copy(update={"record_id": f"id-{len(self.calls)}"})
                for record in result.records
            ]
            return result.model_copy(update={"records": records})

    class RepeatLogsPlanner:
        def choose(self, request, memory):
            tool = ToolName.TRANSACTION if not memory.audit else ToolName.LOGS
            return Action(tool, "Synthetic repeated-record scenario.")

    gateway = ChangingIdsGateway(results)
    report = await InvestigationEngine(gateway, planner=RepeatLogsPlanner()).run(
        InvestigationRequest()
    )
    assert report.stop_reason == "planner_error"
    assert report.calls_used == 2
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS]
    assert report.supported_failure_point is None


async def test_local_error_changes_the_next_tool(results):
    results[ToolName.LOGS] = result_for(ToolName.LOGS, local_exception=True)
    gateway = FakeGateway(results)
    await InvestigationEngine(gateway).run(InvestigationRequest())
    assert gateway.calls[:4] == [
        ToolName.TRANSACTION,
        ToolName.LOGS,
        ToolName.DEPLOYMENTS,
        ToolName.TRACES,
    ]


async def test_conflicting_trace_does_not_support_the_failure_point(results):
    results[ToolName.TRACES] = result_for(
        ToolName.TRACES, service="settlement", successful_response=True
    )
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.status == "paused"
    assert report.stop_reason == "conflicting_evidence"
    assert report.calls_used == 3
    assert report.supported_failure_point is None


async def test_successful_unrelated_span_does_not_hide_failed_settlement_span(results):
    traces = results[ToolName.TRACES]
    unrelated = traces.records[0].model_copy(
        update={
            "record_id": "billing-span",
            "reference": "synthetic://traces/TX784521/billing-span",
            "facts": {"service": "billing", "successful_response": True},
        }
    )
    results[ToolName.TRACES] = traces.model_copy(update={"records": [unrelated, *traces.records]})
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.status == "finished"
    assert report.supported_failure_point == "Settlement-service call"


def test_validated_limits_cannot_be_changed_after_creation():
    limits = InvestigationLimits()
    with pytest.raises(ValidationError):
        limits.max_tool_calls = 12


@pytest.mark.parametrize(
    "limits",
    [
        {"max_tool_calls": 9},
        {"max_tool_calls": 0},
        {"max_retries_per_tool": 3},
        {"max_retries_per_tool": -1},
        {"max_steps_without_evidence": 3},
    ],
)
def test_limits_cannot_expand_safety_boundaries(limits):
    with pytest.raises(ValidationError):
        InvestigationLimits(**limits)


@pytest.mark.parametrize(
    "request_data",
    [
        {"transaction_id": "TX784521; DROP TABLE transactions"},
        {"transaction_id": "TX12345"},
        {"incident_id": "INC-1234"},
        {"scope": "change-production"},
    ],
)
def test_invalid_request_is_rejected_before_tool_usage(request_data):
    with pytest.raises(ValidationError):
        InvestigationRequest(**request_data)
