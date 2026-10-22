"""The loop enforces its safety rules even with an incorrect planner or gateway."""

from types import SimpleNamespace

import pytest
from conftest import FakeGateway, result_for
from incident_client.investigation import InvestigationEngine
from incident_client.policy import Action
from incident_common.models import InvestigationRequest, Source, ToolName


class ScriptedPlanner:
    def __init__(self, *actions):
        self.actions = iter(actions)

    def choose(self, request, memory):
        return next(self.actions, None)


async def test_premature_planner_stop_does_not_claim_the_request_was_answered(results):
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway, planner=ScriptedPlanner()).run(
        InvestigationRequest()
    )
    assert report.status == "paused"
    assert report.stop_reason == "planner_error"
    assert report.planner_error and "required checks" in report.planner_error
    assert report.calls_used == 0
    assert not gateway.calls


@pytest.mark.parametrize(
    "mode,tool",
    [
        ("without-rag", ToolName.HISTORY),
        ("without-rag", ToolName.SIMILAR),
        ("with-rag", ToolName.SIMILAR),
        ("legacy", ToolName.TRACES),
    ],
)
async def test_planner_cannot_skip_setup_or_use_tools_disabled_for_the_mode(results, mode, tool):
    gateway = FakeGateway(results)
    report = await InvestigationEngine(
        gateway, planner=ScriptedPlanner(Action(tool, "Skip the required checks."))
    ).run(InvestigationRequest(mode=mode))
    assert report.status == "paused"
    assert report.stop_reason == "planner_error"
    assert report.calls_used == 0
    assert not gateway.calls


@pytest.mark.parametrize(
    "decision",
    [
        SimpleNamespace(tool=ToolName.TRANSACTION, reason="A decision with the wrong type."),
        Action("get_transaction", "An unvalidated tool name."),
        Action(ToolName.TRANSACTION, " "),
        Action(ToolName.TRANSACTION, "x" * 1001),
        Action(ToolName.TRANSACTION, "Read status.\x1b[31m"),
        Action(ToolName.TRANSACTION, "Invent a source.", ("synthetic://invented",)),
        Action(ToolName.TRANSACTION, "Use malformed references.", "synthetic://invented"),
    ],
)
async def test_invalid_action_is_rejected_before_executing_a_tool(results, decision):
    gateway = FakeGateway(results)
    report = await InvestigationEngine(gateway, planner=ScriptedPlanner(decision)).run(
        InvestigationRequest()
    )
    assert report.stop_reason == "planner_error"
    assert not gateway.calls
    assert report.calls_used == 0


async def test_planner_cannot_repeat_a_completed_tool(results):
    gateway = FakeGateway(results)
    action = Action(ToolName.TRANSACTION, "Read the transaction.")
    report = await InvestigationEngine(gateway, planner=ScriptedPlanner(action, action)).run(
        InvestigationRequest()
    )
    assert report.stop_reason == "planner_error"
    assert gateway.calls == [ToolName.TRANSACTION]
    assert report.calls_used == 1


async def test_known_decision_source_is_accepted_but_invented_source_is_rejected(results):
    transaction_reference = results[ToolName.TRANSACTION].records[0].reference
    gateway = FakeGateway(results)
    planner = ScriptedPlanner(
        Action(ToolName.TRANSACTION, "Read the transaction."),
        Action(ToolName.LOGS, "Read its errors.", (transaction_reference,)),
        Action(ToolName.TRACES, "Check the request path.", ("synthetic://not-returned",)),
    )
    report = await InvestigationEngine(gateway, planner=planner).run(InvestigationRequest())
    assert report.stop_reason == "planner_error"
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS]
    assert report.audit[1].decision_references == [transaction_reference]


@pytest.mark.parametrize("bad_field", ["source", "transaction_id", "records", "raw"])
async def test_invalid_gateway_evidence_is_counted_as_failure_and_never_used(results, bad_field):
    class InvalidGateway(FakeGateway):
        async def call_tool(self, tool, transaction_id):
            result = await super().call_tool(tool, transaction_id)
            if bad_field == "raw":
                return result.model_dump()
            replacement = {
                "source": Source.HISTORY,
                "transaction_id": "TX000001",
                "records": [],
            }[bad_field]
            return result.model_copy(update={bad_field: replacement})

    gateway = InvalidGateway(results)
    progress = []
    report = await InvestigationEngine(gateway).run(
        InvestigationRequest(), on_progress=progress.append
    )
    assert report.status == "paused"
    assert report.stop_reason == "no_new_evidence"
    assert report.calls_used == 2
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.TRANSACTION]
    assert [entry.status for entry in report.audit] == ["error", "error"]
    assert not any(entry.new_evidence for entry in report.audit)
    assert not any(entry.record_references for entry in report.audit)
    assert progress == report.audit
    assert report.supported_failure_point is None
    assert Source.TRANSACTIONS in report.missing_sources


async def test_two_different_tools_cannot_reuse_one_reference_as_two_sources(results):
    trace = results[ToolName.TRACES]
    reused_reference = results[ToolName.LOGS].records[0].reference
    results[ToolName.TRACES] = trace.model_copy(
        update={"records": [trace.records[0].model_copy(update={"reference": reused_reference})]}
    )
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.status == "paused"
    assert report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed


async def test_a_separate_trace_reference_can_supply_independent_support(results):
    trace = results[ToolName.TRACES]
    reused = trace.records[0].model_copy(
        update={
            "record_id": "copied-log-reference",
            "reference": results[ToolName.LOGS].records[0].reference,
        }
    )
    results[ToolName.TRACES] = trace.model_copy(update={"records": [reused, *trace.records]})
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.status == "finished"
    assert report.supported_failure_point == "Settlement-service call"


async def test_unrelated_records_do_not_complete_a_full_investigation(results):
    for tool in (ToolName.LOGS, ToolName.TRACES, ToolName.MONITORING, ToolName.DEPLOYMENTS):
        results[tool] = result_for(tool, service="another-service", unrelated=True)
    report = await InvestigationEngine(FakeGateway(results)).run(
        InvestigationRequest(mode="without-rag")
    )
    assert report.calls_used == 5
    assert report.status == "paused"
    assert report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point is None
    assert set(report.missing_sources) == {
        Source.LOGS,
        Source.TRACES,
        Source.MONITORING,
        Source.DEPLOYMENTS,
    }
    assert any("whether to continue" in step for step in report.next_steps)


async def test_found_transaction_without_status_does_not_answer_a_status_question(results):
    results[ToolName.TRANSACTION] = result_for(ToolName.TRANSACTION, unrelated=True)
    report = await InvestigationEngine(FakeGateway(results)).run(
        InvestigationRequest(scope="status")
    )
    assert report.status == "paused"
    assert report.stop_reason == "insufficient_evidence"
    assert report.calls_used == 1
    assert Source.TRANSACTIONS in report.missing_sources


async def test_unexpected_planner_failure_returns_a_safe_paused_report(results):
    class BrokenPlanner:
        def choose(self, request, memory):
            raise RuntimeError("sk-private-test-key should not be displayed")

    report = await InvestigationEngine(FakeGateway(results), planner=BrokenPlanner()).run(
        InvestigationRequest()
    )
    assert report.stop_reason == "planner_error"
    assert report.planner_error == "The planner could not produce a safe next check."
    assert "sk-private" not in report.model_dump_json()


async def test_unexpected_gateway_error_is_counted_without_exposing_private_text(results):
    class BrokenGateway(FakeGateway):
        async def call_tool(self, tool, transaction_id):
            await super().call_tool(tool, transaction_id)
            raise RuntimeError("synthetic-private-connector-detail")

    gateway = BrokenGateway(results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.status == "paused" and report.stop_reason == "no_new_evidence"
    assert report.calls_used == 2
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.TRANSACTION]
    assert all(entry.status == "error" for entry in report.audit)
    assert all(not entry.record_references for entry in report.audit)
    assert "synthetic-private-connector-detail" not in report.model_dump_json()
