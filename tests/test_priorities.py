"""Check visible next-step priorities and reports without API calls or stored writes."""

import json
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest
from conftest import FakeGateway, result_for
from incident_client import main as cli
from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner
from incident_client.memory import InvestigationMemory
from incident_client.policy import DemoPlanner, prioritize_checks
from incident_client.reporting import render_cause_assessments
from incident_common.models import InvestigationRequest, ToolName
from test_llm_planner import decision_for, fake_client
from test_rag_flow import RagGateway


def observed_memory(*, history=False, **observations):
    memory = InvestigationMemory(
        results={
            ToolName.TRANSACTION: result_for(
                ToolName.TRANSACTION, status="SETTLEMENT_SENT", age_hours=26
            ),
            ToolName.LOGS: result_for(ToolName.LOGS, service="settlement", http_status=503),
        }
    )
    memory.results.update({ToolName(name): result for name, result in observations.items()})
    if history:
        memory.results[ToolName.SIMILAR] = result_for(
            ToolName.SIMILAR,
            service="settlement",
            historical=True,
            past_incident_id="INC-09876",
            useful_check="get_monitoring",
            check_reason="Check the current service health against the past service errors.",
            past_root_cause="A settlement-service availability issue.",
        )
    return memory


def failed_trace():
    return result_for(
        ToolName.TRACES,
        service="settlement",
        successful_response=False,
        retries_exhausted=True,
    )


def monitoring_result():
    return result_for(
        ToolName.MONITORING,
        service="settlement",
        application_healthy=True,
        elevated_errors=True,
        monitoring_window={
            "window_start": "2026-01-15T09:55:00Z",
            "window_end": "2026-01-15T10:10:00Z",
            "error_started_at": "2026-01-15T10:00:00Z",
            "service": "settlement",
            "checked_transaction_count": 3,
            "affected_transaction_count": 3,
            "successful_transaction_count": 0,
            "observations": [
                {
                    "transaction_id": transaction,
                    "observed_at": time,
                    "outcome": "failed",
                    "http_status": 503,
                    "reference": f"synthetic://monitoring/{transaction}",
                }
                for transaction, time in (
                    ("TX784521", "2026-01-15T10:00:02Z"),
                    ("TX784522", "2026-01-15T10:00:04Z"),
                    ("TX784523", "2026-01-15T10:01:00Z"),
                )
            ],
        },
    )


def test_failed_calls_and_no_related_deployment_prioritize_service_health():
    memory = observed_memory(
        get_trace=failed_trace(),
        get_deployments=result_for(
            ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
        ),
    )
    checks = prioritize_checks(InvestigationRequest(mode="without-rag"), memory)
    assert checks[0].tool == ToolName.MONITORING
    assert checks[0].priority == "high"
    assert "deployment" in checks[0].reason.lower()
    assert "time window" in checks[0].reason
    assert "wider service" in checks[0].reason
    assert "transaction-specific" in checks[0].reason
    assert set(checks[0].tests_causes) == {
        "broad_service_issue",
        "transaction_specific_issue",
    }
    assert set(checks[0].evidence_references) == {
        memory.results[tool].records[0].reference
        for tool in (ToolName.LOGS, ToolName.TRACES, ToolName.DEPLOYMENTS)
    }


def test_monitoring_already_collected_is_reused_before_current_trace():
    memory = observed_memory(history=True, get_monitoring=monitoring_result())
    checks = prioritize_checks(InvestigationRequest(mode="with-rag"), memory)
    assert checks[0].tool == ToolName.TRACES
    assert "already collected" in checks[0].reason
    assert ToolName.MONITORING not in {check.tool for check in checks}
    assert memory.results[ToolName.MONITORING].records[0].reference in checks[0].evidence_references


def test_local_application_error_takes_priority_over_historical_service_hint():
    memory = observed_memory(history=True)
    memory.results[ToolName.LOGS] = result_for(
        ToolName.LOGS, service="billing", local_exception=True
    )
    action = DemoPlanner().choose(InvestigationRequest(mode="with-rag"), memory)
    assert action.tool == ToolName.DEPLOYMENTS
    assert "local error" in action.reason.lower()
    history_reference = memory.results[ToolName.SIMILAR].records[0].reference
    assert history_reference not in action.evidence_references


def test_current_failed_trace_takes_priority_over_historical_deployment_hint():
    memory = observed_memory(history=True, get_trace=failed_trace())
    history = memory.results[ToolName.SIMILAR]
    record = history.records[0]
    memory.results[ToolName.SIMILAR] = history.model_copy(
        update={
            "records": [
                record.model_copy(
                    update={"facts": {**record.facts, "useful_check": "get_deployments"}}
                )
            ]
        }
    )
    checks = prioritize_checks(InvestigationRequest(mode="with-rag"), memory)
    assert checks[0].tool == ToolName.MONITORING
    assert "Logs and the trace" in checks[0].reason
    assert record.reference not in checks[0].evidence_references


def test_empty_deployment_result_does_not_become_no_change_evidence():
    memory = observed_memory(
        get_trace=failed_trace(),
        get_deployments=result_for(ToolName.DEPLOYMENTS, empty=True),
    )
    checks = prioritize_checks(InvestigationRequest(mode="without-rag"), memory)
    assert checks[0].tool == ToolName.MONITORING
    assert "No related deployment" not in checks[0].reason
    assert not any("get_deployments" in reference for reference in checks[0].evidence_references)


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "unrelated-service"])
def test_missing_relevant_monitoring_does_not_invent_a_wider_pattern(empty):
    observed_monitoring = result_for(
        ToolName.MONITORING,
        empty=empty,
        service="billing",
        elevated_errors=True,
    )
    memory = observed_memory(get_monitoring=observed_monitoring)
    checks = prioritize_checks(InvestigationRequest(mode="without-rag"), memory)
    assert checks[0].tool == ToolName.TRACES
    reason = checks[0].reason.lower()
    assert "missing" in reason or "no relevant" in reason
    assert "wider pattern" not in reason
    assert "monitoring" in reason
    assert ToolName.MONITORING not in {check.tool for check in checks}


@pytest.mark.parametrize(
    "completion", [{"check_completed": False}, {}], ids=["incomplete", "unknown"]
)
async def test_incomplete_deployment_check_does_not_count_as_no_related_change(results, completion):
    deployment = result_for(ToolName.DEPLOYMENTS, relevant_change=False, **completion)
    memory = observed_memory(get_trace=failed_trace(), get_deployments=deployment)
    checks = prioritize_checks(InvestigationRequest(mode="without-rag"), memory)
    assert checks[0].tool == ToolName.MONITORING
    assert "no related" not in checks[0].reason.lower()
    assert deployment.records[0].reference not in checks[0].evidence_references

    results[ToolName.DEPLOYMENTS] = deployment
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    recent_change = next(
        cause for cause in report.cause_assessments if cause.cause_id == "recent_deployment"
    )
    assert recent_change.status == "needs_evidence"
    assert not recent_change.opposing_evidence
    assert not any(
        "recent deployment problem is not supported" in item.lower()
        for item in report.unsupported_explanations
    )


async def test_unrelated_monitoring_errors_do_not_create_a_likely_settlement_cause(results):
    results[ToolName.MONITORING] = result_for(
        ToolName.MONITORING,
        service="billing",
        elevated_errors=True,
        application_healthy=True,
    )
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.supported_failure_point == "Settlement-service call"
    assert report.likely_cause is None
    broad = next(
        cause for cause in report.cause_assessments if cause.cause_id == "broad_service_issue"
    )
    assert broad.status == "needs_evidence"
    monitoring_reference = results[ToolName.MONITORING].records[0].reference
    assert all(note.reference != monitoring_reference for note in broad.supporting_evidence)


def test_history_changes_priority_only_with_observed_matching_service_and_citations():
    before = observed_memory()
    after = observed_memory(history=True)
    baseline = prioritize_checks(InvestigationRequest(mode="without-rag"), before)
    retrieved = prioritize_checks(InvestigationRequest(mode="with-rag"), after)
    assert baseline[0].tool == ToolName.TRACES
    assert retrieved[0].tool == ToolName.MONITORING
    assert "INC-09876" in retrieved[0].reason
    assert set(retrieved[0].evidence_references) == {
        after.results[ToolName.LOGS].records[0].reference,
        after.results[ToolName.SIMILAR].records[0].reference,
    }
    for memory, checks in ((before, baseline), (after, retrieved)):
        known = {
            record.reference for result in memory.results.values() for record in result.records
        }
        assert all(set(check.evidence_references) <= known for check in checks)


async def test_api_gets_priorities_and_cause_context_from_current_observations():
    memory = observed_memory(
        get_trace=failed_trace(),
        get_deployments=result_for(
            ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
        ),
    )
    # Leave two useful checks open so this exercises a model request rather than
    # the one-remaining-tool shortcut.
    memory.results.pop(ToolName.TRACES)
    log_reference = memory.results[ToolName.LOGS].records[0].reference
    client = fake_client(decision=decision_for(ToolName.TRACES, log_reference))
    action = await OpenAIPlanner(client=client).choose(
        InvestigationRequest(mode="without-rag"), memory
    )
    assert action.tool == ToolName.TRACES
    payload = json.loads(client.responses.parse.call_args.kwargs["input"][1]["content"])
    assert "mode" not in payload["request"]
    assert {check["tool"] for check in payload["prioritized_checks"]} == {
        ToolName.TRACES,
        ToolName.MONITORING,
    }
    assert all(check["reason"] and check["tests_causes"] for check in payload["prioritized_checks"])
    recent_change = next(
        cause for cause in payload["cause_assessments"] if cause["cause_id"] == "recent_deployment"
    )
    assert recent_change["status"] == "less_supported"
    assert recent_change["opposing_evidence"][0]["reference"] == (
        memory.results[ToolName.DEPLOYMENTS].records[0].reference
    )


async def test_final_report_cites_narrowed_causes_without_confirming_root_cause(
    results,
):
    results[ToolName.MONITORING] = monitoring_result()
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.status == "finished"
    assert report.supported_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed
    assert not report.production_changes_made
    assert not report.knowledge_base_saved
    causes = {cause.cause_id: cause for cause in report.cause_assessments}
    broad = causes["broad_service_issue"]
    assert broad.status == "possible"
    assert broad.priority == "high"
    assert any("TX784522" in item.observation for item in broad.supporting_evidence)
    assert causes["transaction_specific_issue"].status == "less_supported"
    assert causes["recent_deployment"].opposing_evidence
    assert all(cause.missing_evidence and cause.next_check for cause in report.cause_assessments)
    assert all(cause.next_tool is None for cause in report.cause_assessments)
    known = {reference for entry in report.audit for reference in entry.record_references}
    assert all(
        note.reference in known
        for cause in report.cause_assessments
        for note in (*cause.supporting_evidence, *cause.opposing_evidence)
    )


async def test_shared_service_pattern_cannot_replace_missing_current_trace(results):
    results[ToolName.MONITORING] = monitoring_result()
    results[ToolName.TRACES] = result_for(ToolName.TRACES, empty=True)
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    broad = next(
        cause for cause in report.cause_assessments if cause.cause_id == "broad_service_issue"
    )
    assert broad.status == "possible" and broad.priority == "high"
    assert report.status == "paused" and report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed


async def test_unfinished_human_decision_remains_context_without_established_cause(results):
    results[ToolName.SIMILAR] = result_for(
        ToolName.SIMILAR,
        service="settlement",
        historical=True,
        past_incident_id="INC-00121",
        human_decision="Check with the downstream posting and billing team.",
        action_taken="Manual check with the downstream system.",
        outcome="unknown",
        past_root_cause=None,
        root_cause_confirmed=False,
    )
    report = await InvestigationEngine(RagGateway(results)).run(
        InvestigationRequest(mode="with-rag")
    )
    saved = report.historical_context[0]
    assert saved.facts["past_incident_id"] == "INC-00121"
    assert saved.facts["human_decision"] == "Check with the downstream posting and billing team."
    assert saved.facts["outcome"] == "unknown"
    assert not report.root_cause_confirmed
    assert not report.production_changes_made
    historical_reference = saved.reference
    assert all(
        historical_reference != note.reference
        for cause in report.cause_assessments
        for note in cause.supporting_evidence
    )
    assert "connection_pool_issue" not in {cause.cause_id for cause in report.cause_assessments}


async def test_terminal_cause_details_include_evidence_gaps_and_next_check(results):
    results[ToolName.MONITORING] = monitoring_result()
    report = await InvestigationEngine(FakeGateway(results)).run(
        InvestigationRequest(mode="without-rag")
    )
    details = render_cause_assessments(report.cause_assessments).lower()
    assert "supporting" in details
    assert "evidence against" in details
    assert "missing" in details
    assert "next check" in details
    assert "tx784522" in details
    assert results[ToolName.MONITORING].records[0].reference.lower() in details


async def test_console_shows_cause_evidence_and_next_check(results, monkeypatch, capsys):
    results[ToolName.MONITORING] = monitoring_result()

    class ConsoleGateway(FakeGateway):
        async def list_tools(self):
            return [tool.value for tool in ToolName]

    @asynccontextmanager
    async def connect(_timeout):
        yield ConsoleGateway(results)

    monkeypatch.setattr(cli, "connect", connect)
    await cli._run(
        Namespace(
            planner="demo",
            list_tools=False,
            json=False,
            verbose=True,
            incident_id="INC-10234",
            transaction_id="TX784521",
            scope="full",
            mode="without-rag",
        ),
        Settings(),
    )
    output = capsys.readouterr().out.lower()
    assert "supporting evidence" in output
    assert "evidence against" in output
    assert "missing evidence" in output
    assert "next check" in output
    assert "tx784522" in output
    assert "root cause confirmed: no" in output
