"""RAG toggles retrieval without spending calls on skipped checks."""

import json
import sys
from unittest.mock import AsyncMock

import pytest
from incident_client import main as cli
from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.policy import DemoPlanner, InvestigationLimits
from incident_client.reporting import render_progress
from incident_common.models import InvestigationRequest, SkippedCheck, ToolName
from pydantic import ValidationError
from test_console_output import cli_args, install_console_gateway
from test_rag_flow import RagGateway, historical_result


@pytest.mark.parametrize("enabled,mode", [(True, "with-rag"), (False, "without-rag")])
def test_boolean_and_existing_modes_resolve_to_same_request(enabled, mode):
    assert InvestigationRequest(use_rag=enabled) == InvestigationRequest(mode=mode)
    assert InvestigationRequest(use_rag=enabled).use_rag is enabled


@pytest.mark.parametrize("value", ["False", "True", 0, 1, None])
def test_python_flag_requires_a_real_boolean(value):
    with pytest.raises(ValidationError):
        InvestigationRequest(use_rag=value)


@pytest.mark.parametrize("enabled,mode", [(True, "without-rag"), (False, "with-rag")])
def test_conflicting_controls_are_rejected(enabled, mode):
    with pytest.raises(ValidationError, match="must agree"):
        InvestigationRequest(use_rag=enabled, mode=mode)


@pytest.mark.parametrize("verbose", [False, True])
async def test_skip_is_between_logs_and_trace_without_consuming_a_call(results, verbose):
    events = []
    gateway = RagGateway(results)
    planner = DemoPlanner()
    report = await InvestigationEngine(gateway, planner=planner).run(
        InvestigationRequest(use_rag=False), on_progress=events.append
    )
    skip = report.skipped_checks[0]
    assert isinstance(events[2], SkippedCheck)
    assert events[2] == skip
    assert render_progress(skip, verbose=verbose) == (
        "[Skipped] Similar incidents: RAG disabled; knowledge database not checked."
    )
    assert events[3].tool == ToolName.TRACES and events[3].call_number == 3
    assert report.calls_used == len(gateway.calls) == len(report.audit) == 5
    assert [entry.call_number for entry in report.audit] == [1, 2, 3, 4, 5]
    assert not {ToolName.HISTORY, ToolName.SIMILAR}.intersection(gateway.calls)
    assert report.retrieval is None and report.historical_context == []
    assert report.use_rag is False
    assert report.status == "finished"


async def test_enabled_rag_uses_same_current_tools_and_rules(results):
    results[ToolName.SIMILAR] = historical_result()
    reports, gateways = [], []
    for enabled in (False, True):
        gateway = RagGateway(results)
        gateways.append(gateway)
        reports.append(
            await InvestigationEngine(gateway, planner=DemoPlanner()).run(
                InvestigationRequest(use_rag=enabled)
            )
        )
    assert reports[1].use_rag is True
    assert reports[1].skipped_checks == []
    assert gateways[1].calls.count(ToolName.SIMILAR) == 1
    assert set(gateways[0].calls) == set(gateways[1].calls) - {ToolName.SIMILAR}
    assert reports[0].supported_failure_point == reports[1].supported_failure_point
    assert reports[0].call_limit == reports[1].call_limit == 8
    assert reports[0].planner == reports[1].planner == "demo"


async def test_skip_does_not_use_the_last_available_call(results):
    events = []
    report = await InvestigationEngine(
        RagGateway(results), limits=InvestigationLimits(max_tool_calls=3)
    ).run(InvestigationRequest(use_rag=False), on_progress=events.append)
    assert report.calls_used == 3
    assert report.audit[-1].tool == ToolName.TRACES
    assert len(report.skipped_checks) == 1
    assert report.stop_reason == "call_budget"


@pytest.mark.parametrize("scope,limit", [("status", 8), ("full", 1)])
async def test_no_skip_before_retrieval_stage(results, scope, limit):
    events = []
    report = await InvestigationEngine(
        RagGateway(results), limits=InvestigationLimits(max_tool_calls=limit)
    ).run(InvestigationRequest(use_rag=False, scope=scope), on_progress=events.append)
    assert report.skipped_checks == []
    assert not any(isinstance(event, SkippedCheck) for event in events)


async def test_disabled_json_exposes_skip_without_numbered_audit_entry(
    results, monkeypatch, capsys
):
    install_console_gateway(monkeypatch, results)
    args = cli_args(json_output=True)
    args.mode, args.use_rag, args.planner = None, False, "demo"
    assert await cli._run(args, Settings()) == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert output.err == ""
    assert data["use_rag"] is False
    assert len(data["skipped_checks"]) == 1
    assert "call_number" not in data["skipped_checks"][0]
    assert data["calls_used"] == len(data["audit"]) == 5


@pytest.mark.parametrize("flag,enabled", [("--use-rag", True), ("--no-use-rag", False)])
def test_cli_boolean_keeps_the_same_default_planner(flag, enabled, monkeypatch):
    runner = AsyncMock(return_value=0)
    monkeypatch.setattr(cli, "_run", runner)
    monkeypatch.setattr(sys, "argv", ["incident-client", flag])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    args = runner.call_args.args[0]
    assert args.use_rag is enabled
    assert args.planner == "openai"
