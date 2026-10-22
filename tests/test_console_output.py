"""Short console output keeps important facts; verbose and JSON retain full evidence."""

import json
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest
from incident_client import main as cli
from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import PlannerError
from incident_client.policy import Action, DemoPlanner
from incident_client.reporting import render_cause_assessments, render_summary
from incident_common.models import InvestigationRequest, ToolName
from test_rag_flow import RagGateway, historical_result


class PausedAPIPlanner:
    """Fail the first model request after deterministic setup, like the reported run."""

    name = "openai"
    model = "gpt-4.1-mini-2025-04-14"
    calls = 1
    response_ids = []

    def choose(self, request, memory):
        if ToolName.SIMILAR in memory.results:
            raise PlannerError("The API planner request failed. Check the API setup and try again.")
        action = DemoPlanner().choose(request, memory)
        references = tuple(
            record.reference
            for tool in (ToolName.TRANSACTION, ToolName.LOGS)
            if tool in memory.results
            for record in memory.results[tool].records
        )
        return Action(action.tool, action.reason, references)

    async def aclose(self):
        pass


@pytest.fixture
async def paused_report(results):
    results[ToolName.SIMILAR] = historical_result()
    return await InvestigationEngine(RagGateway(results), planner=PausedAPIPlanner()).run(
        InvestigationRequest(mode="with-rag")
    )


def cli_args(*, verbose=False, json_output=False):
    return Namespace(
        planner="openai",
        list_tools=False,
        json=json_output,
        verbose=verbose,
        incident_id="INC-10234",
        transaction_id="TX784521",
        scope="full",
        mode="with-rag",
    )


def install_console_gateway(monkeypatch, results):
    class ConsoleGateway(RagGateway):
        async def list_tools(self):
            return [tool.value for tool in ToolName]

    @asynccontextmanager
    async def connect(_timeout):
        yield ConsoleGateway(results)

    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "OpenAIPlanner", PausedAPIPlanner)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key-never-sent")


def test_compact_paused_summary_keeps_api_failure_and_uncertain_finding(paused_report):
    summary = render_summary(paused_report)
    text = summary.lower()
    assert "INC-10234" in summary and "TX784521" in summary
    assert ("with-rag" in text or "with rag" in text) and "paused" in text and "3/8" in text
    assert "api" in text and ("failed" in text or "issue" in text)
    assert "not established" in text
    assert "possible" in text and "settlement" in text
    assert "trace" in text and ("monitoring" in text or "health" in text)
    assert "engineer" in text
    assert "root cause" in text and ("not confirmed" in text or "confirmed: no" in text)
    assert "approve" in text or "approval" in text
    assert "planner_error" not in text
    assert len(summary.splitlines()) <= 22


def test_compact_summary_shows_history_once_without_raw_evidence_dump(paused_report):
    summary = render_summary(paused_report)
    assert summary.count("INC-09876") == 1
    assert "synthetic://" not in summary
    assert "0.800" not in summary and "similarity:" not in summary.lower()
    for repeated_heading in ("Supporting evidence:", "Evidence against:", "Missing evidence:"):
        assert repeated_heading not in summary
    assert "difference demonstrated" not in summary.lower()
    assert "changed the next" not in summary.lower()
    full_causes = render_cause_assessments(paused_report.cause_assessments)
    assert len(summary) < len(full_causes)


async def test_compact_finished_summary_preserves_supported_point_and_manual_boundary(results):
    report = await InvestigationEngine(RagGateway(results)).run(
        InvestigationRequest(mode="without-rag")
    )
    summary = render_summary(report)
    text = summary.lower()
    assert "finished" in text and "5/8" in text
    assert "supported failure point: settlement-service call" in text
    assert "root cause" in text and ("not confirmed" in text or "confirmed: no" in text)
    assert "manual" in text or "production" in text
    assert "approve" in text or "approval" in text
    assert "synthetic://" not in summary


def test_compact_progress_is_one_line_without_full_reason_or_sources(paused_report, capsys):
    entry = paused_report.audit[2].model_copy(
        update={
            "observation": (
                "Past synthetic INC-09876: Billing settlement requests return HTTP 503. "
                "Check settlement-service errors and application health before exploring "
                "a recent application change. The past cause does not confirm the current cause."
            )
        }
    )
    cli._progress(entry)
    captured = capsys.readouterr()
    assert not captured.out
    assert len(captured.err.strip().splitlines()) == 1
    assert "[3/8]" in captured.err
    assert "Why:" not in captured.err and "Decision sources:" not in captured.err
    assert "synthetic://" not in captured.err
    assert len(captured.err.strip()) <= 220


def test_verbose_progress_retains_complete_reason_observation_and_sources(paused_report, capsys):
    entry = paused_report.audit[2]
    cli._progress(entry, verbose=True)
    text = capsys.readouterr().err
    assert entry.reason in text and entry.observation in text
    assert "Why:" in text and "Found:" in text and "Decision sources:" in text
    assert all(reference in text for reference in entry.decision_references)


async def test_json_output_keeps_full_evidence_and_has_no_progress(results, monkeypatch, capsys):
    results[ToolName.SIMILAR] = historical_result()
    install_console_gateway(monkeypatch, results)
    await cli._run(cli_args(json_output=True), Settings())
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["status"] == "paused" and report["stop_reason"] == "planner_error"
    assert report["calls_used"] == 3
    assert report["planner_error"]
    assert report["supported_failure_point"] is None
    assert report["retrieval"]["matches"][0]["incident_id"] == "INC-09876"
    assert report["cause_assessments"][0]["supporting_evidence"]
    assert report["cause_assessments"][0]["missing_evidence"]
    assert report["audit"][2]["decision_references"]
    assert not any(
        report[name]
        for name in ("root_cause_confirmed", "knowledge_base_saved", "production_changes_made")
    )


async def test_default_cli_uses_short_output_without_a_verbose_argument(
    results, monkeypatch, capsys
):
    results[ToolName.SIMILAR] = historical_result()
    install_console_gateway(monkeypatch, results)
    args = cli_args()
    del args.verbose
    await cli._run(args, Settings())
    captured = capsys.readouterr()
    assert "paused" in captured.out.lower() and "3/8" in captured.out
    assert "api" in captured.out.lower()
    assert "Supporting evidence:" not in captured.out
    assert "synthetic://" not in captured.out + captured.err
    assert "Why:" not in captured.err
    assert len(captured.out.splitlines()) <= 23


async def test_verbose_cli_keeps_the_detailed_cause_report(results, monkeypatch, capsys):
    results[ToolName.SIMILAR] = historical_result()
    install_console_gateway(monkeypatch, results)
    await cli._run(cli_args(verbose=True), Settings())
    captured = capsys.readouterr()
    assert "Supporting evidence:" in captured.out
    assert "Evidence against:" in captured.out
    assert "Missing evidence:" in captured.out
    assert "synthetic://" in captured.out
    assert "Why:" in captured.err and "Decision sources:" in captured.err
