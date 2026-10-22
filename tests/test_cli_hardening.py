"""Check terminal-only output, safe process outcomes, and bounded discovery."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from incident_client import compare
from incident_client import main as cli
from incident_client.config import Settings
from incident_client.mcp_client import MCPGateway, ToolExecutionError
from incident_common.models import InvestigationReport, ToolName
from test_console_output import cli_args, install_console_gateway
from test_rag_flow import historical_result


@pytest.mark.parametrize("json_output", [False, True])
async def test_paused_run_returns_two_with_terminal_output_without_files(
    results, monkeypatch, capsys, tmp_path, json_output
):
    results[ToolName.SIMILAR] = historical_result()
    install_console_gateway(monkeypatch, results)
    monkeypatch.chdir(tmp_path)

    code = await cli._run(cli_args(json_output=json_output), Settings())

    console = capsys.readouterr()
    assert code == 2
    assert list(tmp_path.iterdir()) == []
    if json_output:
        result = json.loads(console.out)
        assert result["status"] == "paused" and result["calls_used"] == 3
        assert result["audit"][2]["decision_references"]
        assert console.err == ""
    else:
        assert "Paused" in console.out
        assert "synthetic://" not in console.out


async def test_finished_status_request_returns_zero(results, monkeypatch, capsys):
    install_console_gateway(monkeypatch, results)
    args = cli_args(json_output=True)
    args.planner, args.mode, args.scope = "demo", "without-rag", "status"

    assert await cli._run(args, Settings()) == 0
    assert json.loads(capsys.readouterr().out)["calls_used"] == 1


def test_client_main_returns_the_investigation_exit_code(monkeypatch):
    async def paused(args, settings):
        return 2

    monkeypatch.setattr(cli, "_run", paused)
    monkeypatch.setattr(sys, "argv", ["incident-client"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_unexpected_client_error_does_not_print_private_exception(monkeypatch, caplog):
    async def fail(*_args):
        raise RuntimeError("synthetic-private-api-key")

    monkeypatch.setattr(cli, "_run", fail)
    monkeypatch.setattr(sys, "argv", ["incident-client"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 1
    assert "synthetic-private-api-key" not in caplog.text
    assert "local setup" in caplog.text


async def test_discovery_collects_pages_within_one_deadline():
    session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=[SimpleNamespace(name="get_transaction")], nextCursor="two"),
                SimpleNamespace(tools=[SimpleNamespace(name="search_logs")], nextCursor=None),
            ]
        )
    )
    assert await MCPGateway(session, 1).list_tools() == ["get_transaction", "search_logs"]
    assert session.list_tools.await_args_list[1].kwargs == {"cursor": "two"}


@pytest.mark.parametrize("problem", ["cursor_loop", "duplicate", "too_many"])
async def test_discovery_rejects_unbounded_or_repeated_pages(problem):
    if problem == "cursor_loop":
        pages = [
            SimpleNamespace(tools=[SimpleNamespace(name="first")], nextCursor="repeat"),
            SimpleNamespace(tools=[SimpleNamespace(name="second")], nextCursor="repeat"),
        ]
    elif problem == "duplicate":
        pages = [SimpleNamespace(tools=[SimpleNamespace(name="same")] * 2, nextCursor=None)]
    else:
        pages = [
            SimpleNamespace(
                tools=[SimpleNamespace(name=str(index)) for index in range(51)], nextCursor=None
            )
        ]
    session = SimpleNamespace(list_tools=AsyncMock(side_effect=pages))
    with pytest.raises(ToolExecutionError, match="limits|repeated"):
        await MCPGateway(session, 1).list_tools()


async def test_discovery_timeout_is_bounded_and_safe():
    async def hanging_discovery(**_kwargs):
        await asyncio.sleep(60)

    session = SimpleNamespace(list_tools=AsyncMock(side_effect=hanging_discovery))
    with pytest.raises(ToolExecutionError, match="timed out"):
        await MCPGateway(session, 0.005).list_tools()
    session.list_tools.assert_awaited_once()


@pytest.mark.parametrize("paused", [False, True])
def test_comparison_displays_results_without_creating_files(paused, monkeypatch, tmp_path, capsys):
    reports = [
        InvestigationReport(
            run_id=f"synthetic-{mode}",
            mode=mode,
            incident_id="INC-10234",
            transaction_id="TX784521",
            status="paused" if paused else "finished",
            stop_reason="planner_error" if paused else "request_answered",
            calls_used=3 if paused else 5,
        )
        for mode in ("without-rag", "with-rag")
    ]
    runner = AsyncMock(return_value=tuple(reports))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "run_comparison", runner)
    monkeypatch.setattr(sys, "argv", ["incident-compare", "--planner", "demo"])

    if paused:
        with pytest.raises(SystemExit) as result:
            compare.main()
        assert result.value.code == 2
    else:
        compare.main()

    runner.assert_awaited_once_with("demo", verbose=False)
    assert list(tmp_path.iterdir()) == []
    output = capsys.readouterr()
    assert "Without RAG:" in output.out and "with RAG:" in output.out
    assert output.err == ""
