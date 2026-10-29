"""Labeled evaluation checks detect wrong findings and irrelevant retrievals."""

import json
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from incident_client import compare, evaluation
from incident_client.investigation import InvestigationEngine
from incident_client.metrics import finding_call
from incident_client.policy import DemoPlanner
from incident_common.models import InvestigationRequest, Record, ToolName
from incident_common.rag_models import RetrievalMatch
from incident_server.repository import FixtureRepository
from test_rag_flow import historical_result


def two_matches():
    result = historical_result()
    data = result.model_dump()
    data["records"].append(
        Record(
            record_id="history-INC-09402",
            summary="Past receipt-email queue delay; unrelated to settlement failures.",
            observed_at="2025-08-12T16:00:00Z",
            reference="synthetic://incidents/INC-09402",
            facts={
                "past_incident_id": "INC-09402",
                "historical": True,
                "service": "notifications",
                "useful_check": "get_monitoring",
            },
        ).model_dump()
    )
    data["retrieval"]["matches"].append(
        RetrievalMatch(
            incident_id="INC-09402", score=0.99, reference="synthetic://incidents/INC-09402"
        ).model_dump()
    )
    return type(result).model_validate(data)


class EvaluationGateway:
    def __init__(self):
        self.repository = FixtureRepository.load()

    async def call_tool(self, tool, transaction_id, **kwargs):
        if tool == ToolName.SIMILAR:
            assert kwargs["query"] and kwargs["current_incident_id"].startswith("INC-900")
            result = (
                historical_result(empty=True) if transaction_id == "TX900004" else two_matches()
            )
            return result.model_copy(update={"transaction_id": transaction_id})
        from incident_common.models import TOOL_SOURCES

        return self.repository.get(TOOL_SOURCES[tool], transaction_id)


async def scored(case, enabled=True):
    report = await InvestigationEngine(EvaluationGateway()).run(
        InvestigationRequest(
            incident_id=case.incident_id, transaction_id=case.transaction_id, use_rag=enabled
        )
    )
    return report, evaluation.score_run(
        case, report, elapsed_ms=100, call_times={n: n * 10 for n in range(1, 9)}
    )


async def test_six_cases_check_findings_causes_retrieval_and_readiness_separately():
    runs = []
    for case in evaluation.load_cases():
        for enabled in (False, True):
            report, run = await scored(case, enabled)
            assert run["quality_check_passed"] and run["finding_check_passed"]
            assert report.calls_used == (6 if enabled else 5)
            assert not report.root_cause_confirmed and not report.knowledge_base_saved
            assert run["calls_to_finding_or_pause"] is not None
            assert run["ms_to_finding_or_pause"] == run["calls_to_finding_or_pause"] * 10
            runs.append(run)
    summary = evaluation.summarize(runs)
    assert summary["without_rag"]["quality_checks_passed"] == 6
    assert summary["with_rag"]["cause_labels_correct"] == 15
    assert summary["retrieval"]["precision_among_returned"] == 0.5
    assert summary["retrieval"]["expected_match_hits"] == 5
    assert summary["retrieval"]["relevant_records_expected"] == 6
    assert summary["retrieval"]["anchoring_checks_exercised"] == 3
    assert (
        summary["with_rag"]["mean_calls_to_finding_or_pause"]
        > summary["without_rag"]["mean_calls_to_finding_or_pause"]
    )


async def test_misleading_history_cannot_receive_credit_for_a_wrong_cause_label():
    case = next(case for case in evaluation.load_cases() if case.case_id == "misleading")
    report, good = await scored(case)
    assert good["anchoring_check_passed"] is True
    cause = next(c for c in report.cause_assessments if c.cause_id == "broad_service_issue")
    report.cause_assessments = [
        c.model_copy(update={"status": "possible"}) if c is cause else c
        for c in report.cause_assessments
    ]
    bad = evaluation.score_run(case, report, elapsed_ms=100, call_times={})
    assert not bad["quality_check_passed"] and not bad["anchoring_check_passed"]
    assert bad["calls_to_finding_or_pause"] is None


async def test_missing_trace_cannot_get_credit_for_an_unsupported_failure_point():
    case = next(case for case in evaluation.load_cases() if case.case_id == "missing_trace")
    report, score = await scored(case)
    assert score["quality_check_passed"] and finding_call(report) is None
    report.supported_failure_point = "Settlement-service call"
    bad = evaluation.score_run(case, report, elapsed_ms=1, call_times={})
    assert bad["unsafe_claim"] and not bad["finding_check_passed"]
    assert bad["calls_to_finding_or_pause"] is None


async def test_no_results_have_undefined_precision_and_do_not_fake_a_known_match_hit():
    case = next(case for case in evaluation.load_cases() if case.case_id == "no_match")
    report, run = await scored(case)
    assert run["retrieval_precision"] is None and run["no_match_correct"] is True
    assert run["expected_match_hit"] is None
    known = evaluation.load_cases()[0]
    report.retrieval = historical_result(empty=True).retrieval
    score = evaluation.score_run(known, report, elapsed_ms=1, call_times={})
    assert score["expected_match_hit"] is False


async def test_incomplete_api_run_is_not_misreported_as_a_correct_pause_or_speed_gain():
    case = next(case for case in evaluation.load_cases() if case.case_id == "missing_trace")
    report, _ = await scored(case)
    report.stop_reason = "planner_error"
    report.retrieval = None
    score = evaluation.score_run(case, report, elapsed_ms=1, call_times={})
    assert score["operational_error"] and not score["quality_check_passed"]
    assert score["anchoring_check_passed"] is None and score["expected_match_hit"] is None
    summary = evaluation.summarize([score, {**score, "with_rag": False}])
    assert summary["retrieval"]["known_match_queries_requested"] == 1
    assert summary["retrieval"]["known_match_queries_completed"] == 0
    assert summary["with_rag"]["mean_calls_to_finding_or_pause"] is None


def test_answer_key_is_separate_from_current_evidence_and_historical_corpus():
    from importlib.resources import files

    data = files("incident_server").joinpath("fixtures", "evaluation_incidents.json").read_text()
    assert "expected_" not in data and "relevant_incident_ids" not in data
    assert len(evaluation.load_cases()) == 6
    assert not ({case.incident_id for case in evaluation.load_cases()} & evaluation.SEED_IDS)
    evidence = json.loads(data)
    assert len({item["transaction_id"] for item in evidence["results"]}) == 6


async def test_runner_alternates_order_keeps_answers_private_and_closes_fresh_planners(monkeypatch):
    planners = []
    requests = []

    class CheckedPlanner(DemoPlanner):
        def __init__(self):
            self.aclose = AsyncMock()
            planners.append(self)

        def choose(self, request, memory):
            assert set(request.model_dump()) == {
                "incident_id",
                "transaction_id",
                "scope",
                "mode",
                "use_rag",
            }
            assert not any(
                "expected_" in r.model_dump_json()
                for result in memory.results.values()
                for r in result.records
            )
            return super().choose(request, memory)

    class Engine(InvestigationEngine):
        async def run(self, request, **kwargs):
            requests.append((request.incident_id, request.use_rag))
            return await super().run(request, **kwargs)

    @asynccontextmanager
    async def corpus():
        yield None, None

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield EvaluationGateway()

    monkeypatch.setattr(evaluation, "prepared_corpus", corpus)
    monkeypatch.setattr(evaluation, "connect", connect)
    monkeypatch.setattr(evaluation, "DemoPlanner", CheckedPlanner)
    monkeypatch.setattr(evaluation, "InvestigationEngine", Engine)
    result = await evaluation.run_evaluation(repeats=2)
    assert len(planners) == 24 and len(result["runs"]) == 24
    assert [enabled for _, enabled in requests[:4]] == [False, True, True, False]
    assert requests[0][1] != requests[12][1]
    assert all(planner.aclose.await_count == 1 for planner in planners)
    text = evaluation.render_evaluation(result)
    assert (
        "useful history 2/4" in text
        and "Root-cause accuracy and engineer time: not measured" in text
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"repeats": 0},
        {"repeats": 4},
        {"repeats": True},
        {"case_id": "unknown"},
        {"planner_name": "other"},
    ],
)
async def test_evaluation_rejects_unbounded_or_unknown_requests(arguments):
    with pytest.raises(ValueError):
        await evaluation.run_evaluation(**arguments)


async def test_private_index_build_strips_api_secrets_and_leaves_local_stores_alone(
    tmp_path, monkeypatch
):
    cache = tmp_path / "local" / "model-cache"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_text("synthetic cached model")
    monkeypatch.setenv("INCIDENT_RAG_DIR", str(cache.parent))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-key")
    process = SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(evaluation.asyncio, "create_subprocess_exec", spawn)
    async with evaluation.prepared_corpus() as (directory, database):
        assert directory != cache.parent and not database.exists()
        assert (directory / "model-cache").resolve() == cache
        captured = spawn.call_args.kwargs["env"]
        assert "OPENAI_API_KEY" not in captured and captured["HF_HUB_OFFLINE"] == "1"
        assert captured["INCIDENT_RAG_DIR"] == str(directory)
    assert not directory.exists() and cache.is_dir()
    assert (cache / "model.bin").read_text() == "synthetic cached model"


@pytest.mark.parametrize("good", [True, False])
def test_evaluation_cli_defaults_to_demo_prints_json_and_never_exports_files(
    good, monkeypatch, capsys, tmp_path
):
    run = AsyncMock(return_value={"runs": [{"quality_check_passed": good}]})
    monkeypatch.setattr(evaluation, "run_evaluation", run)
    monkeypatch.setattr(sys, "argv", ["incident-compare", "--evaluate", "--json"])
    monkeypatch.chdir(tmp_path)
    if good:
        compare.main()
    else:
        with pytest.raises(SystemExit) as result:
            compare.main()
        assert result.value.code == 2
    assert json.loads(capsys.readouterr().out)["runs"][0]["quality_check_passed"] == good
    run.assert_awaited_once_with("demo", repeats=1, case_id=None, verbose=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [["--evaluate", "--reasoning", "tot"], ["--evaluate", "--repeats", "4"], ["--case", "match"]],
)
def test_cli_rejects_confounded_or_unbounded_evaluation_options(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["incident-compare", *arguments])
    with pytest.raises(SystemExit) as result:
        compare.main()
    assert result.value.code == 2


@pytest.mark.skipif(
    os.getenv("RUN_RAG_INTEGRATION") != "1", reason="Needs the cached embedding model"
)
async def test_full_evaluation_uses_real_embeddings_and_mcp_without_touching_saved_decisions():
    result = await evaluation.run_evaluation()
    assert len(result["runs"]) == 12
    assert all(run["quality_check_passed"] for run in result["runs"])
    assert result["summary"]["retrieval"]["expected_match_hits"] == 5
    assert result["summary"]["retrieval"]["precision_among_returned"] == 0.5
    assert result["summary"]["with_rag"]["mean_tool_calls"] == 6
    assert all(run["model_requests"] == 0 for run in result["runs"])
