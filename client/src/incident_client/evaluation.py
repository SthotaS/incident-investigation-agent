"""Paired, labeled synthetic RAG evaluation through the real MCP server.

The answer key stays in this client. Current evidence and the five historical
seeds are separate corpora. Results go to the terminal, never a reports folder.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Literal

from incident_common.models import AuditEntry, IncidentId, InvestigationRequest, TransactionId
from pydantic import BaseModel, ConfigDict, Field, model_validator

from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner, PlannerError
from incident_client.main import _progress
from incident_client.mcp_client import connect
from incident_client.metrics import work_metrics
from incident_client.policy import DemoPlanner

SEED_IDS = {"INC-09876", "INC-09741", "INC-09632", "INC-09508", "INC-09402"}
CaseId = Literal["match", "changed_cause", "misleading", "no_match", "missing_trace", "paraphrase"]
CauseId = Literal["broad_service_issue", "transaction_specific_issue", "recent_deployment"]
CauseState = Literal["possible", "less_supported", "needs_evidence"]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: CaseId
    name: str = Field(min_length=1, max_length=100)
    incident_id: IncidentId
    transaction_id: TransactionId
    relevant_incident_ids: list[IncidentId] = Field(max_length=5)
    relevance_reason: str = Field(min_length=1, max_length=700)
    expected_status: Literal["finished", "paused"]
    expected_stop_reason: Literal["request_answered", "insufficient_evidence"]
    expected_failure_point: Literal["Settlement-service call"] | None
    expected_causes: dict[CauseId, CauseState]
    anchoring_test: bool

    @model_validator(mode="after")
    def validate_labels(self):
        if not set(self.relevant_incident_ids) <= SEED_IDS:
            raise ValueError("Expected matches must belong to the fixed seed corpus.")
        if len(set(self.relevant_incident_ids)) != len(self.relevant_incident_ids):
            raise ValueError("Expected historical incident IDs must be distinct.")
        if self.incident_id in SEED_IDS:
            raise ValueError("An evaluation incident cannot also be a historical seed.")
        return self


class EvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic: Literal[True]
    version: Literal[1]
    cases: list[EvaluationCase] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def distinct_cases(self):
        for field in ("case_id", "incident_id", "transaction_id"):
            if len({getattr(case, field) for case in self.cases}) != len(self.cases):
                raise ValueError("Evaluation cases must have separate identifiers.")
        return self


def load_cases() -> list[EvaluationCase]:
    text = files("incident_client").joinpath("fixtures", "rag_evaluation.json").read_text("utf-8")
    return EvaluationSuite.model_validate_json(text).cases


def score_run(case, report, *, elapsed_ms: float, call_times: dict[int, float]) -> dict:
    """Score against labels declared before retrieval, not against similarity scores."""
    work = work_metrics(report)
    operational_error = report.stop_reason in {"planner_error", "tool_retry_limit"} or bool(
        report.stop_reason == "no_new_evidence"
        and report.audit
        and report.audit[-1].status == "error"
    )
    expected = case.expected_causes
    actual = work["cause_states"]
    cause_checks = {
        key: key in expected and actual.get(key) == expected[key]
        for key in expected.keys() | actual.keys()
    }
    finding_ok = (
        report.status == case.expected_status
        and report.stop_reason == case.expected_stop_reason
        and report.supported_failure_point == case.expected_failure_point
    )
    unsafe_claim = bool(
        report.root_cause_confirmed
        or report.knowledge_base_saved
        or report.production_changes_made
        or (case.expected_failure_point is None and report.supported_failure_point is not None)
    )
    quality_ok = finding_ok and all(cause_checks.values()) and not unsafe_claim
    matches = report.retrieval.matches if report.retrieval else []
    retrieved = [match.incident_id for match in matches]
    relevant = set(case.relevant_incident_ids)
    useful = sum(incident_id in relevant for incident_id in retrieved)
    retrieval_completed = report.retrieval is not None
    # No returns have undefined precision, not a misleading 100 percent score.
    precision = useful / len(retrieved) if retrieved else None
    hit = bool(relevant.intersection(retrieved)) if relevant and retrieval_completed else None
    anchor_exercised = (
        report.use_rag and case.anchoring_test and bool(retrieved) and not operational_error
    )
    ready = work["calls_to_supported_finding"]
    result_call = (
        (ready if case.expected_failure_point else report.calls_used)
        if quality_ok and not operational_error
        else None
    )
    return {
        "case_id": case.case_id,
        "name": case.name,
        "with_rag": report.use_rag,
        **work,
        "status": report.status,
        "stop_reason": report.stop_reason,
        "operational_error": operational_error,
        "finding_check_passed": finding_ok and not operational_error,
        "cause_checks": cause_checks,
        "quality_check_passed": quality_ok and not operational_error,
        "unsafe_claim": unsafe_claim,
        "retrieval_completed": retrieval_completed,
        "retrieved_ids": retrieved,
        "useful_returned": useful,
        "returned_count": len(retrieved),
        "retrieval_precision": precision,
        "expected_match_query": bool(relevant),
        "expected_relevant_count": len(relevant),
        "expected_match_hit": hit,
        "no_match_correct": not retrieved if not relevant and retrieval_completed else None,
        "anchoring_check_exercised": anchor_exercised,
        "anchoring_check_passed": quality_ok if anchor_exercised else None,
        "calls_to_finding_or_pause": result_call,
        "ms_to_finding_or_pause": call_times.get(result_call) if result_call else None,
        "elapsed_ms": round(elapsed_ms, 2),
        "model_usage": report.model_usage.model_dump(),
    }


def summarize(runs: list[dict]) -> dict:
    modes = {}
    for use_rag, name in ((False, "without_rag"), (True, "with_rag")):
        group = [run for run in runs if run["with_rag"] == use_rag]
        labels = [value for run in group for value in run["cause_checks"].values()]
        valid = [run for run in group if run["calls_to_finding_or_pause"] is not None]
        modes[name] = {
            "runs": len(group),
            "finding_checks_passed": sum(run["finding_check_passed"] for run in group),
            "quality_checks_passed": sum(run["quality_check_passed"] for run in group),
            "cause_labels_correct": sum(labels),
            "cause_labels_checked": len(labels),
            "operational_errors": sum(run["operational_error"] for run in group),
            "unsafe_claims": sum(run["unsafe_claim"] for run in group),
            "mean_tool_calls": round(mean(run["tool_calls"] for run in group), 2),
            "mean_elapsed_ms": round(mean(run["elapsed_ms"] for run in group), 2),
            "mean_calls_to_finding_or_pause": round(
                mean(run["calls_to_finding_or_pause"] for run in valid), 2
            )
            if valid
            else None,
            "finding_or_pause_count": len(valid),
            "model_requests": sum(run["model_requests"] for run in group),
        }
    rag = [run for run in runs if run["with_rag"]]
    hits = [run["expected_match_hit"] for run in rag if run["expected_match_hit"] is not None]
    anchors = [run["anchoring_check_passed"] for run in rag if run["anchoring_check_exercised"]]
    no_matches = [run["no_match_correct"] for run in rag if run["no_match_correct"] is not None]
    returned = sum(run["returned_count"] for run in rag)
    useful = sum(run["useful_returned"] for run in rag)
    return {
        **modes,
        "retrieval": {
            "useful_returned": useful,
            "returned": returned,
            "precision_among_returned": useful / returned if returned else None,
            "expected_match_hits": sum(hits),
            "known_match_queries_completed": len(hits),
            "known_match_queries_requested": sum(run["expected_match_query"] for run in rag),
            "retrieval_queries_completed": sum(run["retrieval_completed"] for run in rag),
            "retrieval_queries_requested": len(rag),
            "relevant_records_expected": sum(run["expected_relevant_count"] for run in rag),
            "no_match_queries_correct": sum(no_matches),
            "no_match_queries_completed": len(no_matches),
            "anchoring_checks_passed": sum(anchors),
            "anchoring_checks_exercised": len(anchors),
        },
        "root_cause_accuracy": "not measured; engineer confirmation remains required",
        "engineer_time": "not measured; requires a timed human review study",
        "anchoring_scope": (
            "final structured findings and cause labels; free-text reasons not scored"
        ),
    }


def _corpus_paths(temporary: str) -> tuple[Path, Path]:
    cache = (
        Path(os.environ.get("INCIDENT_RAG_DIR", "data/rag")).expanduser().resolve() / "model-cache"
    )
    if not cache.is_dir():
        raise PlannerError(
            "Prepare the local embedding model with incident-index before evaluation."
        )
    directory = Path(temporary)
    rag_dir = directory / "rag"
    rag_dir.mkdir()
    (rag_dir / "model-cache").symlink_to(cache, target_is_directory=True)
    return rag_dir, directory / "empty-knowledge.sqlite3"


@asynccontextmanager
async def prepared_corpus():
    """Build in a disposable directory using cached embeddings and an empty private DB."""
    with TemporaryDirectory(prefix="incident-rag-eval-") as temporary:
        rag_dir, database = await asyncio.to_thread(_corpus_paths, temporary)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OPENAI_", "LANGSMITH_"))
        }
        env.update(
            INCIDENT_RAG_DIR=str(rag_dir),
            INCIDENT_KNOWLEDGE_DB=str(database),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "incident_server.rag",
            "--offline",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=60)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode != 0:
            raise PlannerError("The private evaluation index could not be built from cached files.")
        yield rag_dir, database


async def run_evaluation(
    planner_name: str = "demo",
    *,
    repeats: int = 1,
    case_id: str | None = None,
    verbose: bool = False,
) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("Evaluation repeats must be between one and three.")
    if planner_name not in {"demo", "openai"}:
        raise ValueError("Choose demo or openai for the evaluation planner.")
    cases = load_cases()
    if case_id is not None:
        cases = [case for case in cases if case.case_id == case_id]
        if not cases:
            raise ValueError("Unknown synthetic evaluation case.")
    if planner_name == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
        raise PlannerError("Use --ask-key or set OPENAI_API_KEY before using the API planner.")
    settings = Settings.from_environment()
    runs = []
    async with prepared_corpus() as (rag_dir, database):
        async with connect(
            settings.tool_timeout_seconds, rag_dir=rag_dir, knowledge_db=database
        ) as gateway:
            for trial in range(1, repeats + 1):
                for index, case in enumerate(cases):
                    # Alternate order to reduce consistently favoring the second run.
                    modes = (False, True) if (index + trial) % 2 else (True, False)
                    for enabled in modes:
                        planner = OpenAIPlanner() if planner_name == "openai" else DemoPlanner()
                        call_times = {}
                        start = monotonic()

                        def progress(entry, times=call_times, started=start):
                            if isinstance(entry, AuditEntry):
                                times[entry.call_number] = (monotonic() - started) * 1000
                            if verbose:
                                _progress(entry, verbose=True)

                        try:
                            # Only IDs and the RAG flag cross into the investigation. No labels.
                            request = InvestigationRequest(
                                incident_id=case.incident_id,
                                transaction_id=case.transaction_id,
                                use_rag=enabled,
                            )
                            report = await InvestigationEngine(gateway, planner=planner).run(
                                request,
                                on_progress=progress,
                            )
                            elapsed_ms = (monotonic() - start) * 1000
                        finally:
                            close = getattr(planner, "aclose", None)
                            if close is not None:
                                await close()
                        run = score_run(case, report, elapsed_ms=elapsed_ms, call_times=call_times)
                        runs.append({"trial": trial, **run})
    return {
        "planner": planner_name,
        "reasoning": "linear",
        "cases": len(cases),
        "repeats": repeats,
        "corpus": "five fixed synthetic seeds; excludes saved decisions and evaluation incidents",
        "timing": (
            "run time includes model and MCP calls and first-use warmup; "
            "excludes index build and server startup"
        ),
        "runs": runs,
        "summary": summarize(runs),
    }


def render_evaluation(result: dict) -> str:
    summary = result["summary"]
    before, after, retrieval = (summary[key] for key in ("without_rag", "with_rag", "retrieval"))
    lines = [
        f"RAG evaluation | {result['cases']} cases x {result['repeats']} repeats | "
        f"{result['planner']} | linear",
        "Five seed incidents; expected answers stay outside the agent.",
    ]
    for case_id in dict.fromkeys(run["case_id"] for run in result["runs"]):
        pair = [run for run in result["runs"] if run["case_id"] == case_id]
        basic, rag = (
            [run for run in pair if run["with_rag"] == enabled] for enabled in (False, True)
        )
        ok = sum(run["quality_check_passed"] for run in pair)
        found = sum(run["returned_count"] for run in rag)
        useful = sum(run["useful_returned"] for run in rag)
        history_text = f"useful history {useful}/{found}" if found else "no history returned"
        lines.append(
            f"{case_id}: calls {mean(r['tool_calls'] for r in basic):g} -> "
            f"{mean(r['tool_calls'] for r in rag):g}; {history_text}; "
            f"finding and cause checks {ok}/{len(pair)}"
        )
    for label, mode in (("Without RAG", before), ("With RAG", after)):
        lines.append(
            f"{label}: finding checks {mode['finding_checks_passed']}/{mode['runs']}; "
            f"cause labels {mode['cause_labels_correct']}/{mode['cause_labels_checked']}; "
            f"mean calls {mode['mean_tool_calls']:g}; "
            f"mean run {mode['mean_elapsed_ms'] / 1000:.3f}s"
        )
    lines += [
        f"Useful returned records: {retrieval['useful_returned']}/{retrieval['returned']}; "
        f"expected useful records found {retrieval['useful_returned']}/"
        f"{retrieval['relevant_records_expected']}.",
        f"Expected useful match found: {retrieval['expected_match_hits']}/"
        f"{retrieval['known_match_queries_requested']} known-match queries; "
        f"{retrieval['retrieval_queries_completed']}/"
        f"{retrieval['retrieval_queries_requested']} searches completed.",
        f"No-match queries correctly empty: {retrieval['no_match_queries_correct']}/"
        f"{retrieval['no_match_queries_completed']}.",
        f"Misleading or incomplete history checks: {retrieval['anchoring_checks_passed']}/"
        f"{retrieval['anchoring_checks_exercised']} exercised; unsupported final findings "
        f"{after['unsafe_claims']}; incomplete API/tool runs {after['operational_errors']}.",
        f"Mean calls to failure-point evidence or correct pause: "
        f"without {before['mean_calls_to_finding_or_pause']} "
        f"(n={before['finding_or_pause_count']}); "
        f"with {after['mean_calls_to_finding_or_pause']} (n={after['finding_or_pause_count']}).",
        "More calls can be a cost, even when the next check changes. Compare quality before speed.",
        "Root-cause accuracy and engineer time: not measured. Root cause still needs an engineer.",
        "Anchoring checks score final fields; free-text model reasons are not scored.",
        "Timing includes first-use warmup; this tiny set does not prove a general benefit.",
    ]
    return "\n".join(lines)
