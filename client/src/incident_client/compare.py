"""Compare the same incident without and with RAG in the terminal."""

import argparse
import asyncio
import json
import os
from functools import partial
from getpass import getpass

from incident_common.models import InvestigationReport, InvestigationRequest, ToolName

from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner, PlannerError
from incident_client.main import _progress
from incident_client.mcp_client import connect
from incident_client.metrics import work_metrics
from incident_client.policy import DemoPlanner


def comparison_details(before: InvestigationReport, after: InvestigationReport) -> dict:
    def next_check(report, after_tool):
        seen = False
        for entry in report.audit:
            if entry.tool == after_tool and entry.status != "error":
                seen = True
            elif seen and entry.tool not in {ToolName.SIMILAR, ToolName.HISTORY}:
                return entry
        return None

    baseline = next_check(before, ToolName.LOGS)
    retrieved = next_check(after, ToolName.SIMILAR)
    changed = bool(baseline and retrieved and baseline.tool != retrieved.tool)
    refs = {m.reference for m in after.retrieval.matches} if after.retrieval else set()
    cites_history = bool(retrieved and refs.intersection(retrieved.decision_references))
    return {
        "without_rag_next_check": baseline.tool.value if baseline else None,
        "with_rag_next_check": retrieved.tool.value if retrieved else None,
        "next_check_changed": changed,
        "changed_step_cites_history": cites_history,
        "difference_demonstrated": changed and cites_history,
        "without_rag_work": work_metrics(before),
        "with_rag_work": work_metrics(after),
        "supported_finding_unchanged": before.supported_failure_point is not None
        and before.supported_failure_point == after.supported_failure_point,
        "accuracy_scored": False,
        "note": (
            "The next check changed and its action summary cites a retrieved past case."
            if changed and cites_history
            else "This run did not demonstrate a change supported by a historical citation. "
            "Review the actual actions; model choices may vary."
        ),
    }


async def run_comparison(
    planner_name: str, *, verbose: bool = False, quiet: bool = False
) -> tuple[InvestigationReport, InvestigationReport]:
    if planner_name == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
        raise PlannerError("Use --ask-key or set OPENAI_API_KEY before running the API comparison.")
    settings = Settings.from_environment()
    reports = []
    async with connect(settings.tool_timeout_seconds) as gateway:
        for use_rag in (False, True):
            planner = OpenAIPlanner() if planner_name == "openai" else DemoPlanner()
            try:
                report = await InvestigationEngine(gateway, planner=planner).run(
                    InvestigationRequest(use_rag=use_rag),
                    on_progress=None if quiet else partial(_progress, verbose=verbose),
                )
                reports.append(report)
            finally:
                close = getattr(planner, "aclose", None)
                if close is not None:
                    await close()
    return reports[0], reports[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the same incident without and with RAG.")
    parser.add_argument("--planner", choices=["openai", "demo"], default=None)
    parser.add_argument("--ask-key", action="store_true", help="Read an API key without saving it")
    parser.add_argument("--verbose", action="store_true", help="Show full live action details")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run six labeled synthetic RAG cases"
    )
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument(
        "--case",
        dest="case_id",
        choices=["match", "changed_cause", "misleading", "no_match", "missing_trace", "paraphrase"],
        help="Evaluate one case instead of all six",
    )
    parser.add_argument("--json", action="store_true", help="Print structured results to stdout")
    args = parser.parse_args()
    if not args.evaluate and (args.case_id is not None or args.repeats != 1):
        parser.error("--case and --repeats are for --evaluate.")
    args.planner = args.planner or ("demo" if args.evaluate else "openai")
    if args.ask_key:
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key (hidden, not saved): ").strip()
    try:
        if args.evaluate:
            from incident_client.evaluation import render_evaluation, run_evaluation

            result = asyncio.run(
                run_evaluation(
                    args.planner,
                    repeats=args.repeats,
                    case_id=args.case_id,
                    verbose=args.verbose and not args.json,
                )
            )
            print(json.dumps(result, indent=2) if args.json else render_evaluation(result))
            if not all(run["quality_check_passed"] for run in result["runs"]):
                raise SystemExit(2)
            return
        before, after = asyncio.run(
            run_comparison(
                args.planner,
                verbose=args.verbose,
                **({"quiet": True} if args.json else {}),
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except PlannerError as exc:
        parser.exit(1, f"{exc}\n")
    except Exception:
        parser.exit(1, "Comparison could not start. Check the local setup and configuration.\n")
    details = comparison_details(before, after)
    if args.json:
        print(json.dumps(details, indent=2))
        if before.status == "paused" or after.status == "paused":
            raise SystemExit(2)
        return
    print(details["note"])
    print(
        f"Without RAG: {details['without_rag_next_check']}; "
        f"with RAG: {details['with_rag_next_check']}"
    )
    basic, rag = details["without_rag_work"], details["with_rag_work"]
    print(
        f"Tool calls: without {basic['tool_calls']}; with {rag['tool_calls']} "
        f"(including {rag['retrieval_attempts']} retrieval attempts)."
    )
    print(
        f"Calls until current logs and trace support the finding: "
        f"without {basic['calls_to_supported_finding']}; with {rag['calls_to_supported_finding']}."
    )
    print(
        "Changed order alone is not an efficiency gain. Use --evaluate for labeled quality checks."
    )
    if before.status == "paused" or after.status == "paused":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
