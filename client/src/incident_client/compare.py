"""Compare the same incident without and with RAG in the terminal."""

import argparse
import asyncio
import os
from functools import partial
from getpass import getpass

from incident_common.models import InvestigationReport, InvestigationRequest, ToolName

from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner, PlannerError
from incident_client.main import _progress
from incident_client.mcp_client import connect
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
        "note": (
            "The next check changed and its action summary cites a retrieved past case."
            if changed and cites_history
            else "This run did not demonstrate a change supported by a historical citation. "
            "Review the actual actions; model choices may vary."
        ),
    }


async def run_comparison(
    planner_name: str, *, verbose: bool = False, reasoning: str = "linear"
) -> tuple[InvestigationReport, InvestigationReport]:
    if planner_name == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
        raise PlannerError("Use --ask-key or set OPENAI_API_KEY before running the API comparison.")
    settings = Settings.from_environment()
    reports = []
    async with connect(settings.tool_timeout_seconds) as gateway:
        for use_rag in (False, True):
            planner = OpenAIPlanner() if planner_name == "openai" else DemoPlanner()
            if reasoning == "tot":
                from incident_client.tot import ToTPlanner

                planner = ToTPlanner(planner)
            try:
                report = await InvestigationEngine(gateway, planner=planner).run(
                    InvestigationRequest(use_rag=use_rag),
                    on_progress=partial(_progress, verbose=verbose),
                )
                reports.append(report)
            finally:
                close = getattr(planner, "aclose", None)
                if close is not None:
                    await close()
    return reports[0], reports[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the same incident without and with RAG.")
    parser.add_argument("--planner", choices=["openai", "demo"], default="openai")
    parser.add_argument("--ask-key", action="store_true", help="Read an API key without saving it")
    parser.add_argument("--verbose", action="store_true", help="Show full live action details")
    parser.add_argument("--reasoning", choices=["linear", "tot"], default="linear")
    args = parser.parse_args()
    if args.ask_key:
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key (hidden, not saved): ").strip()
    try:
        before, after = asyncio.run(
            run_comparison(args.planner, verbose=args.verbose, reasoning=args.reasoning)
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except PlannerError as exc:
        parser.exit(1, f"{exc}\n")
    except Exception:
        parser.exit(1, "Comparison could not start. Check the local setup and configuration.\n")
    details = comparison_details(before, after)
    print(details["note"])
    print(
        f"Without RAG: {details['without_rag_next_check']}; "
        f"with RAG: {details['with_rag_next_check']}"
    )
    if before.status == "paused" or after.status == "paused":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
