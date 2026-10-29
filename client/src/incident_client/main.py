"""Run the local dummy-data demo: python -m incident_client."""

import argparse
import asyncio
import json
import logging
import os
import sys
from functools import partial
from getpass import getpass

from incident_common.models import AuditEntry, InvestigationRequest, SkippedCheck, ToolName
from pydantic import ValidationError

from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner, PlannerError
from incident_client.mcp_client import connect
from incident_client.policy import DemoPlanner
from incident_client.reporting import render_cause_assessments, render_progress, render_summary


def _progress(entry: AuditEntry | SkippedCheck, *, verbose: bool = False) -> None:
    print(render_progress(entry, verbose=verbose), file=sys.stderr)


def _rag_control(args: argparse.Namespace) -> dict:
    enabled = getattr(args, "use_rag", None)
    if enabled is None:
        return {"mode": args.mode or "legacy"}
    control = {"use_rag": enabled}
    if args.mode is not None:
        control["mode"] = args.mode
    return control


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    if (
        args.planner == "openai"
        and not args.list_tools
        and not os.getenv("OPENAI_API_KEY", "").strip()
    ):
        raise PlannerError("Use --ask-key or set OPENAI_API_KEY before using the API planner.")
    async with connect(settings.tool_timeout_seconds) as gateway:
        tools = await gateway.list_tools()
        if args.list_tools:
            print(json.dumps({"tools": tools}, indent=2) if args.json else "\n".join(tools))
            return 0
        missing = {t.value for t in ToolName} - set(tools)
        if missing:
            raise RuntimeError(
                f"The server is missing required tools: {', '.join(sorted(missing))}"
            )
        request = InvestigationRequest(
            incident_id=args.incident_id,
            transaction_id=args.transaction_id,
            scope=args.scope,
            **_rag_control(args),
        )
        planner = OpenAIPlanner() if args.planner == "openai" else DemoPlanner()
        try:
            report = await InvestigationEngine(gateway, planner=planner).run(
                request,
                on_progress=None
                if args.json
                else partial(_progress, verbose=getattr(args, "verbose", False)),
            )
        finally:
            close = getattr(planner, "aclose", None)
            if close is not None:
                await close()
    if args.json:
        print(report.model_dump_json(indent=2))
        return 0 if report.status == "finished" else 2
    if not getattr(args, "verbose", False):
        print("\n" + render_summary(report))
        return 0 if report.status == "finished" else 2
    print(f"\nSynthetic incident investigation — {report.mode}; planner: {report.planner}")
    if report.model:
        print(f"API model: {report.model}; model requests: {report.model_calls}")
        if report.model_usage.measured_responses:
            usage = report.model_usage
            print(
                f"API tokens reported: {usage.input_tokens} input, {usage.output_tokens} output; "
                f"usage received for {usage.measured_responses}/{report.model_calls} attempts"
            )
    if report.planner_error:
        print(f"Model issue: {report.planner_error}")
    if report.retrieval:
        for match in report.retrieval.matches:
            print(
                f"Past case: {match.incident_id}; similarity: {match.score:.3f}; {match.reference}"
            )
    print(f"Incident: {report.incident_id}; transaction: {report.transaction_id}")
    print(f"State: {report.status}; reason: {report.stop_reason}")
    print(f"Tool calls: {report.calls_used}/{report.call_limit} (retries included)")
    print(f"Supported failure point: {report.supported_failure_point or 'Not established'}")
    if report.possible_failure_point:
        print(f"Possible failure point: {report.possible_failure_point}")
    if report.likely_cause:
        print(f"Likely cause: {report.likely_cause}")
    if report.cause_assessments:
        print("\n" + render_cause_assessments(report.cause_assessments) + "\n")
    for item in report.unsupported_explanations:
        print(f"Not supported: {item}")
    for item in report.open_possible_causes:
        print(f"Still open: {item}")
    if report.missing_sources:
        print("Sources missing: " + ", ".join(s.value for s in report.missing_sources))
    for item in report.next_steps:
        print(f"Next: {item}")
    print("Root cause confirmed: no; knowledge saved: no; production changes: none")
    return 0 if report.status == "finished" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Python MCP capstone dummy-data client")
    parser.add_argument("--incident-id", default="INC-10234")
    parser.add_argument("--transaction-id", default="TX784521")
    parser.add_argument("--scope", choices=["full", "status"], default="full")
    parser.add_argument("--mode", choices=["legacy", "without-rag", "with-rag"], default=None)
    parser.add_argument(
        "--use-rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Search past incidents; --no-use-rag skips the knowledge database",
    )
    parser.add_argument("--planner", choices=["demo", "openai"], default=None)
    parser.add_argument(
        "--ask-key", action="store_true", help="Read an API key without echoing or saving it"
    )
    parser.add_argument("--list-tools", action="store_true", help="Discover the server's tools")
    parser.add_argument("--json", action="store_true", help="Write only the final JSON to stdout")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full evidence, source links, and action details",
    )
    args = parser.parse_args()
    try:
        request = InvestigationRequest(**_rag_control(args))
    except ValidationError:
        parser.error("mode and use_rag must agree.")
    args.planner = args.planner or ("demo" if request.mode == "legacy" else "openai")
    if args.ask_key:
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key (hidden, not saved): ").strip()
    try:
        settings = Settings.from_environment()
        # Validate input before opening a server process.
        InvestigationRequest(
            incident_id=args.incident_id,
            transaction_id=args.transaction_id,
            scope=args.scope,
            **_rag_control(args),
        )
    except ValidationError:
        parser.error("Invalid incident ID, transaction ID, or runtime settings.")
    logging.basicConfig(
        level=settings.log_level, stream=sys.stderr, format="%(levelname)s %(message)s"
    )
    try:
        code = asyncio.run(_run(args, settings))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except PlannerError as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from None
    except Exception:
        logging.error(
            "The investigation could not finish. Check the local setup and configuration."
        )
        raise SystemExit(1) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
