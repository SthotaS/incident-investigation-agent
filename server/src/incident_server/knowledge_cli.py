"""Human-operated local knowledge management, separate from autonomous MCP tools."""

import argparse
import json
from pathlib import Path

from incident_common.knowledge_models import HumanDecisionInput
from pydantic import ValidationError

from incident_server.knowledge import DecisionConflictError, DecisionStore, KnowledgeStoreError
from incident_server.rag import MAX_CASES, RagIndexError, build_index, load_seed_cases


def save_and_index(decision: HumanDecisionInput, *, approve_save: bool) -> dict:
    """Commit approved knowledge first; a failed index rebuild must never lose it."""
    seed_cases = load_seed_cases()
    if decision.incident_id in {case.incident_id for case in seed_cases}:
        raise DecisionConflictError("That incident ID belongs to an existing synthetic seed case.")
    store = DecisionStore(max_records=MAX_CASES - len(seed_cases))
    saved = store.save(decision, approve_save=approve_save)
    result = {
        "status": saved.status,
        "database_path": str(store.path),
        "record": saved.record.model_dump(mode="json"),
        "search_index_updated": False,
        "index_error": None,
    }
    try:
        # No downloads or API requests on a human save. Prepare the model once with incident-index.
        build_index(local_files_only=True)
        result["search_index_updated"] = True
    except RagIndexError:
        result["index_error"] = (
            "The decision is saved, but search needs an index rebuild. Run incident-index. "
            "Until rebuilt, the old index will be rejected rather than return outdated matches."
        )
    return result


def _ask(label: str, default: str | None = None, *, optional: bool = False) -> str | None:
    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{label}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        if optional:
            return None
        print("Please enter a value.")


def _interactive_decision() -> HumanDecisionInput:
    print("Save a human-reviewed synthetic incident for future searches.")
    print("Enter what was actually decided and done. A save does not confirm an unknown cause.")
    payload = {
        "incident_id": _ask("Incident ID", "INC-10234"),
        "title": _ask("Short incident title"),
        "symptoms": _ask("Symptoms and important observations"),
        "service": _ask("Affected service"),
        "human_decision": _ask("Human decision and reason"),
        "action_taken": _ask("Action actually taken (or say no action yet)"),
        "outcome": _ask("Result: resolved, improved, no_change, pending, or unknown", "unknown"),
        "approved_by": _ask("Engineer approving this save"),
        "confirmed_root_cause": _ask(
            "Confirmed root cause (Enter if still unknown)", optional=True
        ),
        "synthetic": True,
    }
    decision = HumanDecisionInput.model_validate(payload)
    print("\nRecord to save:")
    print(decision.model_dump_json(indent=2))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Save and view human-approved incident knowledge.")
    sub = parser.add_subparsers(dest="command", required=True)
    save = sub.add_parser("save", help="Save a reviewed synthetic decision and action")
    save.add_argument(
        "--file", type=Path, help="Read the record from a JSON file instead of prompts"
    )
    save.add_argument("--approve-save", action="store_true", help="Explicitly approve this save")
    save.add_argument("--json", action="store_true", help="Print the saved record and index status")
    sub.add_parser("list", help="List saved human decisions without changing the database")
    show = sub.add_parser("show", help="Read one saved decision")
    show.add_argument("--incident-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            records = DecisionStore().list()
            if not records:
                print("No human decisions have been saved yet.")
            for record in records:
                print(
                    f"{record.incident_id}: {record.title} | outcome: {record.outcome} "
                    f"| approved by: {record.approved_by}"
                )
            return
        if args.command == "show":
            record = DecisionStore().get(args.incident_id)
            if record is None:
                parser.exit(1, "No saved human decision was found for that incident ID.\n")
            print(record.model_dump_json(indent=2))
            return
        if args.file:
            if not args.approve_save:
                parser.error("Review the JSON file, then add --approve-save to save it.")
            if args.file.stat().st_size > 65536:
                parser.error("The decision JSON file must be smaller than 64 KiB.")
            decision = HumanDecisionInput.model_validate_json(args.file.read_text(encoding="utf-8"))
        else:
            decision = _interactive_decision()
            if not args.approve_save:
                approved = input(
                    "Save this reviewed record for future searches? Type SAVE: "
                ).strip()
                if approved != "SAVE":
                    print("Save cancelled. No record was written.")
                    return
        result = save_and_index(decision, approve_save=True)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{result['status']}: {decision.incident_id}")
            print(f"Database: {result['database_path']}")
            print(f"Human decision: {decision.human_decision}")
            print(f"Action taken: {decision.action_taken}")
            print(f"Outcome: {decision.outcome}")
            print("Search index updated: " + ("yes" if result["search_index_updated"] else "no"))
            if result["index_error"]:
                print(result["index_error"])
        if not result["search_index_updated"]:
            raise SystemExit(2)
    except (KnowledgeStoreError, ValidationError, OSError, ValueError) as exc:
        parser.exit(1, f"Could not save or read the decision: {exc}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nCancelled.\n")


if __name__ == "__main__":
    main()
