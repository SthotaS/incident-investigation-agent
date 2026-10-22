"""SQLite storage for decisions a human explicitly approves saving.

The autonomous investigation server has no write tool for this store. Approval to
save records an engineer's decision; it does not mean a cause was confirmed, an
issue was resolved, or a production change was made by this application.
"""

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from incident_common.knowledge_models import (
    HumanDecisionInput,
    HumanDecisionRecord,
    IncidentIdentifier,
    SaveResult,
)
from pydantic import TypeAdapter, ValidationError

SCHEMA_VERSION = 1
TABLE_NAME = "human_decisions"
COLUMNS = (
    "incident_id",
    "title",
    "symptoms",
    "service",
    "human_decision",
    "action_taken",
    "outcome",
    "approved_by",
    "confirmed_root_cause",
    "occurred_at",
    "evidence_references",
    "useful_check",
    "check_reason",
    "synthetic",
    "saved_at",
    "record_source",
)
CREATE_TABLE = """
CREATE TABLE human_decisions (
    incident_id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    symptoms TEXT NOT NULL,
    service TEXT NOT NULL,
    human_decision TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('resolved', 'improved', 'no_change', 'pending', 'unknown')),
    approved_by TEXT NOT NULL,
    confirmed_root_cause TEXT,
    occurred_at TEXT,
    evidence_references TEXT NOT NULL,
    useful_check TEXT NOT NULL
        CHECK (useful_check IN ('get_monitoring', 'get_trace', 'get_deployments')),
    check_reason TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1),
    saved_at TEXT NOT NULL,
    record_source TEXT NOT NULL CHECK (record_source = 'human_approved')
)
"""


class KnowledgeStoreError(RuntimeError):
    """The local decision database could not be read or written safely."""


class DecisionConflictError(KnowledgeStoreError):
    """An incident ID already has a different approved decision."""


def get_database_path() -> Path:
    return (
        Path(os.environ.get("INCIDENT_KNOWLEDGE_DB", "data/knowledge/incidents.sqlite3"))
        .expanduser()
        .resolve()
    )


def _check_schema(connection: sqlite3.Connection) -> bool:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if version == 0 and not tables:
        return False
    if version != SCHEMA_VERSION or TABLE_NAME not in tables:
        raise KnowledgeStoreError("The decision database has an unsupported schema version.")
    names = tuple(row[1] for row in connection.execute("PRAGMA table_info(human_decisions)"))
    if names != COLUMNS:
        raise KnowledgeStoreError("The decision database schema is invalid.")
    return True


def _record_from_row(row: sqlite3.Row) -> HumanDecisionRecord:
    try:
        data = dict(row)
        if data["synthetic"] != 1:
            raise ValueError("Only synthetic records are supported.")
        data["synthetic"] = True
        data["evidence_references"] = json.loads(data["evidence_references"])
        return HumanDecisionRecord.model_validate(data)
    except (TypeError, ValueError, KeyError, ValidationError) as exc:
        raise KnowledgeStoreError("The decision database contains an invalid record.") from exc


class DecisionStore:
    def __init__(
        self,
        path: Path | str | None = None,
        *,
        reserved_ids: Iterable[str] = (),
        max_records: int | None = None,
    ):
        if max_records is not None and (type(max_records) is not int or max_records < 1):
            raise ValueError("The saved-record limit must be a positive integer.")
        self.path = Path(path).expanduser().resolve() if path is not None else get_database_path()
        self.reserved_ids = frozenset(reserved_ids)
        self.max_records = max_records

    def _open_readonly(self) -> sqlite3.Connection | None:
        if not self.path.exists():
            return None
        if not self.path.is_file():
            raise KnowledgeStoreError("The decision database path is not a file.")
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            if not _check_schema(connection):
                connection.close()
                return None
            return connection
        except Exception:
            connection.close()
            raise

    def list(self, *, limit: int | None = None) -> list[HumanDecisionRecord]:
        """Read approved records without creating a missing directory or database."""
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("The read limit must be a positive integer.")
        try:
            connection = self._open_readonly()
            if connection is None:
                return []
            with closing(connection):
                query = "SELECT * FROM human_decisions ORDER BY incident_id"
                rows = (
                    connection.execute(query).fetchall()
                    if limit is None
                    else connection.execute(query + " LIMIT ?", (limit,)).fetchall()
                )
                return [_record_from_row(row) for row in rows]
        except (sqlite3.Error, OSError) as exc:
            raise KnowledgeStoreError("Could not read the local decision database.") from exc

    def get(self, incident_id: str) -> HumanDecisionRecord | None:
        try:
            incident_id = TypeAdapter(IncidentIdentifier).validate_python(incident_id)
        except ValidationError as exc:
            raise KnowledgeStoreError("The incident ID is invalid.") from exc
        try:
            connection = self._open_readonly()
            if connection is None:
                return None
            with closing(connection):
                row = connection.execute(
                    "SELECT * FROM human_decisions WHERE incident_id = ?", (incident_id,)
                ).fetchone()
                return _record_from_row(row) if row is not None else None
        except (sqlite3.Error, OSError) as exc:
            raise KnowledgeStoreError("Could not read the local decision database.") from exc

    def save(
        self,
        decision: HumanDecisionInput | Mapping[str, Any],
        *,
        approve_save: bool,
    ) -> SaveResult:
        """Save one immutable approved record; identical repeat saves are idempotent."""
        if approve_save is not True:
            raise KnowledgeStoreError(
                "Saving requires explicit human approval with --approve-save."
            )
        if isinstance(decision, HumanDecisionInput):
            decision = decision.model_dump()
        validated = HumanDecisionInput.model_validate(decision)
        if validated.incident_id in self.reserved_ids:
            raise DecisionConflictError("This incident ID belongs to a packaged historical case.")
        record = HumanDecisionRecord(**validated.model_dump(), saved_at=datetime.now(UTC))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if not _check_schema(connection):
                        connection.execute(CREATE_TABLE)
                        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    old = connection.execute(
                        "SELECT * FROM human_decisions WHERE incident_id = ?",
                        (validated.incident_id,),
                    ).fetchone()
                    if old is not None:
                        existing = _record_from_row(old)
                        existing_input = existing.model_dump(exclude={"saved_at", "record_source"})
                        if existing_input != validated.model_dump():
                            raise DecisionConflictError(
                                "This incident already has a different approved decision; "
                                "the saved record was not changed."
                            )
                        return SaveResult(status="already_saved", record=existing)
                    if self.max_records is not None:
                        saved_count = connection.execute(
                            "SELECT COUNT(*) FROM human_decisions"
                        ).fetchone()[0]
                        if saved_count >= self.max_records:
                            raise KnowledgeStoreError(
                                "The saved-incident limit has been reached. No record was written."
                            )
                    values = record.model_dump(mode="json")
                    values["evidence_references"] = json.dumps(record.evidence_references)
                    values["synthetic"] = 1
                    placeholders = ", ".join("?" for _ in COLUMNS)
                    columns = ", ".join(COLUMNS)
                    connection.execute(
                        f"INSERT INTO human_decisions ({columns}) VALUES ({placeholders})",
                        tuple(values[column] for column in COLUMNS),
                    )
                return SaveResult(status="saved", record=record)
        except (sqlite3.Error, OSError) as exc:
            raise KnowledgeStoreError(
                "Could not save the approved decision to the database."
            ) from exc
