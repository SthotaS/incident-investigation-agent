"""Real SQLite checks for explicit human approval and durable decision storage."""

import sqlite3
from datetime import UTC, datetime

import pytest
from incident_common.knowledge_models import HumanDecisionInput
from incident_server.knowledge import DecisionConflictError, DecisionStore, KnowledgeStoreError
from pydantic import ValidationError


@pytest.fixture
def decision():
    return HumanDecisionInput(
        incident_id="INC-10801",
        title="Synthetic settlement delay",
        symptoms="A synthetic transaction returned settlement HTTP 503 errors.",
        service="settlement",
        human_decision="Wait for the service owner's health check before another attempt.",
        action_taken="The engineer contacted the synthetic service owner; no retry was made.",
        approved_by="Synthetic engineer",
        evidence_references=["synthetic://logs/TX123456/503"],
    )


def test_saved_record_persists_all_fields_after_reopening(tmp_path, decision):
    path = tmp_path / "knowledge" / "incidents.sqlite3"
    before = datetime.now(UTC)
    saved = DecisionStore(path).save(decision, approve_save=True)
    assert saved.status == "saved"
    assert saved.record.saved_at >= before
    assert saved.record.saved_at.utcoffset().total_seconds() == 0
    assert saved.record.record_source == "human_approved"
    reopened = DecisionStore(path)
    assert reopened.list() == [saved.record]
    assert reopened.get(decision.incident_id) == saved.record
    assert reopened.get("INC-10802") is None
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT human_decision, action_taken, outcome, approved_by FROM human_decisions"
        ).fetchone()
    assert row == (
        decision.human_decision,
        decision.action_taken,
        "unknown",
        decision.approved_by,
    )


def test_unapproved_save_does_not_create_database_or_directory(tmp_path, decision):
    path = tmp_path / "missing" / "incidents.sqlite3"
    with pytest.raises(KnowledgeStoreError, match="explicit human approval"):
        DecisionStore(path).save(decision, approve_save=False)
    assert not path.parent.exists()


def test_same_decision_is_idempotent_and_preserves_original_saved_time(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3")
    first = store.save(decision, approve_save=True)
    second = store.save(decision.model_dump(), approve_save=True)
    assert second.status == "already_saved"
    assert second.record == first.record
    assert len(store.list()) == 1


def test_conflicting_decision_does_not_overwrite_approved_record(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3")
    original = store.save(decision, approve_save=True).record
    changed = decision.model_copy(update={"action_taken": "A different synthetic action."})
    with pytest.raises(DecisionConflictError, match="not changed"):
        store.save(changed, approve_save=True)
    assert store.get(decision.incident_id) == original


def test_reserved_seed_id_rejected_before_creating_database(tmp_path, decision):
    path = tmp_path / "incidents.sqlite3"
    with pytest.raises(DecisionConflictError, match="packaged"):
        DecisionStore(path, reserved_ids={decision.incident_id}).save(decision, approve_save=True)
    assert not path.exists()


def test_sql_punctuation_is_stored_as_text(tmp_path, decision):
    action = "Engineer wrote: '); DROP TABLE human_decisions; -- (synthetic example)."
    decision = decision.model_copy(update={"action_taken": action})
    store = DecisionStore(tmp_path / "incidents.sqlite3")
    store.save(decision, approve_save=True)
    assert store.get(decision.incident_id).action_taken == action
    assert len(store.list()) == 1


def test_missing_database_reads_do_not_create_files(tmp_path):
    path = tmp_path / "missing" / "incidents.sqlite3"
    store = DecisionStore(path)
    assert store.list() == []
    assert store.get("INC-10801") is None
    assert not path.parent.exists()


def test_reads_use_sqlite_readonly_connection(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3")
    store.save(decision, approve_save=True)
    connection = store._open_readonly()
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM human_decisions")
    finally:
        connection.close()
    assert len(store.list()) == 1


def test_save_approval_does_not_claim_cause_or_resolution(tmp_path, decision):
    record = DecisionStore(tmp_path / "incidents.sqlite3").save(decision, approve_save=True).record
    assert record.outcome == "unknown"
    assert record.confirmed_root_cause is None
    assert record.occurred_at is None
    pending = HumanDecisionInput.model_validate({**decision.model_dump(), "outcome": "pending"})
    assert pending.confirmed_root_cause is None


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "symptoms",
        "service",
        "human_decision",
        "action_taken",
        "approved_by",
        "check_reason",
    ],
)
def test_required_text_cannot_be_blank(decision, field):
    with pytest.raises(ValidationError):
        HumanDecisionInput.model_validate({**decision.model_dump(), field: " \t\n "})


@pytest.mark.parametrize("field", ["human_decision", "action_taken", "approved_by"])
def test_required_human_fields_cannot_be_missing(decision, field):
    payload = decision.model_dump()
    del payload[field]
    with pytest.raises(ValidationError):
        HumanDecisionInput.model_validate(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"incident_id": "INC-1080"},
        {"synthetic": False},
        {"outcome": "assumed_resolved"},
        {"confirmed_root_cause": " "},
        {"occurred_at": "2026-10-22T10:00:00"},
        {"evidence_references": ["https://example.com/private/log"]},
        {"evidence_references": ["synthetic://"]},
        {"evidence_references": ["synthetic://some log"]},
        {"action_taken": "x" * 2001},
        {"unknown_field": "extra"},
    ],
)
def test_invalid_record_fields_are_rejected(decision, updates):
    with pytest.raises(ValidationError):
        HumanDecisionInput.model_validate({**decision.model_dump(), **updates})


def test_whitespace_is_trimmed_and_saved_in_utc(tmp_path, decision):
    payload = decision.model_dump()
    payload.update(
        incident_id=" INC-10801 ",
        human_decision="  Review current service health.  ",
        evidence_references=[" synthetic://logs/TX123456/503 "],
        occurred_at="2026-10-22T10:00:00-05:00",
    )
    saved = DecisionStore(tmp_path / "incidents.sqlite3").save(payload, approve_save=True)
    assert saved.record.incident_id == "INC-10801"
    assert saved.record.human_decision == "Review current service health."
    assert saved.record.evidence_references == ["synthetic://logs/TX123456/503"]
    assert saved.record.occurred_at.utcoffset().total_seconds() == -5 * 60 * 60


def test_existing_empty_sqlite_database_is_read_without_schema_writes(tmp_path):
    path = tmp_path / "incidents.sqlite3"
    sqlite3.connect(path).close()
    assert DecisionStore(path).list() == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_unknown_schema_version_is_rejected_without_changes(tmp_path, decision):
    path = tmp_path / "incidents.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")
    store = DecisionStore(path)
    with pytest.raises(KnowledgeStoreError, match="schema version"):
        store.list()
    with pytest.raises(KnowledgeStoreError, match="schema version"):
        store.save(decision, approve_save=True)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_corrupted_database_reports_safe_error(tmp_path):
    path = tmp_path / "incidents.sqlite3"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(KnowledgeStoreError, match="Could not read"):
        DecisionStore(path).list()


def test_invalid_persisted_record_is_rejected(tmp_path, decision):
    path = tmp_path / "incidents.sqlite3"
    store = DecisionStore(path)
    store.save(decision, approve_save=True)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE human_decisions SET evidence_references = 'invalid JSON'")
    with pytest.raises(KnowledgeStoreError, match="invalid record"):
        store.list()


def test_environment_selects_database_without_creating_it(tmp_path, monkeypatch):
    path = tmp_path / "chosen" / "history.sqlite3"
    monkeypatch.setenv("INCIDENT_KNOWLEDGE_DB", str(path))
    store = DecisionStore()
    assert store.path == path
    assert store.list() == []
    assert not path.parent.exists()
