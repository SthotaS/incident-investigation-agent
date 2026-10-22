"""Bounded, approved knowledge saves under real SQLite write contention."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from incident_common.knowledge_models import HumanDecisionInput
from incident_server import knowledge_cli
from incident_server.knowledge import DecisionConflictError, DecisionStore, KnowledgeStoreError
from pydantic import ValidationError


@pytest.fixture
def decision():
    return HumanDecisionInput(
        incident_id="INC-11001",
        title="Synthetic settlement delay",
        symptoms="Settlement requests returned HTTP 503.",
        service="settlement",
        human_decision="Ask the service team to review current errors.",
        action_taken="No recovery action yet.",
        approved_by="Synthetic engineer",
    )


def test_capacity_rejects_new_record_without_changing_saved_decision(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3", max_records=1)
    saved = store.save(decision, approve_save=True).record
    extra = decision.model_copy(update={"incident_id": "INC-11002"})

    with pytest.raises(KnowledgeStoreError, match="limit.*No record was written"):
        store.save(extra, approve_save=True)

    assert store.list() == [saved]
    assert store.get(extra.incident_id) is None
    assert store.save(decision, approve_save=True).record == saved
    assert store.save(decision, approve_save=True).status == "already_saved"


def test_conflict_still_reports_existing_record_at_capacity(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3", max_records=1)
    store.save(decision, approve_save=True)
    changed = decision.model_copy(update={"action_taken": "A different approved action."})
    with pytest.raises(DecisionConflictError, match="not changed"):
        store.save(changed, approve_save=True)


def test_concurrent_new_saves_cannot_exceed_capacity(tmp_path, decision):
    path = tmp_path / "incidents.sqlite3"
    writers = 10
    barrier = Barrier(writers)

    def save_one(number):
        barrier.wait(timeout=10)
        payload = decision.model_copy(update={"incident_id": f"INC-{11000 + number:05d}"})
        try:
            return DecisionStore(path, max_records=3).save(payload, approve_save=True).status
        except KnowledgeStoreError as exc:
            assert "limit" in str(exc)
            return "full"

    with ThreadPoolExecutor(max_workers=writers) as pool:
        results = list(pool.map(save_one, range(writers)))

    assert results.count("saved") == 3
    assert results.count("full") == 7
    assert len(DecisionStore(path).list()) == 3


def test_concurrent_identical_saves_are_idempotent_at_capacity(tmp_path, decision):
    path = tmp_path / "incidents.sqlite3"
    barrier = Barrier(2)

    def save_one(_):
        barrier.wait(timeout=10)
        return DecisionStore(path, max_records=1).save(decision, approve_save=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save_one, range(2)))

    assert sorted(result.status for result in results) == ["already_saved", "saved"]
    assert results[0].record == results[1].record
    assert len(DecisionStore(path).list()) == 1


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, "2"])
def test_invalid_capacity_is_rejected_without_creating_data(tmp_path, maximum):
    path = tmp_path / "missing" / "incidents.sqlite3"
    with pytest.raises(ValueError, match="positive integer"):
        DecisionStore(path, max_records=maximum)
    assert not path.parent.exists()


def test_list_can_read_a_bounded_ordered_sample(tmp_path, decision):
    store = DecisionStore(tmp_path / "incidents.sqlite3")
    for incident_id in ("INC-11003", "INC-11001", "INC-11002"):
        store.save(decision.model_copy(update={"incident_id": incident_id}), approve_save=True)

    assert [item.incident_id for item in store.list(limit=2)] == ["INC-11001", "INC-11002"]
    assert len(store.list()) == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_read_limit_is_rejected_without_creating_data(tmp_path, limit):
    path = tmp_path / "missing" / "incidents.sqlite3"
    with pytest.raises(ValueError, match="positive integer"):
        DecisionStore(path).list(limit=limit)
    assert not path.parent.exists()


def test_save_cli_reserves_capacity_for_seed_cases(tmp_path, monkeypatch, decision):
    database = tmp_path / "incidents.sqlite3"
    monkeypatch.setenv("INCIDENT_KNOWLEDGE_DB", str(database))
    seed_count = len(knowledge_cli.load_seed_cases())
    monkeypatch.setattr(knowledge_cli, "MAX_CASES", seed_count + 1)
    index_updates = []
    monkeypatch.setattr(knowledge_cli, "build_index", lambda **kwargs: index_updates.append(kwargs))

    assert knowledge_cli.save_and_index(decision, approve_save=True)["search_index_updated"]
    extra = decision.model_copy(update={"incident_id": "INC-11002"})
    with pytest.raises(KnowledgeStoreError, match="limit"):
        knowledge_cli.save_and_index(extra, approve_save=True)

    assert len(index_updates) == 1
    assert len(DecisionStore(database).list()) == 1


def test_reference_size_and_count_limits(decision):
    payload = decision.model_dump()
    valid = [f"synthetic://logs/{number}" for number in range(32)]
    assert (
        len(
            HumanDecisionInput.model_validate(
                {**payload, "evidence_references": valid}
            ).evidence_references
        )
        == 32
    )
    for references in (valid + ["synthetic://extra"], ["synthetic://" + "x" * 502]):
        with pytest.raises(ValidationError):
            HumanDecisionInput.model_validate({**payload, "evidence_references": references})


def test_repeated_references_are_trimmed_and_removed(decision):
    saved = HumanDecisionInput.model_validate(
        {
            **decision.model_dump(),
            "evidence_references": [
                " synthetic://logs/TX784521/503 ",
                "synthetic://logs/TX784521/503",
                "synthetic://traces/TX784521",
            ],
        }
    )
    assert saved.evidence_references == [
        "synthetic://logs/TX784521/503",
        "synthetic://traces/TX784521",
    ]


@pytest.mark.parametrize("control", ["\x00", "\x08", "\r", "\x1b[2J", "\x7f", "\x85", "\x9b"])
def test_unsafe_controls_are_rejected_in_human_text(decision, control):
    for field in (
        "title",
        "symptoms",
        "service",
        "human_decision",
        "action_taken",
        "approved_by",
        "confirmed_root_cause",
        "check_reason",
    ):
        with pytest.raises(ValidationError, match="control characters"):
            HumanDecisionInput.model_validate({**decision.model_dump(), field: f"a{control}b"})


@pytest.mark.parametrize("control", ["\x00", "\x1b", "\x7f", "\x85", "\x9b", "\n", "\t"])
def test_controls_and_whitespace_are_rejected_in_references(decision, control):
    with pytest.raises(ValidationError):
        HumanDecisionInput.model_validate(
            {**decision.model_dump(), "evidence_references": [f"synthetic://a{control}b"]}
        )


def test_notes_allow_newlines_tabs_and_normalize_windows_line_endings(tmp_path, decision):
    payload = decision.model_dump()
    payload["action_taken"] = " Reviewed errors.\r\n\tContacted the service team. "
    saved = DecisionStore(tmp_path / "incidents.sqlite3").save(payload, approve_save=True).record
    assert saved.action_taken == "Reviewed errors.\n\tContacted the service team."
    assert saved.confirmed_root_cause is None
    assert saved.outcome == "unknown"
