"""SQLite-to-FAISS integration using real files and small, offline embeddings."""

import json

import numpy as np
import pytest
from incident_common.knowledge_models import HumanDecisionInput
from incident_server import knowledge_cli, rag
from incident_server.knowledge import (
    DecisionConflictError,
    DecisionStore,
    KnowledgeStoreError,
)


class ContentEmbeddings:
    """Choose a vector from content, so appended cases work without fixture positions."""

    @staticmethod
    def _vector(text):
        text = text.casefold()
        topics = (
            "recoverymarker",
            "settlement",
            "authorization",
            "database lock",
            "refund",
            "receipt",
        )
        position = next((i for i, topic in enumerate(topics) if topic in text), 10)
        vector = np.zeros(rag.EMBEDDING_DIMENSION, dtype=np.float32)
        vector[position] = 2
        return vector

    def passage_embed(self, documents):
        return (self._vector(document) for document in documents)

    def query_embed(self, query):
        return iter([self._vector(query)])


@pytest.fixture
def local_knowledge(tmp_path, monkeypatch):
    database = tmp_path / "knowledge" / "incidents.sqlite3"
    index = tmp_path / "rag"
    monkeypatch.setenv("INCIDENT_KNOWLEDGE_DB", str(database))
    monkeypatch.setenv("INCIDENT_RAG_DIR", str(index))
    model_calls = []

    def cached_model(directory, *, local_files_only):
        model_calls.append((directory, local_files_only))
        return ContentEmbeddings()

    monkeypatch.setattr(rag, "_embedding_model", cached_model)
    rag._cached_index.cache_clear()
    yield database, index, model_calls
    rag._cached_index.cache_clear()


@pytest.fixture
def approved_decision():
    return HumanDecisionInput(
        incident_id="INC-10801",
        title="Synthetic recoverymarker settlement delay",
        symptoms="Recoverymarker: transaction remained pending with downstream HTTP 503 errors.",
        service="settlement",
        human_decision="The engineer decided to wait for confirmed service recovery before replay.",
        action_taken="The engineer manually replayed the synthetic transaction after recovery.",
        outcome="resolved",
        approved_by="Synthetic on-call engineer",
        confirmed_root_cause="The engineer confirmed a synthetic downstream service outage.",
        occurred_at="2026-10-22T10:00:00Z",
        evidence_references=[
            "synthetic://logs/TX123456/503",
            "synthetic://traces/TX123456",
        ],
        useful_check="get_monitoring",
        check_reason="Check current service recovery before suggesting any manual replay.",
    )


def test_saved_decision_and_action_survive_database_reopen_and_vector_retrieval(
    local_knowledge, approved_decision
):
    database, index, model_calls = local_knowledge
    result = knowledge_cli.save_and_index(approved_decision, approve_save=True)

    assert result["status"] == "saved"
    assert result["database_path"] == str(database)
    assert result["search_index_updated"] is True
    assert result["index_error"] is None
    saved = DecisionStore().get(approved_decision.incident_id)
    assert saved.human_decision == approved_decision.human_decision
    assert saved.action_taken == approved_decision.action_taken
    assert result["record"] == saved.model_dump(mode="json")
    assert model_calls == [(index, True)]

    tool_result = rag.search_similar_incidents("TX784521", "recoverymarker HTTP 503")
    assert tool_result.status == "found"
    assert len(tool_result.records) == 1
    record = tool_result.records[0]
    assert tool_result.retrieval.matches[0].incident_id == approved_decision.incident_id
    assert record.reference == f"synthetic://incidents/{approved_decision.incident_id}"
    assert record.facts["human_decision"] == approved_decision.human_decision
    assert record.facts["action_taken"] == approved_decision.action_taken
    assert record.facts["outcome"] == "resolved"
    assert record.facts["past_root_cause"] == approved_decision.confirmed_root_cause
    assert record.facts["approved_by"] == approved_decision.approved_by
    assert record.facts["evidence_references"] == approved_decision.evidence_references
    assert record.facts["saved_at"] == saved.saved_at.isoformat()
    assert record.facts["record_source"] == "human_approved"
    assert record.facts["historical"] is True
    assert approved_decision.human_decision in record.summary
    assert approved_decision.action_taken in record.summary
    assert "does not confirm the current cause" in record.summary
    assert "transaction_id" not in record.facts
    assert "root_cause_confirmed" not in record.facts
    assert "knowledge_base_saved" not in record.facts


def test_loaded_corpus_keeps_five_seeds_and_orders_saved_records(
    local_knowledge, approved_decision
):
    database, _, _ = local_knowledge
    seeds = rag.load_seed_cases()
    assert rag.load_cases() == seeds
    assert not database.exists()
    for incident_id in ("INC-10802", "INC-10234", "INC-10801"):
        DecisionStore().save(
            approved_decision.model_copy(update={"incident_id": incident_id}),
            approve_save=True,
        )

    cases = rag.load_cases()
    assert cases[:5] == seeds
    assert [case.incident_id for case in cases[5:]] == [
        "INC-10234",
        "INC-10801",
        "INC-10802",
    ]
    assert all(case.record_source == "human_approved" for case in cases[5:])
    saved = cases[5]
    assert saved.past_root_cause == approved_decision.confirmed_root_cause
    assert approved_decision.action_taken not in saved.embedding_text()
    assert approved_decision.human_decision not in saved.embedding_text()


def test_current_incident_is_saved_but_excluded_from_its_own_retrieval(
    local_knowledge, approved_decision
):
    _, index, _ = local_knowledge
    for incident_id in ("INC-10234", "INC-10801"):
        DecisionStore().save(
            approved_decision.model_copy(update={"incident_id": incident_id}),
            approve_save=True,
        )
    rag.build_index(local_files_only=True)
    assert DecisionStore().get("INC-10234") is not None
    retrieved = rag.IncidentIndex.load(index).search(
        "recoverymarker", top_k=2, exclude_incident_id="INC-10234"
    )
    assert [match.incident_id for match in retrieved.matches] == ["INC-10801"]
    result = rag.search_similar_incidents(
        "TX784521", "recoverymarker", current_incident_id="INC-10234"
    )
    assert [record.facts["past_incident_id"] for record in result.records] == ["INC-10801"]


def test_new_database_record_invalidates_already_cached_index_until_rebuilt(
    local_knowledge, approved_decision
):
    _, index, model_calls = local_knowledge
    rag.build_index(local_files_only=True)
    assert rag.get_index().metadata.incident_ids == [
        case.incident_id for case in rag.load_seed_cases()
    ]
    assert len(model_calls) == 2

    DecisionStore().save(approved_decision, approve_save=True)
    with pytest.raises(rag.RagIndexError, match="stale"):
        rag.search_similar_incidents("TX784521", "recoverymarker")
    # Stale metadata is rejected before any additional embedding model load.
    assert len(model_calls) == 2
    rag.build_index(local_files_only=True)
    retrieved = rag.search_similar_incidents("TX784521", "recoverymarker")
    assert retrieved.retrieval.matches[0].incident_id == approved_decision.incident_id
    metadata = json.loads(rag._index_files(index)[0].read_text())
    assert len(metadata["incident_ids"]) == 6


def test_index_failure_keeps_approved_record_durable_and_reports_rebuild_needed(
    local_knowledge, approved_decision, monkeypatch
):
    _, _, _ = local_knowledge
    rag.build_index(local_files_only=True)

    def failed_rebuild(*, local_files_only):
        assert local_files_only is True
        raise rag.RagIndexError("Synthetic cached-model failure.")

    monkeypatch.setattr(knowledge_cli, "build_index", failed_rebuild)
    result = knowledge_cli.save_and_index(approved_decision, approve_save=True)
    assert result["status"] == "saved"
    assert result["search_index_updated"] is False
    assert "decision is saved" in result["index_error"]
    assert "incident-index" in result["index_error"]
    assert (
        DecisionStore().get(approved_decision.incident_id).action_taken
        == approved_decision.action_taken
    )
    with pytest.raises(rag.RagIndexError, match="stale"):
        rag.get_index()
    rag.build_index(local_files_only=True)
    assert rag.search_similar_incidents("TX784521", "recoverymarker").status == "found"


def test_repeat_save_does_not_duplicate_or_change_corpus_fingerprint(
    local_knowledge, approved_decision
):
    first = knowledge_cli.save_and_index(approved_decision, approve_save=True)
    fingerprint = rag.get_index().metadata.fingerprint
    second = knowledge_cli.save_and_index(approved_decision, approve_save=True)
    assert second["status"] == "already_saved"
    assert second["record"] == first["record"]
    assert len(DecisionStore().list()) == 1
    assert len(rag.load_cases()) == 6
    assert rag.get_index().metadata.fingerprint == fingerprint


def test_conflicting_save_preserves_database_record_and_search_index(
    local_knowledge, approved_decision
):
    first = knowledge_cli.save_and_index(approved_decision, approve_save=True)
    fingerprint = rag.get_index().metadata.fingerprint
    different = approved_decision.model_copy(update={"action_taken": "Different synthetic action."})
    with pytest.raises(DecisionConflictError, match="not changed"):
        knowledge_cli.save_and_index(different, approve_save=True)
    assert (
        DecisionStore().get(approved_decision.incident_id).model_dump(mode="json")
        == first["record"]
    )
    assert rag.get_index().metadata.fingerprint == fingerprint


def test_seed_id_save_rejected_without_creating_database_or_index(
    local_knowledge, approved_decision
):
    database, index, model_calls = local_knowledge
    seed_id = rag.load_seed_cases()[0].incident_id
    with pytest.raises(DecisionConflictError, match="seed"):
        knowledge_cli.save_and_index(
            approved_decision.model_copy(update={"incident_id": seed_id}),
            approve_save=True,
        )
    assert not database.exists()
    assert not index.exists()
    assert model_calls == []


def test_unapproved_save_cannot_write_database_or_index(local_knowledge, approved_decision):
    database, index, model_calls = local_knowledge
    with pytest.raises(KnowledgeStoreError, match="approval"):
        knowledge_cli.save_and_index(approved_decision, approve_save=False)
    assert not database.exists()
    assert not index.exists()
    assert model_calls == []


def test_pending_action_can_be_retrieved_without_claiming_confirmed_cause(
    local_knowledge, approved_decision
):
    pending = approved_decision.model_copy(
        update={
            "outcome": "pending",
            "confirmed_root_cause": None,
            "action_taken": "The engineer contacted the service owner; no replay was made.",
        }
    )
    result = knowledge_cli.save_and_index(pending, approve_save=True)
    assert result["record"]["confirmed_root_cause"] is None
    retrieved = rag.search_similar_incidents("TX784521", "recoverymarker")
    assert retrieved.records[0].facts["outcome"] == "pending"
    assert retrieved.records[0].facts["past_root_cause"] is None
    assert retrieved.records[0].facts["action_taken"] == pending.action_taken
    assert "does not confirm the current cause" in retrieved.records[0].summary
