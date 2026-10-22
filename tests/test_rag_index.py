"""Real FAISS persistence checks with tiny fake vectors; tests need no downloads."""

import json

import numpy as np
import pytest
from incident_server import rag
from pydantic import ValidationError


class FakeEmbeddings:
    def passage_embed(self, documents):
        for position, _ in enumerate(documents):
            vector = np.zeros(rag.EMBEDDING_DIMENSION, dtype=np.float32)
            vector[position] = 2
            yield vector

    def query_embed(self, query):
        vector = np.zeros(rag.EMBEDDING_DIMENSION, dtype=np.float32)
        vector[0 if "settlement" in query else 10] = 3
        yield vector


@pytest.fixture
def fake_model(monkeypatch):
    loaded = []

    def model_factory(directory, *, local_files_only):
        loaded.append(local_files_only)
        return FakeEmbeddings()

    monkeypatch.setattr(rag, "_embedding_model", model_factory)
    rag._cached_index.cache_clear()
    return loaded


def test_five_distinct_synthetic_cases_exclude_current_incident():
    cases = rag.load_cases()
    assert len(cases) == 5
    assert len({case.incident_id for case in cases}) == 5
    assert "INC-09876" in {case.incident_id for case in cases}
    assert "INC-10234" not in {case.incident_id for case in cases}
    assert all(case.synthetic for case in cases)
    for case in cases:
        assert case.past_root_cause not in case.embedding_text()
        assert case.check_reason not in case.embedding_text()


def test_persisted_index_uses_cosine_and_search_loads_only_cached_model(tmp_path, fake_model):
    rag.build_index(tmp_path)
    index = rag.IncidentIndex.load(tmp_path)
    retrieved = index.search("settlement HTTP 503", top_k=2)
    assert fake_model == [False, True]
    assert len(retrieved.matches) == 1
    assert retrieved.matches[0].incident_id == "INC-09876"
    assert retrieved.matches[0].score == pytest.approx(1)
    assert retrieved.embedding_model == rag.EMBEDDING_MODEL
    assert retrieved.fingerprint == index.metadata.fingerprint
    assert rag._index_files(tmp_path)[1].is_file()


def test_unrelated_query_returns_empty_matches_not_failure(tmp_path, fake_model):
    rag.build_index(tmp_path)
    assert rag.IncidentIndex.load(tmp_path).search("weather and gardening").matches == []


def test_missing_index_fails_before_loading_model(tmp_path, fake_model):
    with pytest.raises(rag.RagIndexError, match="incident-index"):
        rag.IncidentIndex.load(tmp_path)
    assert fake_model == []


def test_corrupted_index_is_rejected_before_faiss_deserialization(
    tmp_path, fake_model, monkeypatch
):
    import faiss

    rag.build_index(tmp_path)
    rag._index_files(tmp_path)[1].write_bytes(b"damaged")

    def forbidden_load(_):
        pytest.fail("Corrupt bytes reached the FAISS decoder.")

    monkeypatch.setattr(faiss, "deserialize_index", forbidden_load)
    with pytest.raises(rag.RagIndexError, match="damaged"):
        rag.IncidentIndex.load(tmp_path)
    assert fake_model == [False]


def test_stale_corpus_metadata_is_rejected(tmp_path, fake_model):
    rag.build_index(tmp_path)
    metadata_path, _ = rag._index_files(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["corpus_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(rag.RagIndexError, match="stale"):
        rag.IncidentIndex.load(tmp_path)


def test_search_tool_keeps_scores_out_of_evidence_facts(tmp_path, fake_model, monkeypatch):
    rag.build_index(tmp_path)
    monkeypatch.setenv("INCIDENT_RAG_DIR", str(tmp_path))
    result = rag.search_similar_incidents("TX784521", "settlement HTTP 503")
    assert result.status == "found"
    assert result.retrieval.matches[0].incident_id == "INC-09876"
    assert result.records[0].facts["historical"] is True
    assert "score" not in result.records[0].facts
    assert "transaction_id" not in result.records[0].facts
    assert result.records[0].reference == "synthetic://incidents/INC-09876"
    empty = rag.search_similar_incidents("TX784521", "weather and gardening")
    assert empty.status == "empty"
    assert empty.records == []
    assert empty.retrieval.matches == []


@pytest.mark.parametrize("top_k", [0, 6, True])
def test_search_rejects_invalid_result_limit(tmp_path, fake_model, top_k):
    rag.build_index(tmp_path)
    with pytest.raises(ValidationError):
        rag.IncidentIndex.load(tmp_path).search("settlement", top_k=top_k)


def test_search_rejects_blank_query(tmp_path, fake_model):
    rag.build_index(tmp_path)
    with pytest.raises(ValidationError):
        rag.IncidentIndex.load(tmp_path).search("   ")
