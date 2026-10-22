"""Check that historical retrieval changes actions without replacing current evidence."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import FakeGateway, result_for
from incident_client.compare import comparison_details
from incident_client.investigation import InvestigationEngine
from incident_client.llm import Decision, OpenAIPlanner
from incident_client.mcp_client import connect
from incident_client.policy import Action, InvestigationLimits
from incident_common.models import (
    InvestigationRequest,
    Record,
    Source,
    ToolName,
    ToolResult,
)
from incident_common.rag_models import RetrievalInfo, RetrievalMatch

HISTORY_REFERENCE = "synthetic://incidents/INC-09876"


def historical_result(*, empty=False, service="settlement", score=0.8, query="settlement 503"):
    return ToolResult(
        source=Source.HISTORY,
        transaction_id="TX784521",
        status="empty" if empty else "found",
        records=[]
        if empty
        else [
            Record(
                record_id="history-INC-09876",
                summary="Past synthetic settlement 503 incident; check current service health.",
                observed_at="2026-09-15T10:00:00Z",
                reference=HISTORY_REFERENCE,
                facts={
                    "past_incident_id": "INC-09876",
                    "service": service,
                    "symptoms": "SETTLEMENT_SENT over 24 hours and settlement HTTP 503s.",
                    "useful_check": ToolName.MONITORING.value,
                    "check_reason": "Check current health against the past service failure.",
                    "past_root_cause": "A synthetic settlement service outage.",
                    "historical": True,
                },
            )
        ],
        retrieval=RetrievalInfo(
            query=query,
            embedding_model="test-embedding-model",
            index_version="1",
            fingerprint="a" * 64,
            matches=[]
            if empty
            else [
                RetrievalMatch(incident_id="INC-09876", score=score, reference=HISTORY_REFERENCE)
            ],
        ),
    )


class RagGateway(FakeGateway):
    def __init__(self, results, **kwargs):
        super().__init__(results, **kwargs)
        self.queries = []

    async def call_tool(self, tool, transaction_id, query=None, current_incident_id=None):
        if tool == ToolName.SIMILAR:
            assert query and query.strip()
            self.queries.append(query)
        else:
            assert query is None
        return await super().call_tool(tool, transaction_id)


@pytest.fixture
def rag_results(results):
    return {**results, ToolName.SIMILAR: historical_result()}


async def test_without_rag_excludes_both_history_tools(rag_results):
    gateway = RagGateway(rag_results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest(mode="without-rag"))
    assert gateway.calls == [
        ToolName.TRANSACTION,
        ToolName.LOGS,
        ToolName.TRACES,
        ToolName.MONITORING,
        ToolName.DEPLOYMENTS,
    ]
    assert report.status == "finished"
    assert report.calls_used == 5
    assert report.retrieval is None
    assert report.historical_context == []
    assert Source.HISTORY not in report.missing_sources


async def test_matching_history_changes_next_check_and_cites_its_evidence(rag_results):
    before = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="without-rag")
    )
    gateway = RagGateway(rag_results)
    after = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert gateway.calls == [
        ToolName.TRANSACTION,
        ToolName.LOGS,
        ToolName.SIMILAR,
        ToolName.MONITORING,
        ToolName.TRACES,
        ToolName.DEPLOYMENTS,
    ]
    assert after.audit[3].decision_references == [
        rag_results[ToolName.LOGS].records[0].reference,
        HISTORY_REFERENCE,
    ]
    assert "INC-09876" in after.audit[3].reason
    assert after.calls_used == 6
    assert after.status == "finished"
    assert after.supported_failure_point == before.supported_failure_point
    assert after.retrieval == rag_results[ToolName.SIMILAR].retrieval
    assert after.historical_context == rag_results[ToolName.SIMILAR].records
    assert not after.root_cause_confirmed
    assert not after.knowledge_base_saved
    assert not after.production_changes_made
    expected_query = " ".join(
        record.summary
        for tool in (ToolName.TRANSACTION, ToolName.LOGS)
        for record in rag_results[tool].records
    )
    assert gateway.queries == [expected_query]
    assert "INC-09876" not in gateway.queries[0]
    assert comparison_details(before, after)["difference_demonstrated"] is True


@pytest.mark.parametrize(
    "retrieved", [historical_result(empty=True), historical_result(service="email")]
)
async def test_empty_or_unrelated_history_leaves_next_current_check_unchanged(
    rag_results, retrieved
):
    rag_results[ToolName.SIMILAR] = retrieved
    gateway = RagGateway(rag_results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert gateway.calls[2:4] == [ToolName.SIMILAR, ToolName.TRACES]
    assert gateway.attempts[ToolName.SIMILAR] == 1
    assert report.audit[3].decision_references == [rag_results[ToolName.LOGS].records[0].reference]
    assert report.status == "finished"
    if retrieved.status == "empty":
        assert not report.audit[2].new_evidence
        assert Source.HISTORY in report.missing_sources


async def test_past_case_and_current_logs_cannot_replace_a_current_trace(rag_results):
    rag_results[ToolName.TRACES] = result_for(ToolName.TRACES, empty=True)
    report = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="with-rag")
    )
    assert report.status == "paused"
    assert report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert report.historical_context
    assert Source.TRACES in report.missing_sources
    assert any("two-source support is missing" in cause for cause in report.open_possible_causes)


async def test_vector_search_failures_and_retries_use_the_same_eight_call_budget(
    rag_results,
):
    gateway = RagGateway(
        rag_results, fail_first={ToolName.LOGS, ToolName.SIMILAR, ToolName.MONITORING}
    )
    report = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert report.status == "paused"
    assert report.stop_reason == "call_budget"
    assert report.calls_used == len(gateway.calls) == 8
    assert gateway.attempts[ToolName.SIMILAR] == 2
    assert gateway.queries[0] == gateway.queries[1]
    assert sum(entry.status == "error" for entry in report.audit) == 3
    assert ToolName.DEPLOYMENTS not in gateway.calls
    assert [entry.call_number for entry in report.audit] == list(range(1, 9))


async def test_two_failed_vector_search_attempts_pause_with_partial_evidence(
    rag_results,
):
    gateway = RagGateway(rag_results, always_fail={ToolName.SIMILAR})
    report = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert gateway.calls == [
        ToolName.TRANSACTION,
        ToolName.LOGS,
        ToolName.SIMILAR,
        ToolName.SIMILAR,
    ]
    assert report.status == "paused"
    assert report.stop_reason == "no_new_evidence"
    assert report.calls_used == 4
    assert report.retrieval is None
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert any("whether to continue" in step for step in report.next_steps)


async def test_completed_history_search_cannot_be_repeated_with_new_metadata(rag_results):
    class ChangingMetadataGateway(RagGateway):
        async def call_tool(self, tool, transaction_id, query=None, current_incident_id=None):
            result = await super().call_tool(tool, transaction_id, query, current_incident_id)
            if tool != ToolName.SIMILAR:
                return result
            result = result.model_copy(deep=True)
            count = self.attempts[tool]
            result.records = [
                record.model_copy(update={"record_id": f"new-id-{count}"})
                for record in result.records
            ]
            result.retrieval = result.retrieval.model_copy(
                update={
                    "query": f"query variation {count}",
                    "matches": [
                        match.model_copy(update={"score": 0.6 + count / 100})
                        for match in result.retrieval.matches
                    ],
                }
            )
            return result

    class RepeatHistoryPlanner:
        def choose(self, request, memory):
            if not memory.audit:
                tool = ToolName.TRANSACTION
            elif len(memory.audit) == 1:
                tool = ToolName.LOGS
            else:
                tool = ToolName.SIMILAR
            return Action(tool, "Test whether repeated retrieval has new incident facts.")

    gateway = ChangingMetadataGateway(rag_results)
    report = await InvestigationEngine(gateway, planner=RepeatHistoryPlanner()).run(
        InvestigationRequest(mode="with-rag")
    )
    assert report.stop_reason == "planner_error"
    assert report.calls_used == 3
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS, ToolName.SIMILAR]
    assert report.retrieval.query == "query variation 1"
    assert report.historical_context[0].record_id == "new-id-1"
    assert report.supported_failure_point is None


async def test_conflicting_current_trace_overrides_matching_history(rag_results):
    rag_results[ToolName.TRACES] = result_for(
        ToolName.TRACES, service="settlement", successful_response=True
    )
    gateway = RagGateway(rag_results)
    report = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert report.status == "paused"
    assert report.stop_reason == "conflicting_evidence"
    assert report.historical_context
    assert gateway.calls[-1] == ToolName.TRACES
    assert ToolName.DEPLOYMENTS not in gateway.calls
    assert report.supported_failure_point is None
    assert not report.root_cause_confirmed


async def test_api_failure_keeps_partial_audit_and_returns_a_safe_pause(rag_results):
    parse = AsyncMock(side_effect=RuntimeError("private provider details sk-test-secret"))
    planner = OpenAIPlanner(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    gateway = RagGateway(rag_results)
    report = await InvestigationEngine(gateway, planner=planner).run(
        InvestigationRequest(mode="with-rag")
    )
    assert report.status == "paused"
    assert report.stop_reason == "planner_error"
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS, ToolName.SIMILAR]
    assert report.calls_used == 3
    assert len(report.audit) == 3
    assert report.model_calls == 1
    assert report.historical_context
    assert report.possible_failure_point == "Settlement-service call"
    assert report.supported_failure_point is None
    assert "API client error" in report.planner_error
    assert "sk-test-secret" not in report.model_dump_json()
    assert any("whether to continue" in step for step in report.next_steps)
    parse.assert_awaited_once()


async def test_exhausted_tool_budget_prevents_another_model_request(rag_results):
    parse = AsyncMock(
        return_value=SimpleNamespace(
            id="resp-synthetic",
            status="completed",
            output=[],
            output_parsed=Decision(
                tool=ToolName.TRACES,
                reason="Current settlement 503 logs need a trace check.",
                evidence_references=[rag_results[ToolName.LOGS].records[0].reference],
            ),
        )
    )
    planner = OpenAIPlanner(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    gateway = RagGateway(rag_results)
    report = await InvestigationEngine(
        gateway, limits=InvestigationLimits(max_tool_calls=3), planner=planner
    ).run(InvestigationRequest(mode="without-rag"))
    assert report.status == "paused"
    assert report.stop_reason == "call_budget"
    assert report.calls_used == 3
    assert report.model_calls == 1
    assert report.model_response_ids == ["resp-synthetic"]
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS, ToolName.TRACES]
    parse.assert_awaited_once()


async def test_comparison_does_not_claim_a_difference_when_actions_match(rag_results):
    before = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="without-rag")
    )
    rag_results[ToolName.SIMILAR] = historical_result(service="email")
    after = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="with-rag")
    )
    # A citation alone must not turn the same next tool into a claimed difference.
    after.audit[3].decision_references = [HISTORY_REFERENCE]
    details = comparison_details(before, after)
    assert details["next_check_changed"] is False
    assert details["changed_step_cites_history"] is True
    assert details["difference_demonstrated"] is False


async def test_comparison_does_not_credit_history_without_a_retrieved_citation(
    rag_results,
):
    before = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="without-rag")
    )
    after = await InvestigationEngine(RagGateway(rag_results)).run(
        InvestigationRequest(mode="with-rag")
    )
    after.audit[3].decision_references = [rag_results[ToolName.LOGS].records[0].reference]
    details = comparison_details(before, after)
    assert details["next_check_changed"] is True
    assert details["changed_step_cites_history"] is False
    assert details["difference_demonstrated"] is False


@pytest.fixture
def prepared_real_index(tmp_path, monkeypatch):
    """Use the cached real model with private test records and private index files."""
    from incident_server import rag

    original_cache = rag.get_index_path() / "model-cache"
    if not original_cache.is_dir():
        pytest.fail("Prepare the embedding model with incident-index before real-index tests.")
    directory = tmp_path / "rag"
    directory.mkdir()
    (directory / "model-cache").symlink_to(original_cache, target_is_directory=True)
    monkeypatch.setenv("INCIDENT_RAG_DIR", str(directory))
    rag.build_index(directory, local_files_only=True)
    return directory


@pytest.mark.skipif(
    os.environ.get("RUN_RAG_INTEGRATION") != "1",
    reason="Set RUN_RAG_INTEGRATION=1 after building the local embedding index.",
)
async def test_real_mcp_retrieval_changes_next_check_with_cached_embedding_index(
    prepared_real_index,
):
    async with connect(timeout_seconds=60) as gateway:
        before = await InvestigationEngine(gateway).run(InvestigationRequest(mode="without-rag"))
        after = await InvestigationEngine(gateway).run(InvestigationRequest(mode="with-rag"))
    assert before.status == after.status == "finished"
    assert before.calls_used == 5
    assert after.calls_used == 6
    assert after.retrieval.matches[0].incident_id == "INC-09876"
    assert after.retrieval.embedding_model == "BAAI/bge-small-en-v1.5"
    assert comparison_details(before, after)["difference_demonstrated"] is True
    assert after.supported_failure_point == "Settlement-service call"
    assert not after.root_cause_confirmed
    assert not after.knowledge_base_saved


@pytest.mark.skipif(
    os.environ.get("RUN_RAG_INTEGRATION") != "1",
    reason="Set RUN_RAG_INTEGRATION=1 after building the local embedding index.",
)
@pytest.mark.parametrize(
    ("query", "expected_incident"),
    [
        (
            "A billing payment has stayed SETTLEMENT_SENT for over a day. Calls to settlement "
            "return 503 Service Unavailable and retries never succeed, but billing is healthy.",
            "INC-09876",
        ),
        (
            "After a billing release, card authorization returns 500 and a local null pointer "
            "exception before it reaches the card provider.",
            "INC-09741",
        ),
        (
            "Balance updates time out while waiting on account row locks held by a long "
            "reconciliation transaction. Database lock waits are high.",
            "INC-09632",
        ),
        (
            "Refund calls receive 401 Unauthorized because their access token is expired. "
            "The provider accepts requests made using a fresh token.",
            "INC-09508",
        ),
        (
            "Payment status is completed, but customers wait for receipt emails. The email "
            "queue is growing and notification workers are no longer reading messages.",
            "INC-09402",
        ),
        ("How do I bake a chocolate cake with butter and flour?", None),
        ("Will it rain tomorrow and what is the weather forecast?", None),
    ],
)
def test_real_embedding_retrieval_evaluation(query, expected_incident, prepared_real_index):
    # Load the real cached model and persisted FAISS index. No mocked vectors or API model.
    from incident_server.rag import get_index

    retrieved = get_index().search(query, top_k=2)
    if expected_incident is None:
        assert retrieved.matches == []
    else:
        assert retrieved.matches
        assert retrieved.matches[0].incident_id == expected_incident
