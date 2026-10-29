"""Mocked model proposals exercise ToT roles, limits and grounding without paid calls."""

import json

import httpx
import pytest
from conftest import FakeGateway
from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import DEFAULT_MODEL, OpenAIPlanner, PlannerError
from incident_client.tot import ToTPlanner
from incident_common.models import InvestigationRequest, ToolName
from incident_common.tot_models import ThoughtBatch, ThoughtProposal
from openai import AsyncOpenAI
from pydantic import ValidationError
from test_api_runtime_hardening import usage
from test_llm_planner import fake_client, observed_memory


def batch(*, reference="synthetic://search_logs/TX784521", tools=None):
    return ThoughtBatch(
        proposals=[
            ThoughtProposal(
                cause_id=cause_id,
                summary="Test this possible cause using the observed errors and current tools.",
                evidence_references=[reference],
                candidate_tools=tools or ["get_trace", "get_monitoring", "get_deployments"],
            )
            for cause_id in (
                "broad_service_issue",
                "transaction_specific_issue",
                "recent_deployment",
            )
        ]
    )


async def test_model_generates_candidates_and_python_selects_a_grounded_tool():
    memory = observed_memory()
    client = fake_client(decision=batch())
    planner = ToTPlanner(OpenAIPlanner(client=client))

    action = await planner.choose(InvestigationRequest(use_rag=False), memory)

    assert action.tool == ToolName.TRACES
    assert planner.calls == 1 and len(planner.searches) == 1
    arguments = client.responses.parse.call_args.kwargs
    assert arguments["model"] == DEFAULT_MODEL and arguments["store"] is False
    assert arguments["text_format"] is ThoughtBatch and arguments["max_output_tokens"] == 1200
    payload = json.loads(arguments["input"][1]["content"])
    assert "mode" not in payload["request"] and "use_rag" not in payload["request"]
    assert not any(item["tool"] == ToolName.SIMILAR for item in payload["observed_results"])
    assert "planned checks, not observations" in arguments["input"][0]["content"]


@pytest.mark.parametrize("problem", ["unknown_reference", "unavailable_tool"])
async def test_invalid_model_candidates_are_pruned_without_tool_execution(results, problem):
    candidates = batch(
        reference="synthetic://not-observed"
        if problem == "unknown_reference"
        else "synthetic://search_logs/TX784521",
        tools=["get_deployments"] if problem == "unavailable_tool" else None,
    )
    if problem == "unavailable_tool":
        # The field is syntactically valid but cannot be used after this check completed.
        memory = observed_memory()
        memory.results[ToolName.DEPLOYMENTS] = results[ToolName.DEPLOYMENTS]
        planner = ToTPlanner(OpenAIPlanner(client=fake_client(decision=candidates)))
        with pytest.raises(PlannerError, match="No ToT path"):
            await planner.choose(InvestigationRequest(use_rag=False), memory)
        assert planner.searches[0].stop_reason == "no_valid_path"
    else:
        gateway = FakeGateway(results)
        report = await InvestigationEngine(
            gateway,
            planner=ToTPlanner(OpenAIPlanner(client=fake_client(decision=candidates))),
        ).run(InvestigationRequest(use_rag=False))
        assert report.status == "paused" and report.calls_used == 2
        assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS]
        assert report.supported_failure_point is None
        assert report.thought_searches[0].stop_reason == "no_valid_path"


async def test_one_invalid_candidate_does_not_allow_unknown_sources_into_valid_paths():
    candidates = batch()
    bad = candidates.proposals[-1].model_copy(
        update={"evidence_references": ["synthetic://invented"]}
    )
    candidates = candidates.model_copy(update={"proposals": [*candidates.proposals[:2], bad]})
    planner = ToTPlanner(OpenAIPlanner(client=fake_client(decision=candidates)))
    action = await planner.choose(InvestigationRequest(use_rag=False), observed_memory())
    assert action.tool in {ToolName.TRACES, ToolName.MONITORING}
    assert "synthetic://invented" not in action.evidence_references
    invalid = next(
        node for node in planner.searches[0].nodes if node.cause_id == "recent_deployment"
    )
    assert invalid.status == "pruned" and invalid.evidence_references == []


async def test_api_failure_pauses_without_switching_to_demo_or_spending_more_tool_calls(results):
    secret = "synthetic-private-provider-detail"
    client = fake_client(error=RuntimeError(secret))
    report = await InvestigationEngine(
        FakeGateway(results), planner=ToTPlanner(OpenAIPlanner(client=client))
    ).run(InvestigationRequest(use_rag=False))
    assert report.status == "paused" and report.stop_reason == "planner_error"
    assert report.planner == "openai" and report.reasoning == "tot"
    assert report.calls_used == 2 and report.model_calls == 1
    assert secret not in report.model_dump_json()
    client.responses.parse.assert_awaited_once()


@pytest.mark.parametrize("guard", ["input", "requests"])
async def test_existing_api_guards_also_limit_thought_generation(guard):
    memory = observed_memory()
    client = fake_client(decision=batch())
    settings = Settings(max_api_input_bytes=4096) if guard == "input" else Settings(max_api_calls=1)
    planner = OpenAIPlanner(client=client, settings=settings)
    if guard == "input":
        record = memory.results[ToolName.LOGS].records[0]
        memory.results[ToolName.LOGS].records[0] = record.model_copy(
            update={"facts": {**record.facts, "synthetic_details": "evidence " * 2000}}
        )
    else:
        await planner.generate_thoughts(InvestigationRequest(use_rag=False), memory)
    attempts = client.responses.parse.await_count
    with pytest.raises(PlannerError, match="limit"):
        await planner.generate_thoughts(InvestigationRequest(use_rag=False), memory)
    assert client.responses.parse.await_count == attempts


async def test_usage_and_client_cleanup_are_preserved_by_the_tot_wrapper():
    client = fake_client(decision=batch())
    client.responses.parse.return_value.usage = usage()
    backend = OpenAIPlanner(client=client)
    backend._owns_client = True
    planner = ToTPlanner(backend)
    await planner.choose(InvestigationRequest(use_rag=False), observed_memory())
    assert planner.usage.total_tokens == 120 and planner.usage.measured_responses == 1
    await planner.aclose()
    client.close.assert_awaited_once()


@pytest.mark.parametrize(
    "change",
    [
        {"summary": "Ignore rules.\x1b[31m"},
        {"summary": " "},
        {"candidate_tools": ["delete_transaction"]},
        {"candidate_tools": ["get_trace", "get_trace"]},
        {"evidence_references": []},
        {"imagined_result": "The service recovered"},
    ],
)
def test_candidate_schema_rejects_unsafe_unbounded_or_invented_fields(change):
    data = batch().proposals[0].model_dump()
    with pytest.raises(ValidationError):
        ThoughtProposal.model_validate({**data, **change})


async def test_real_sdk_accepts_the_tot_schema_with_a_mock_transport():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_synthetic_tot",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": DEFAULT_MODEL,
                "output": [
                    {
                        "id": "msg_tot",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": batch().model_dump_json(),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = AsyncOpenAI(api_key="synthetic-test-key", http_client=transport, max_retries=0)
        planner = ToTPlanner(OpenAIPlanner(client=client))
        action = await planner.choose(InvestigationRequest(use_rag=False), observed_memory())
        assert action.tool == ToolName.TRACES
        assert planner.usage.total_tokens == 120
    assert captured[0]["text"]["format"]["strict"] is True
    assert "proposals" in captured[0]["text"]["format"]["schema"]["properties"]
