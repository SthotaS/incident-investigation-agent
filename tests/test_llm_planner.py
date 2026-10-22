"""Mock the API boundary; these checks neither spend API credits nor claim a live run."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import result_for
from incident_client.llm import DEFAULT_MODEL, Decision, OpenAIPlanner, PlannerError
from incident_client.memory import InvestigationMemory
from incident_common.models import InvestigationRequest, ToolName


def fake_client(*, decision=None, status="completed", output=None, error=None):
    response = SimpleNamespace(
        id="resp_mock_001",
        status=status,
        output=output or [],
        output_parsed=decision,
    )
    return SimpleNamespace(
        responses=SimpleNamespace(parse=AsyncMock(return_value=response, side_effect=error)),
        close=AsyncMock(),
    )


def observed_memory(*, with_history=False):
    memory = InvestigationMemory()
    memory.results[ToolName.TRANSACTION] = result_for(
        ToolName.TRANSACTION, status="SETTLEMENT_SENT", age_hours=26
    )
    memory.results[ToolName.LOGS] = result_for(ToolName.LOGS, service="settlement", http_status=503)
    if with_history:
        memory.results[ToolName.SIMILAR] = result_for(
            ToolName.SIMILAR,
            incident_id="INC-10001",
            symptoms="Settlement HTTP 503s after SETTLEMENT_SENT",
            suggested_next_check="get_monitoring",
            root_cause="Historical settlement capacity issue, not current proof.",
        )
    return memory


def decision_for(tool, *references):
    return Decision(
        tool=tool,
        reason="Use the observed settlement errors to check the next source.",
        evidence_references=list(references),
    )


async def test_api_planner_uses_observed_history_and_checked_references():
    memory = observed_memory(with_history=True)
    log_ref = memory.results[ToolName.LOGS].records[0].reference
    history_ref = memory.results[ToolName.SIMILAR].records[0].reference
    client = fake_client(decision=decision_for(ToolName.MONITORING, log_ref, history_ref))
    planner = OpenAIPlanner(client=client)

    action = await planner.choose(InvestigationRequest(mode="with-rag"), memory)

    assert action.tool == ToolName.MONITORING
    assert action.evidence_references == (log_ref, history_ref)
    arguments = client.responses.parse.call_args.kwargs
    assert arguments["model"] == DEFAULT_MODEL
    assert arguments["store"] is False
    assert arguments["text_format"] is Decision
    assert arguments["max_output_tokens"] == 700
    payload = json.loads(arguments["input"][1]["content"])
    assert "mode" not in payload["request"]
    historical = next(
        item for item in payload["observed_results"] if item["tool"] == ToolName.SIMILAR
    )
    assert historical["result"]["records"][0]["facts"]["suggested_next_check"] == "get_monitoring"
    assert planner.calls == 1
    assert planner.response_ids == ["resp_mock_001"]
    assert planner.error is None


async def test_same_instructions_without_rag_have_no_unseen_history():
    memory = observed_memory()
    log_ref = memory.results[ToolName.LOGS].records[0].reference
    baseline = fake_client(decision=decision_for(ToolName.TRACES, log_ref))
    await OpenAIPlanner(client=baseline).choose(InvestigationRequest(mode="without-rag"), memory)
    with_history = observed_memory(with_history=True)
    rag = fake_client(decision=decision_for(ToolName.MONITORING, log_ref))
    await OpenAIPlanner(client=rag).choose(InvestigationRequest(mode="with-rag"), with_history)

    baseline_input = baseline.responses.parse.call_args.kwargs["input"]
    rag_input = rag.responses.parse.call_args.kwargs["input"]
    assert baseline_input[0] == rag_input[0]
    payload = json.loads(baseline_input[1]["content"])
    assert all(
        item["tool"] not in (ToolName.SIMILAR, ToolName.HISTORY)
        for item in payload["observed_results"]
    )
    assert "INC-10001" not in baseline_input[1]["content"]


async def test_setup_steps_and_completion_do_not_make_api_calls():
    client = fake_client()
    planner = OpenAIPlanner(client=client)
    request = InvestigationRequest(mode="with-rag")
    memory = InvestigationMemory()
    action = await planner.choose(request, memory)
    assert action.tool == ToolName.TRANSACTION
    memory.results[ToolName.TRANSACTION] = result_for(ToolName.TRANSACTION)
    action = await planner.choose(request, memory)
    assert action.tool == ToolName.LOGS
    memory.results[ToolName.LOGS] = result_for(ToolName.LOGS)
    action = await planner.choose(request, memory)
    assert action.tool == ToolName.SIMILAR
    for tool in (ToolName.SIMILAR, ToolName.TRACES, ToolName.MONITORING, ToolName.DEPLOYMENTS):
        memory.results[tool] = result_for(tool)
    assert await planner.choose(request, memory) is None
    assert planner.calls == 0
    client.responses.parse.assert_not_called()


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (
            decision_for(ToolName.LOGS, "synthetic://search_logs/TX784521"),
            "not available",
        ),
        (
            decision_for(ToolName.TRACES, "synthetic://unseen/INC-99999"),
            "has not observed",
        ),
        (
            Decision.model_construct(
                tool=ToolName.TRACES, reason="Check traces", evidence_references=[]
            ),
            "has not observed",
        ),
    ],
)
async def test_invalid_choices_pause_without_hidden_fallback(decision, message):
    client = fake_client(decision=decision)
    planner = OpenAIPlanner(client=client)
    with pytest.raises(PlannerError, match=message):
        await planner.choose(InvestigationRequest(mode="without-rag"), observed_memory())
    assert planner.calls == 1
    assert message in planner.error
    client.responses.parse.assert_awaited_once()


@pytest.mark.parametrize(
    "response_options",
    [
        {"output": [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]},
        {"status": "incomplete"},
        {},
        {"error": RuntimeError("sensitive provider body sk-private-value")},
    ],
)
async def test_api_failure_refusal_and_invalid_output_are_safe_and_counted(response_options):
    client = fake_client(**response_options)
    planner = OpenAIPlanner(client=client)
    with pytest.raises(PlannerError) as failure:
        await planner.choose(InvestigationRequest(mode="without-rag"), observed_memory())
    assert "sk-private-value" not in str(failure.value)
    assert "sensitive provider body" not in str(failure.value)
    assert planner.calls == 1
    assert planner.error == str(failure.value)
    client.responses.parse.assert_awaited_once()


async def test_injected_client_remains_owned_by_caller():
    client = fake_client()
    await OpenAIPlanner(client=client).aclose()
    client.close.assert_not_called()


async def test_owned_client_disables_sdk_retries_and_closes(monkeypatch):
    client = fake_client()
    construction = {}

    def make_client(**kwargs):
        construction.update(kwargs)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model-override")
    monkeypatch.setattr("incident_client.llm.AsyncOpenAI", make_client)
    planner = OpenAIPlanner()
    assert planner.model == "test-model-override"
    assert construction["max_retries"] == 0
    assert construction["timeout"] == 30.0
    assert planner.calls == 0
    await planner.aclose()
    client.close.assert_awaited_once()


def test_missing_api_key_has_a_clear_safe_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PlannerError, match="Set OPENAI_API_KEY"):
        OpenAIPlanner()


async def test_real_sdk_serializes_schema_and_parses_a_mock_http_response():
    """Exercise the installed SDK without contacting OpenAI or using a real key."""
    import httpx
    from openai import AsyncOpenAI

    memory = observed_memory()
    reference = memory.results[ToolName.LOGS].records[0].reference
    captured = []

    def handler(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        captured.append(payload)
        content = decision_for(ToolName.TRACES, reference).model_dump_json()
        return httpx.Response(
            200,
            json={
                "id": "resp_mock_http",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": DEFAULT_MODEL,
                "output": [
                    {
                        "id": "msg_mock",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "temperature": 0,
                "top_p": 1,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="test-placeholder", http_client=http_client, max_retries=0
        ) as client:
            planner = OpenAIPlanner(client=client)
            action = await planner.choose(InvestigationRequest(mode="without-rag"), memory)
    assert action.tool == ToolName.TRACES
    assert action.evidence_references == (reference,)
    assert captured[0]["store"] is False
    assert captured[0]["text"]["format"]["type"] == "json_schema"
    assert captured[0]["text"]["format"]["strict"] is True
    assert planner.response_ids == ["resp_mock_http"]
