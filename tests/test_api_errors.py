"""Show safe API error categories without paid calls, secrets, or hidden retries."""

import json

import httpx
import pytest
from conftest import FakeGateway, result_for
from incident_client.api_errors import describe_api_error
from incident_client.investigation import InvestigationEngine
from incident_client.llm import DEFAULT_MODEL, Decision, OpenAIPlanner, PlannerError
from incident_client.memory import InvestigationMemory
from incident_common.models import InvestigationRequest, ToolName
from openai import APIResponseValidationError, APIStatusError, AsyncOpenAI, RateLimitError
from pydantic import ValidationError

PRIVATE_TEXT = "sensitive provider body sk-private-value secret-client-path"


def memory_for_api():
    memory = InvestigationMemory()
    memory.results[ToolName.TRANSACTION] = result_for(
        ToolName.TRANSACTION, status="SETTLEMENT_SENT", age_hours=26
    )
    memory.results[ToolName.LOGS] = result_for(ToolName.LOGS, service="settlement", http_status=503)
    return memory


def assert_private_data_hidden(message):
    for value in (PRIVATE_TEXT, "sk-private-value", "secret-client-path", "private-request-id"):
        assert value not in message


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "invalid_api_key", ("401", "key")),
        (403, "permission_denied", ("403", "access")),
        (404, "model_not_found", ("404", "model")),
        (400, "invalid_request_error", ("400", "request")),
        (400, "unsupported_model", ("400", "model")),
        (429, "insufficient_quota", ("429", "quota")),
        (429, "rate_limit_exceeded", ("429", "rate")),
        (429, "code-injection-secret", ("429",)),
        (500, "server_error", ("500", "service")),
        (503, "server_error", ("503", "service")),
        (418, "code-injection-secret", ("418",)),
    ],
)
async def test_real_sdk_status_errors_are_specific_safe_and_not_retried(status, code, expected):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"x-request-id": "private-request-id"},
            json={
                "error": {
                    "message": PRIVATE_TEXT,
                    "type": "secret-client-path",
                    "param": "secret-client-path",
                    "code": code,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="sk-private-value", http_client=http_client, max_retries=0
        ) as client:
            planner = OpenAIPlanner(client=client)
            with pytest.raises(PlannerError) as failure:
                await planner.choose(InvestigationRequest(mode="without-rag"), memory_for_api())

    message = str(failure.value)
    assert all(word in message.lower() for word in expected)
    assert_private_data_hidden(message)
    assert "code-injection-secret" not in message
    assert planner.error == message
    assert planner.calls == len(requests) == 1


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [(httpx.ReadTimeout, ("timeout", "timed out")), (httpx.ConnectError, ("connect",))],
)
async def test_real_sdk_transport_errors_keep_timeout_and_connection_distinct(error_type, expected):
    requests = []

    def handler(request):
        requests.append(request)
        raise error_type(PRIVATE_TEXT, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="sk-private-value", http_client=http_client, max_retries=0
        ) as client:
            planner = OpenAIPlanner(client=client)
            with pytest.raises(PlannerError) as failure:
                await planner.choose(InvestigationRequest(mode="without-rag"), memory_for_api())

    assert any(word in str(failure.value).lower() for word in expected)
    assert_private_data_hidden(str(failure.value))
    assert planner.calls == len(requests) == 1


def test_quota_code_can_come_from_exception_property_without_a_body():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = RateLimitError(PRIVATE_TEXT, response=httpx.Response(429, request=request), body=None)
    error.code = "insufficient_quota"
    message = describe_api_error(error)
    assert "429" in message and "quota" in message.lower()
    assert_private_data_hidden(message)


@pytest.mark.parametrize("kind", ["pydantic", "sdk", "json"])
def test_invalid_structured_output_is_reported_without_its_contents(kind):
    if kind == "pydantic":
        try:
            Decision.model_validate({"tool": PRIVATE_TEXT})
        except ValidationError as failure:
            error = failure
    elif kind == "sdk":
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        error = APIResponseValidationError(
            httpx.Response(200, request=request), {"output": PRIVATE_TEXT}, message=PRIVATE_TEXT
        )
    else:
        try:
            json.loads(PRIVATE_TEXT)
        except json.JSONDecodeError as failure:
            error = failure
    message = describe_api_error(error)
    assert "decision" in message.lower() or "output" in message.lower()
    assert_private_data_hidden(message)


def test_unknown_local_failure_does_not_print_exception_name_or_text():
    class SecretProviderError(Exception):
        pass

    message = describe_api_error(SecretProviderError(PRIVATE_TEXT))
    assert message and "SecretProviderError" not in message
    assert_private_data_hidden(message)


@pytest.mark.parametrize("text", [PRIVATE_TEXT, json.dumps({"tool": PRIVATE_TEXT})])
async def test_real_sdk_invalid_output_is_a_safe_decision_error(text):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "resp_invalid_mock",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": DEFAULT_MODEL,
                "output": [
                    {
                        "id": "msg_invalid_mock",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="sk-private-value", http_client=http_client, max_retries=0
        ) as client:
            planner = OpenAIPlanner(client=client)
            with pytest.raises(PlannerError) as failure:
                await planner.choose(InvestigationRequest(mode="without-rag"), memory_for_api())

    message = str(failure.value)
    assert "decision" in message.lower() or "output" in message.lower()
    assert_private_data_hidden(message)
    assert planner.calls == 1


def test_generic_api_status_keeps_only_the_safe_status_number():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = APIStatusError(
        PRIVATE_TEXT,
        response=httpx.Response(409, request=request),
        body={"code": PRIVATE_TEXT, "message": PRIVATE_TEXT},
    )
    message = describe_api_error(error)
    assert "409" in message
    assert_private_data_hidden(message)


async def test_first_model_failure_pauses_after_three_rag_tools_and_does_not_guess(results):
    class RagGateway(FakeGateway):
        async def call_tool(self, tool, transaction_id, **kwargs):
            return await super().call_tool(tool, transaction_id)

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            401, json={"error": {"code": "invalid_api_key", "message": PRIVATE_TEXT}}
        )

    results[ToolName.SIMILAR] = result_for(
        ToolName.SIMILAR,
        incident_id="INC-09876",
        symptoms="Settlement HTTP 503 errors",
        suggested_next_check="get_monitoring",
    )
    gateway = RagGateway(results)
    progress = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="sk-private-value", http_client=http_client, max_retries=0
        ) as client:
            planner = OpenAIPlanner(client=client)
            report = await InvestigationEngine(gateway, planner=planner).run(
                InvestigationRequest(mode="with-rag"), on_progress=progress.append
            )

    assert report.status == "paused"
    assert report.stop_reason == "planner_error"
    assert report.calls_used == len(progress) == 3
    assert report.call_limit == 8
    assert report.model == DEFAULT_MODEL
    assert report.model_calls == len(requests) == 1
    assert gateway.calls == [ToolName.TRANSACTION, ToolName.LOGS, ToolName.SIMILAR]
    assert all(entry.status == "found" for entry in report.audit)
    assert report.planner_error and "401" in report.planner_error
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed
    assert not report.knowledge_base_saved
    assert not report.production_changes_made
    assert any("whether to continue" in step for step in report.next_steps)
    assert_private_data_hidden(report.model_dump_json())
