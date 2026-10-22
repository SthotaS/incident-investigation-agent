"""Check API time, request and input budgets without making network requests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import FakeGateway, result_for
from incident_client.config import Settings
from incident_client.investigation import InvestigationEngine
from incident_client.llm import OpenAIPlanner, PlannerError
from incident_common.models import InvestigationRequest, ModelUsage, ToolName
from pydantic import ValidationError
from test_llm_planner import decision_for, fake_client, observed_memory


def usage(input_tokens=100, output_tokens=20, *, cached=40, total_tokens=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens if total_tokens is None else total_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def response(memory, measured_usage, *, decision=True, status="completed", refusal=False):
    reference = memory.results[ToolName.LOGS].records[0].reference
    return SimpleNamespace(
        id="resp_runtime_synthetic",
        status=status,
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])] if refusal else [],
        output_parsed=decision_for(ToolName.TRACES, reference) if decision else None,
        usage=measured_usage,
    )


async def test_real_token_counts_are_added_across_measured_responses():
    memory = observed_memory()
    client = fake_client()
    client.responses.parse.side_effect = [
        response(memory, usage()),
        response(memory, usage(200, 30, cached=50)),
    ]
    planner = OpenAIPlanner(client=client)
    request = InvestigationRequest(mode="without-rag")

    await planner.choose(request, memory)
    await planner.choose(request, memory)

    assert planner.calls == 2
    assert planner.usage == ModelUsage(
        input_tokens=300,
        output_tokens=50,
        total_tokens=350,
        cached_input_tokens=90,
        measured_responses=2,
    )
    assert not any("cost" in field or "dollar" in field for field in ModelUsage.model_fields)


@pytest.mark.parametrize(
    "measured_usage",
    [
        None,
        SimpleNamespace(input_tokens=100),
        usage(-1, 20, cached=0),
        usage(100, -1),
        usage(100, 20, total_tokens=999),
        usage(100, 20, cached=101),
        usage(100, 20, cached=-1),
        usage(100, 20, cached=False),
        usage(100, 20, cached=""),
        usage(100, 20, cached=[]),
        SimpleNamespace(input_tokens=True, output_tokens=20, total_tokens=21),
        SimpleNamespace(input_tokens="100", output_tokens=20, total_tokens=120),
    ],
)
async def test_missing_or_invalid_usage_is_unmeasured_instead_of_estimated(measured_usage):
    memory = observed_memory()
    client = fake_client()
    client.responses.parse.return_value = response(memory, measured_usage)
    planner = OpenAIPlanner(client=client)

    await planner.choose(InvestigationRequest(mode="without-rag"), memory)

    assert planner.calls == 1
    assert planner.usage.measured_responses == 0
    assert planner.usage.total_tokens == 0


@pytest.mark.parametrize(
    "options", [{"decision": False}, {"status": "incomplete"}, {"refusal": True}]
)
async def test_received_usage_is_kept_even_when_the_decision_cannot_be_used(options):
    memory = observed_memory()
    client = fake_client()
    client.responses.parse.return_value = response(memory, usage(), **options)
    planner = OpenAIPlanner(client=client)

    with pytest.raises(PlannerError):
        await planner.choose(InvestigationRequest(mode="without-rag"), memory)

    assert planner.calls == 1
    assert planner.usage.total_tokens == 120
    assert planner.usage.measured_responses == 1


async def test_later_missing_usage_does_not_erase_an_earlier_measurement():
    memory = observed_memory()
    client = fake_client()
    client.responses.parse.side_effect = [response(memory, usage()), response(memory, None)]
    planner = OpenAIPlanner(client=client)
    request = InvestigationRequest(mode="without-rag")

    await planner.choose(request, memory)
    await planner.choose(request, memory)

    assert planner.calls == 2
    assert planner.usage.total_tokens == 120
    assert planner.usage.measured_responses == 1


async def test_request_limit_blocks_another_api_request_without_incrementing_count():
    memory = observed_memory()
    client = fake_client()
    client.responses.parse.return_value = response(memory, usage())
    planner = OpenAIPlanner(client=client, settings=Settings(max_api_calls=1))
    request = InvestigationRequest(mode="without-rag")
    await planner.choose(request, memory)

    with pytest.raises(PlannerError, match="API request limit"):
        await planner.choose(request, memory)

    client.responses.parse.assert_awaited_once()
    assert planner.calls == 1
    assert planner.usage.measured_responses == 1


async def test_oversized_input_is_blocked_before_any_api_request():
    memory = observed_memory()
    record = memory.results[ToolName.LOGS].records[0]
    memory.results[ToolName.LOGS].records[0] = record.model_copy(
        update={"facts": {**record.facts, "synthetic_long_note": "evidence " * 2000}}
    )
    client = fake_client()
    planner = OpenAIPlanner(client=client, settings=Settings(max_api_input_bytes=4096))

    with pytest.raises(PlannerError, match="API input limit"):
        await planner.choose(InvestigationRequest(mode="without-rag"), memory)

    client.responses.parse.assert_not_called()
    assert planner.calls == 0
    assert planner.usage.measured_responses == 0


async def test_timeout_also_limits_injected_clients_without_retries_or_private_output():
    private_text = "synthetic-provider-secret"
    cancelled = asyncio.Event()

    async def delayed_response(**kwargs):
        try:
            await asyncio.sleep(60)
            raise RuntimeError(private_text)
        finally:
            cancelled.set()

    client = fake_client()
    client.responses.parse = AsyncMock(side_effect=delayed_response)
    planner = OpenAIPlanner(client=client, settings=Settings(api_timeout_seconds=0.005))

    with pytest.raises(PlannerError, match="timeout") as failure:
        await planner.choose(InvestigationRequest(mode="without-rag"), observed_memory())

    assert cancelled.is_set()
    assert private_text not in str(failure.value)
    client.responses.parse.assert_awaited_once()
    assert planner.calls == 1
    assert planner.usage.measured_responses == 0


async def test_failed_provider_request_is_counted_without_inventing_usage():
    client = fake_client(error=RuntimeError("synthetic-provider-secret"))
    planner = OpenAIPlanner(client=client)

    with pytest.raises(PlannerError) as failure:
        await planner.choose(InvestigationRequest(mode="without-rag"), observed_memory())

    assert "synthetic-provider-secret" not in str(failure.value)
    assert planner.calls == 1
    assert planner.usage.measured_responses == 0


async def test_owned_client_uses_configured_deadline_and_disables_sdk_retries(monkeypatch):
    constructed = {}
    client = fake_client()

    def build_client(**kwargs):
        constructed.update(kwargs)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-never-sent")
    monkeypatch.setattr("incident_client.llm.AsyncOpenAI", build_client)
    planner = OpenAIPlanner(settings=Settings(api_timeout_seconds=2.5))

    assert constructed["timeout"] == 2.5
    assert constructed["max_retries"] == 0
    assert planner.calls == 0
    await planner.aclose()
    client.close.assert_awaited_once()


def test_client_construction_failure_hides_private_configuration(monkeypatch):
    def fail_client(**_kwargs):
        raise RuntimeError("synthetic-private-configuration")

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-never-sent")
    monkeypatch.setattr("incident_client.llm.AsyncOpenAI", fail_client)
    with pytest.raises(PlannerError) as error:
        OpenAIPlanner()
    assert "could not start" in str(error.value)
    assert "synthetic-private-configuration" not in str(error.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"api_timeout_seconds": 0},
        {"api_timeout_seconds": float("inf")},
        {"api_timeout_seconds": float("nan")},
        {"max_api_calls": 0},
        {"max_api_calls": 9},
        {"max_api_input_bytes": 4095},
        {"max_api_input_bytes": 524289},
    ],
)
def test_invalid_runtime_limits_are_rejected(updates):
    with pytest.raises(ValidationError):
        Settings(**updates)


def test_runtime_limits_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("MAX_API_CALLS", "3")
    monkeypatch.setenv("MAX_API_INPUT_BYTES", "8192")
    settings = Settings.from_environment()
    assert settings.api_timeout_seconds == 4
    assert settings.max_api_calls == 3
    assert settings.max_api_input_bytes == 8192


def investigation_results():
    return {
        ToolName.TRANSACTION: result_for(
            ToolName.TRANSACTION, status="SETTLEMENT_SENT", age_hours=26
        ),
        ToolName.LOGS: result_for(ToolName.LOGS, service="settlement", http_status=503),
        ToolName.TRACES: result_for(
            ToolName.TRACES,
            service="settlement",
            successful_response=False,
            retries_exhausted=True,
        ),
        ToolName.MONITORING: result_for(
            ToolName.MONITORING,
            service="settlement",
            elevated_errors=True,
            application_healthy=True,
        ),
        ToolName.DEPLOYMENTS: result_for(
            ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
        ),
    }


@pytest.mark.parametrize("invalid_decision", [False, True])
async def test_final_report_keeps_actual_usage_on_finished_and_paused_runs(invalid_decision):
    results = investigation_results()
    memory = observed_memory()
    client = fake_client()
    first = response(memory, usage(), decision=not invalid_decision)
    second = response(memory, usage(200, 30, cached=50))
    second.output_parsed = decision_for(
        ToolName.MONITORING, memory.results[ToolName.LOGS].records[0].reference
    )
    client.responses.parse.side_effect = [first, second]
    planner = OpenAIPlanner(client=client)

    report = await InvestigationEngine(FakeGateway(results), planner=planner).run(
        InvestigationRequest(mode="without-rag")
    )

    assert report.status == ("paused" if invalid_decision else "finished")
    assert report.model_usage == planner.usage
    assert report.model_usage.total_tokens == (120 if invalid_decision else 350)
    assert report.model_usage.measured_responses == report.model_calls
    assert not report.root_cause_confirmed
    assert not report.knowledge_base_saved
    assert not report.production_changes_made
