"""Reject unsafe, unrelated, or oversized evidence before it becomes a finding."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from conftest import FakeGateway, result_for
from incident_client import mcp_client
from incident_client.investigation import InvestigationEngine
from incident_client.mcp_client import MCPGateway, ToolExecutionError
from incident_common.models import (
    MAX_FACT_BYTES,
    MAX_TOOL_RESULT_BYTES,
    InvestigationRequest,
    Record,
    Source,
    ToolName,
    ToolResult,
)
from pydantic import ValidationError


def record_data(**changes):
    return {
        "record_id": "synthetic-log",
        "summary": "A synthetic settlement request returned HTTP 503.",
        "observed_at": "2026-10-22T10:00:00Z",
        "reference": "synthetic://logs/TX784521/503",
        "facts": {"transaction_id": "TX784521", "service": "settlement", "http_status": 503},
        **changes,
    }


@pytest.mark.parametrize(
    "timestamp",
    ["not-a-date", "2026-10-22T10:00:00", "2026-99-22T10:00:00Z", "2026-10-22"],
)
def test_source_timestamp_must_be_valid_and_include_a_timezone(timestamp):
    with pytest.raises(ValidationError):
        Record(**record_data(observed_at=timestamp))


def test_valid_source_timestamp_keeps_its_original_text():
    timestamp = "2026-10-22T05:00:00-05:00"
    record = Record(**record_data(observed_at=timestamp))
    assert record.observed_at == timestamp


@pytest.mark.parametrize(
    "changes",
    [
        {"record_id": "x" * 201},
        {"summary": "x" * 8193},
        {"reference": "synthetic://" + "x" * 501},
        {"reference": "synthetic://logs/contains a space"},
        {"reference": "https://example.com/untrusted-reference"},
        {"summary": ""},
    ],
)
def test_record_text_and_references_are_bounded(changes):
    with pytest.raises(ValidationError):
        Record(**record_data(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"summary": "Text with \x1b[31mterminal instructions"},
        {"facts": {"nested": ["Text with \x1b[31mterminal instructions"]}},
        {"facts": {"bad\x00key": "A value"}},
        {"reference": "synthetic://logs/hidden\x7fcharacter"},
    ],
)
def test_terminal_control_characters_cannot_enter_evidence(changes):
    with pytest.raises(ValidationError):
        Record(**record_data(**changes))


def test_facts_limit_counts_encoded_bytes_instead_of_only_characters():
    facts = {"details": "é" * (MAX_FACT_BYTES // 2)}
    assert len(facts["details"]) < MAX_FACT_BYTES
    with pytest.raises(ValidationError):
        Record(**record_data(facts=facts))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_not_valid_evidence(value):
    with pytest.raises(ValidationError):
        Record(**record_data(facts={"value": value}))


def test_tool_result_record_count_is_bounded():
    records = [
        Record(**record_data(record_id=f"record-{index}", reference=f"synthetic://logs/{index}"))
        for index in range(11)
    ]
    data = {
        "source": Source.LOGS,
        "transaction_id": "TX784521",
        "status": "found",
        "records": records,
    }
    assert len(ToolResult(**{**data, "records": records[:10]}).records) == 10
    with pytest.raises(ValidationError):
        ToolResult(**data)


@pytest.mark.parametrize("reused_field", ["record_id", "reference"])
def test_duplicate_records_cannot_appear_as_separate_evidence(reused_field):
    first = Record(**record_data())
    second_data = record_data(record_id="second-record", reference="synthetic://logs/second")
    second_data[reused_field] = getattr(first, reused_field)
    with pytest.raises(ValidationError):
        ToolResult(
            source=Source.LOGS,
            transaction_id="TX784521",
            status="found",
            records=[first, Record(**second_data)],
        )


def test_inner_transaction_cannot_disagree_with_requested_transaction():
    record = Record(**record_data(facts={"transaction_id": "TX000001"}))
    with pytest.raises(ValidationError):
        ToolResult(
            source=Source.LOGS,
            transaction_id="TX784521",
            status="found",
            records=[record],
        )


@pytest.mark.parametrize(
    "historical_marker",
    [
        {"historical": True},
        {"past_incident_id": "INC-09876"},
        {"past_root_cause": "A past settlement-service issue"},
    ],
)
def test_past_case_markers_cannot_be_used_as_current_logs(historical_marker):
    record = Record(**record_data(facts={**record_data()["facts"], **historical_marker}))
    with pytest.raises(ValidationError):
        ToolResult(
            source=Source.LOGS,
            transaction_id="TX784521",
            status="found",
            records=[record],
        )


@pytest.mark.parametrize("forgery", ["timestamp", "past_case", "transaction"])
async def test_forged_model_copy_cannot_supply_a_supported_current_failure(results, forgery):
    trace = results[ToolName.TRACES]
    original = trace.records[0]
    changes = {
        "timestamp": {"observed_at": "not-a-date"},
        "past_case": {"facts": {**original.facts, "historical": True}},
        "transaction": {"facts": {**original.facts, "transaction_id": "TX000001"}},
    }[forgery]
    # model_copy intentionally bypasses validation, simulating an older or faulty gateway.
    forged = original.model_copy(update=changes)
    results[ToolName.TRACES] = trace.model_copy(update={"records": [forged]})
    report = await InvestigationEngine(FakeGateway(results)).run(InvestigationRequest())
    assert report.status == "paused"
    assert report.stop_reason == "no_new_evidence"
    assert report.calls_used == 4
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"
    trace_attempts = [entry for entry in report.audit if entry.tool == ToolName.TRACES]
    assert len(trace_attempts) == 2
    assert all(entry.status == "error" and not entry.record_references for entry in trace_attempts)


async def test_oversized_text_result_is_rejected_before_json_parsing(monkeypatch):
    text = json.dumps({"padding": "x" * MAX_TOOL_RESULT_BYTES})
    response = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[SimpleNamespace(type="text", text=text)],
    )
    session = SimpleNamespace(call_tool=AsyncMock(return_value=response))
    parse = Mock(side_effect=ValueError("Parsing should not start"))
    monkeypatch.setattr(mcp_client.json, "loads", parse)
    with pytest.raises(ToolExecutionError):
        await MCPGateway(session, timeout_seconds=1).call_tool(ToolName.LOGS, "TX784521")
    assert parse.call_count == 0


async def test_oversized_structured_result_is_rejected_before_model_validation(monkeypatch):
    response = SimpleNamespace(
        isError=False,
        structuredContent={"padding": "x" * MAX_TOOL_RESULT_BYTES},
        content=[],
    )
    session = SimpleNamespace(call_tool=AsyncMock(return_value=response))
    validate = Mock(side_effect=ValueError("Validation should not start"))
    monkeypatch.setattr(mcp_client.ToolResult, "model_validate", validate)
    with pytest.raises(ToolExecutionError):
        await MCPGateway(session, timeout_seconds=1).call_tool(ToolName.LOGS, "TX784521")
    assert validate.call_count == 0


async def test_small_valid_text_result_still_passes_the_mcp_boundary():
    result = result_for(ToolName.LOGS, service="settlement", http_status=503)
    response = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[SimpleNamespace(type="text", text=result.model_dump_json())],
    )
    session = SimpleNamespace(call_tool=AsyncMock(return_value=response))
    observed = await MCPGateway(session, timeout_seconds=1).call_tool(ToolName.LOGS, "TX784521")
    assert observed == result
