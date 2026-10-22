"""Synthetic responses for checking investigation limits without a model."""

from collections import Counter

import pytest
from incident_client.mcp_client import ToolExecutionError
from incident_common.models import TOOL_SOURCES, Record, ToolName, ToolResult


def result_for(tool: ToolName, *, empty: bool = False, **facts: object) -> ToolResult:
    return ToolResult(
        source=TOOL_SOURCES[tool],
        transaction_id="TX784521",
        status="empty" if empty else "found",
        records=[]
        if empty
        else [
            Record(
                record_id=f"synthetic-{tool.value}",
                summary=f"Synthetic result from {tool.value}",
                observed_at="2026-10-15T10:00:00Z",
                reference=f"synthetic://{tool.value}/TX784521",
                facts=facts,
            )
        ],
    )


class FakeGateway:
    def __init__(self, results, *, fail_first=None, always_fail=None):
        self.results = results
        self.fail_first = fail_first or set()
        self.always_fail = always_fail or set()
        self.calls = []
        self.attempts = Counter()

    async def call_tool(self, tool: ToolName, transaction_id: str) -> ToolResult:
        self.calls.append(tool)
        self.attempts[tool] += 1
        if tool in self.always_fail or (tool in self.fail_first and self.attempts[tool] == 1):
            raise ToolExecutionError("Synthetic service unavailable.")
        return self.results[tool].model_copy(update={"transaction_id": transaction_id})


@pytest.fixture
def results():
    return {
        ToolName.TRANSACTION: result_for(
            ToolName.TRANSACTION, status="SETTLEMENT_SENT", age_hours=25
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
            application_healthy=True,
            elevated_errors=True,
        ),
        ToolName.DEPLOYMENTS: result_for(
            ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
        ),
        ToolName.HISTORY: result_for(ToolName.HISTORY, similar_incident="INC-10001"),
    }


@pytest.fixture(autouse=True)
def isolate_human_knowledge_database(tmp_path, monkeypatch):
    """Tests never read or change the operator's saved human decisions."""
    monkeypatch.setenv("INCIDENT_KNOWLEDGE_DB", str(tmp_path / "knowledge" / "incidents.sqlite3"))
