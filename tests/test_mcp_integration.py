"""Exercise a subprocess over MCP stdio, without directly importing server tools."""

import sys

from incident_client.investigation import InvestigationEngine
from incident_client.mcp_client import connect
from incident_common.models import InvestigationRequest, Source, ToolName, ToolResult
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_stdio_tool_contract_and_input_validation():
    params = StdioServerParameters(command=sys.executable, args=["-m", "incident_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {tool.value for tool in ToolName}
            assert len(listed.tools) == 7
            assert all(tool.annotations.readOnlyHint for tool in listed.tools)
            found = await session.call_tool(
                "get_transaction", arguments={"transaction_id": "TX784521"}
            )
            assert not found.isError
            result = ToolResult.model_validate(found.structuredContent)
            assert result.status == "found"
            assert result.source == Source.TRANSACTIONS
            assert result.transaction_id == "TX784521"
            assert result.records
            missing = await session.call_tool(
                "get_transaction", arguments={"transaction_id": "TX999999"}
            )
            assert not missing.isError
            empty = ToolResult.model_validate(missing.structuredContent)
            assert empty.status == "empty"
            assert empty.records == []
            invalid = await session.call_tool(
                "get_transaction", arguments={"transaction_id": "invalid-id"}
            )
            assert invalid.isError


async def test_real_mcp_client_investigates_synthetic_incident():
    async with connect() as gateway:
        report = await InvestigationEngine(gateway).run(InvestigationRequest())
    assert report.status == "finished"
    assert report.calls_used == 6
    assert report.supported_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed
    assert not report.knowledge_base_saved
    assert not report.production_changes_made
    assert report.missing_sources == []
