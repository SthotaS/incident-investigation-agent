"""A real stdio MCP connection; no direct imports of server tool functions."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from incident_common.models import MAX_TOOL_RESULT_BYTES, TOOL_SOURCES, ToolName, ToolResult
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolExecutionError(Exception):
    """The tool did not complete; this is different from an empty search."""


class ToolGateway(Protocol):
    async def call_tool(
        self,
        tool: ToolName,
        transaction_id: str,
        *,
        query: str | None = None,
        current_incident_id: str | None = None,
    ) -> ToolResult: ...


class MCPGateway:
    def __init__(self, session: ClientSession, timeout_seconds: float) -> None:
        self.session = session
        self.timeout_seconds = timeout_seconds

    async def list_tools(self) -> list[str]:
        names: list[str] = []
        seen_cursors: set[str] = set()
        cursor = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                for _ in range(5):
                    response = await self.session.list_tools(cursor=cursor)
                    names.extend(tool.name for tool in response.tools)
                    if len(names) > 50 or len(names) != len(set(names)):
                        break
                    cursor = response.nextCursor
                    if cursor is None:
                        return names
                    if cursor in seen_cursors:
                        break
                    seen_cursors.add(cursor)
        except Exception:
            raise ToolExecutionError("Tool discovery failed or timed out.") from None
        raise ToolExecutionError("Tool discovery exceeded its limits or repeated a page.")

    async def call_tool(
        self,
        tool: ToolName,
        transaction_id: str,
        *,
        query: str | None = None,
        current_incident_id: str | None = None,
    ) -> ToolResult:
        arguments = {"transaction_id": transaction_id}
        if tool == ToolName.SIMILAR:
            if not query:
                raise ToolExecutionError("A similarity search needs observed symptoms.")
            arguments["query"] = query
            if current_incident_id:
                arguments["current_incident_id"] = current_incident_id
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.session.call_tool(tool.value, arguments=arguments)
            if response.isError:
                raise ToolExecutionError(f"{tool.value} could not finish.")
            payload = response.structuredContent
            if payload is None:
                blocks = [c.text for c in response.content if c.type == "text"]
                if len(blocks) != 1:
                    raise ToolExecutionError("The tool returned an unexpected result.")
                if len(blocks[0].encode("utf-8")) > MAX_TOOL_RESULT_BYTES:
                    raise ToolExecutionError("The tool response exceeded its size limit.")
                payload = json.loads(blocks[0])
            elif len(json.dumps(payload, allow_nan=False).encode("utf-8")) > MAX_TOOL_RESULT_BYTES:
                raise ToolExecutionError("The tool response exceeded its size limit.")
            result = ToolResult.model_validate(payload)
            if result.source != TOOL_SOURCES[tool] or result.transaction_id != transaction_id:
                raise ToolExecutionError("The tool result does not match the request.")
            if tool == ToolName.SIMILAR and (
                result.retrieval is None or result.retrieval.query != query
            ):
                raise ToolExecutionError("The search result does not match its query.")
            if (
                tool == ToolName.SIMILAR
                and current_incident_id
                and any(
                    match.incident_id == current_incident_id for match in result.retrieval.matches
                )
            ):
                raise ToolExecutionError(
                    "The current incident cannot be its own historical evidence."
                )
            return result
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{tool.value} failed or timed out.") from exc


@asynccontextmanager
async def connect(
    timeout_seconds: float = 10,
    *,
    rag_dir: Path | None = None,
    knowledge_db: Path | None = None,
) -> AsyncIterator[MCPGateway]:
    """Start the installed server using the same Python environment as this client."""
    # The server receives local storage paths, never the client's API key.
    env = {
        name: os.environ[name]
        for name in ("INCIDENT_RAG_DIR", "INCIDENT_KNOWLEDGE_DB")
        if name in os.environ
    }
    if rag_dir is not None:
        env["INCIDENT_RAG_DIR"] = os.fspath(rag_dir)
    if knowledge_db is not None:
        env["INCIDENT_KNOWLEDGE_DB"] = os.fspath(knowledge_db)
    params = StdioServerParameters(command=sys.executable, args=["-m", "incident_server"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=timeout_seconds)
        ) as session:
            async with asyncio.timeout(timeout_seconds):
                await session.initialize()
            yield MCPGateway(session, timeout_seconds)
