"""Seven read-only MCP tools. Data is synthetic; no production system is contacted."""

import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from incident_common.models import IncidentId, Source, ToolResult, TransactionId
from incident_common.rag_models import SearchQuery
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from incident_server.repository import get_repository

mcp = FastMCP(
    "Synthetic incident investigation tools",
    log_level="WARNING",
    instructions=(
        "This local learning server returns synthetic records only. It has no production "
        "connections or write tools. Empty results mean no matching fixture evidence; "
        "they do not prove an explanation false. Tool failures are reported as MCP errors."
    ),
)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


@mcp.tool(annotations=READ_ONLY)
def get_transaction(transaction_id: TransactionId) -> ToolResult:
    """Read a synthetic transaction's status and time in that status."""
    return get_repository().get(Source.TRANSACTIONS, transaction_id)


@mcp.tool(annotations=READ_ONLY)
def search_logs(transaction_id: TransactionId) -> ToolResult:
    """Read synthetic application log evidence for one transaction."""
    return get_repository().get(Source.LOGS, transaction_id)


@mcp.tool(annotations=READ_ONLY)
def get_trace(transaction_id: TransactionId) -> ToolResult:
    """Read a synthetic trace, including outcomes of the application's own retries."""
    return get_repository().get(Source.TRACES, transaction_id)


@mcp.tool(annotations=READ_ONLY)
def get_monitoring(transaction_id: TransactionId) -> ToolResult:
    """Read synthetic settlement-service errors and application health evidence."""
    return get_repository().get(Source.MONITORING, transaction_id)


@mcp.tool(annotations=READ_ONLY)
def get_deployments(transaction_id: TransactionId) -> ToolResult:
    """Read synthetic deployment-check results, including checks finding no relevant change."""
    return get_repository().get(Source.DEPLOYMENTS, transaction_id)


@mcp.tool(annotations=READ_ONLY)
def search_incidents(transaction_id: TransactionId) -> ToolResult:
    """Read similar synthetic past incidents; past causes do not confirm the current cause."""
    return get_repository().get(Source.HISTORY, transaction_id)


@mcp.tool(name="searchSimilarIncidents", annotations=READ_ONLY)
def search_similar_incidents(
    transaction_id: TransactionId,
    query: SearchQuery,
    top_k: Annotated[int, Field(ge=1, le=5)] = 2,
    current_incident_id: IncidentId | None = None,
) -> ToolResult:
    """Search the prepared FAISS index for past cases; history is a lead, not current proof."""
    from incident_server.rag import search_similar_incidents as search

    return search(transaction_id, query, top_k, current_incident_id=current_incident_id)


def main() -> None:
    """Start the stdio server; reserve stdout for MCP protocol messages."""
    try:
        project_version = version("incident-investigation-agent")
    except PackageNotFoundError:
        project_version = "0.1.0"
    parser = argparse.ArgumentParser(description="Read-only synthetic incident MCP server")
    parser.add_argument("--version", action="version", version=project_version)
    parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
