"""Evidence summaries for possible causes; these never confirm a root cause."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: str
    observation: str


class CauseAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cause_id: Literal[
        "broad_service_issue",
        "transaction_specific_issue",
        "recent_deployment",
        "connection_pool_issue",
    ]
    explanation: str
    status: Literal["possible", "less_supported", "needs_evidence"]
    priority: Literal["high", "medium", "low"]
    supporting_evidence: list[EvidenceNote] = Field(default_factory=list)
    opposing_evidence: list[EvidenceNote] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    next_check: str
    next_tool: Literal["get_monitoring", "get_trace", "get_deployments"] | None = None
