"""Public, bounded ToT plans and search records; no imagined tool observations."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CauseId = Literal["broad_service_issue", "transaction_specific_issue", "recent_deployment"]
CheckName = Literal["get_trace", "get_monitoring", "get_deployments"]
PublicText = Annotated[str, Field(min_length=1, max_length=400)]
Reference = Annotated[str, Field(min_length=1, max_length=512)]


class ThoughtProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cause_id: CauseId
    summary: PublicText
    evidence_references: list[Reference] = Field(min_length=1, max_length=12)
    candidate_tools: list[CheckName] = Field(min_length=1, max_length=3)

    @field_validator("summary")
    @classmethod
    def safe_summary(cls, value: str) -> str:
        if not value.strip() or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", value):
            raise ValueError("A thought needs a safe, plain-text action summary.")
        return value.strip()

    @model_validator(mode="after")
    def distinct_tools(self) -> "ThoughtProposal":
        if len(set(self.candidate_tools)) != len(self.candidate_tools):
            raise ValueError("A thought cannot propose the same tool twice.")
        return self


class ThoughtBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposals: list[ThoughtProposal] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def distinct_causes(self) -> "ThoughtBatch":
        if len({item.cause_id for item in self.proposals}) != len(self.proposals):
            raise ValueError("Candidate thoughts must cover distinct possible causes.")
        return self


class ToTLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    branching_factor: Literal[3] = 3
    beam_width: int = Field(default=2, ge=1, le=2, strict=True)
    max_depth: int = Field(default=3, ge=2, le=3, strict=True)
    max_nodes: int = Field(default=16, ge=1, le=16, strict=True)
    max_searches: int = Field(default=3, ge=1, le=3, strict=True)
    minimum_score: int = Field(default=10, ge=0, le=100, strict=True)


class ThoughtNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    parent_id: str | None
    cause_id: CauseId | None
    depth: int = Field(ge=0, le=3)
    planned_tools: list[CheckName] = Field(default_factory=list, max_length=2)
    evidence_references: list[str] = Field(default_factory=list, max_length=12)
    score: int = Field(ge=0, le=100)
    score_parts: dict[str, int] = Field(default_factory=dict)
    status: Literal["root", "candidate", "kept", "pruned", "selected"]
    reason: str


class ThoughtSearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_number: int = Field(ge=1, le=3)
    observed_after_call: int = Field(ge=0, le=8)
    strategy: Literal["beam"] = "beam"
    limits: ToTLimits
    nodes: list[ThoughtNode] = Field(max_length=16)
    selected_node_id: str | None = None
    selected_tool: CheckName | None = None
    stop_reason: Literal["selected", "no_valid_path", "node_limit"]


class ToTProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search: ThoughtSearch
