"""Bounded evidence verification outcomes; acceptance never confirms root cause."""

from typing import Literal

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    code: str
    detail: str
    references: list[str] = Field(default_factory=list)
    severity: Literal["revision", "escalation"]


class ReviewPass(BaseModel):
    decision: Literal["accept", "revise", "escalate"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    checked_references: list[str] = Field(default_factory=list)


class EvidenceReview(BaseModel):
    method: Literal["independent_rules_on_raw_snapshot"] = "independent_rules_on_raw_snapshot"
    passes: list[ReviewPass] = Field(default_factory=list)
    revisions_used: int = Field(default=0, ge=0, le=1)
    tool_calls: Literal[0] = 0
    max_revisions: Literal[1] = 1
