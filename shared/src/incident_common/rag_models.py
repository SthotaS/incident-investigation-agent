"""Retrieval details are separate from incident facts and current evidence."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SearchQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)
]


class RetrievalMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(pattern=r"^INC-[0-9]{5}$")
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    reference: str

    @model_validator(mode="after")
    def validate_reference(self) -> "RetrievalMatch":
        if self.reference != f"synthetic://incidents/{self.incident_id}":
            raise ValueError("The reference must match the synthetic incident ID.")
        return self


class RetrievalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: SearchQuery
    embedding_model: str = Field(min_length=1, max_length=200)
    index_version: str = Field(min_length=1, max_length=30)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    matches: list[RetrievalMatch] = Field(default_factory=list, max_length=5)
