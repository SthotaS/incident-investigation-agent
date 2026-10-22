"""Human-approved historical decisions, kept separate from an agent's findings."""

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


def _safe_text(value: object) -> object:
    """Keep ordinary notes readable without accepting terminal control sequences."""
    if not isinstance(value, str):
        return value
    value = value.replace("\r\n", "\n")
    if any(
        (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) <= 159
        for character in value
    ):
        raise ValueError("Text must not contain unsafe control characters.")
    return value


IncidentIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^INC-[0-9]{5}$")
]
ShortText = Annotated[
    str,
    BeforeValidator(_safe_text),
    StringConstraints(strip_whitespace=True, min_length=1),
]
EvidenceReference = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]


class HumanDecisionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, validate_default=True
    )

    incident_id: IncidentIdentifier
    title: Annotated[ShortText, Field(max_length=200)]
    symptoms: Annotated[ShortText, Field(max_length=1500)]
    service: Annotated[ShortText, Field(max_length=60)]
    human_decision: Annotated[ShortText, Field(max_length=2000)]
    action_taken: Annotated[ShortText, Field(max_length=2000)]
    outcome: Literal["resolved", "improved", "no_change", "pending", "unknown"] = "unknown"
    approved_by: Annotated[ShortText, Field(max_length=100)]
    confirmed_root_cause: Annotated[ShortText, Field(max_length=1000)] | None = None
    occurred_at: AwareDatetime | None = None
    evidence_references: list[EvidenceReference] = Field(default_factory=list, max_length=32)
    useful_check: Literal["get_monitoring", "get_trace", "get_deployments"] = "get_trace"
    check_reason: Annotated[ShortText, Field(max_length=500)] = (
        "Compare the saved decision and action with current evidence before reusing them."
    )
    synthetic: Literal[True] = True

    @field_validator("evidence_references")
    @classmethod
    def validate_references(cls, references: list[str]) -> list[str]:
        cleaned = []
        for reference in references:
            reference = reference.strip()
            if (
                not reference.startswith("synthetic://")
                or not reference.removeprefix("synthetic://")
                or any(
                    character.isspace() or ord(character) < 32 or 127 <= ord(character) <= 159
                    for character in reference
                )
            ):
                raise ValueError("Evidence references must be nonempty synthetic:// references.")
            if reference not in cleaned:
                cleaned.append(reference)
        return cleaned


class HumanDecisionRecord(HumanDecisionInput):
    saved_at: AwareDatetime
    record_source: Literal["human_approved"] = "human_approved"


class SaveResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["saved", "already_saved"]
    record: HumanDecisionRecord
