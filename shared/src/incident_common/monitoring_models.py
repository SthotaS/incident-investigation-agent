"""Typed monitoring samples for comparing transaction failures over time.

Counts describe only the transactions in the sample. Several failed transactions
can support a wider service problem, but cannot establish a complete outage.
"""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

MonitoringTransactionId = Annotated[str, Field(pattern=r"^TX[0-9]{6}$")]
SyntheticReference = Annotated[str, Field(pattern=r"^synthetic://.+$")]


class TransactionObservation(BaseModel):
    """One sampled settlement attempt, with a source and an explicit timestamp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: MonitoringTransactionId
    observed_at: AwareDatetime
    outcome: Literal["failed", "succeeded"]
    http_status: Annotated[int, Field(ge=100, le=599, strict=True)] | None = None
    reference: SyntheticReference

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.http_status is None:
            return self
        if self.outcome == "succeeded" and not 200 <= self.http_status <= 299:
            raise ValueError("A successful HTTP observation must have a 2xx status")
        if self.outcome == "failed" and self.http_status < 400:
            raise ValueError("A failed HTTP observation must have a 4xx or 5xx status")
        return self


class MonitoringWindow(BaseModel):
    """A bounded service sample; count fields must match the sampled records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_start: AwareDatetime
    window_end: AwareDatetime
    error_started_at: AwareDatetime | None = None
    service: Annotated[str, Field(min_length=1, max_length=80)]
    observations: Annotated[list[TransactionObservation], Field(min_length=1, max_length=100)]
    checked_transaction_count: Annotated[int, Field(ge=0, strict=True)]
    affected_transaction_count: Annotated[int, Field(ge=0, strict=True)]
    successful_transaction_count: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("Monitoring window end must be after its start")
        if self.error_started_at is not None and not (
            self.window_start <= self.error_started_at <= self.window_end
        ):
            raise ValueError("The error start must be inside the monitoring window")
        seen_ids: set[str] = set()
        seen_references: set[str] = set()
        for observation in self.observations:
            if observation.transaction_id in seen_ids:
                raise ValueError("Sampled transaction IDs must be unique")
            seen_ids.add(observation.transaction_id)
            if observation.reference in seen_references:
                raise ValueError("Sampled observation references must be unique")
            seen_references.add(observation.reference)
            if not self.window_start <= observation.observed_at <= self.window_end:
                raise ValueError("Every observation must be inside the monitoring window")
            if (
                self.error_started_at is not None
                and observation.outcome == "failed"
                and observation.observed_at < self.error_started_at
            ):
                raise ValueError("A failure cannot occur before the reported error start")
        failed = sum(item.outcome == "failed" for item in self.observations)
        succeeded = len(self.observations) - failed
        if (
            self.checked_transaction_count != len(self.observations)
            or self.affected_transaction_count != failed
            or self.successful_transaction_count != succeeded
        ):
            raise ValueError("Monitoring counts must match the sampled transaction observations")
        return self


def get_monitoring_window(facts: Mapping[str, object]) -> MonitoringWindow | None:
    """Read optional typed samples; malformed provided data must never look absent."""
    if "monitoring_window" not in facts:
        return None
    return MonitoringWindow.model_validate(facts["monitoring_window"])
