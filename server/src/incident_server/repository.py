"""Load a fixed package resource; caller input never becomes a file path."""

from datetime import datetime
from functools import lru_cache
from importlib import resources
from typing import Literal

from incident_common.models import Source, ToolResult, TransactionId
from incident_common.monitoring_models import get_monitoring_window
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

_TRANSACTION_ID = TypeAdapter(TransactionId)


class FixtureData(BaseModel):
    """Validate the file and source records before making any data available."""

    model_config = ConfigDict(extra="forbid")

    synthetic: Literal[True]
    description: str
    results: list[ToolResult]

    @model_validator(mode="after")
    def validate_records(self) -> "FixtureData":
        keys: set[tuple[Source, str]] = set()
        record_ids: set[str] = set()
        for result in self.results:
            key = (result.source, result.transaction_id)
            if key in keys:
                raise ValueError("Duplicate transaction and source in synthetic fixtures")
            keys.add(key)
            for record in result.records:
                if record.record_id in record_ids:
                    raise ValueError("Synthetic record IDs must be unique")
                record_ids.add(record.record_id)
                if not record.reference.startswith("synthetic://"):
                    raise ValueError("Fixture references must use synthetic://")
                observed = datetime.fromisoformat(record.observed_at)
                if observed.tzinfo is None:
                    raise ValueError("Record timestamps must include a time zone")
                if record.facts.get("transaction_id") != result.transaction_id:
                    raise ValueError("Record transaction ID must match its result")
                if "monitoring_window" in record.facts:
                    if result.source != Source.MONITORING:
                        raise ValueError("Monitoring samples belong only to the monitoring source")
                    window = get_monitoring_window(record.facts)
                    if window is not None and window.service != record.facts.get("service"):
                        raise ValueError("Monitoring window service must match its source record")
                    if window is not None and window.window_end > observed:
                        raise ValueError(
                            "Monitoring window cannot extend past its source timestamp"
                        )
        return self


class FixtureRepository:
    """Read-only, in-memory view of validated synthetic source records."""

    def __init__(self, data: FixtureData) -> None:
        self._results = {
            (result.source, result.transaction_id): result.model_copy(deep=True)
            for result in data.results
        }

    @classmethod
    def load(cls) -> "FixtureRepository":
        content = (
            resources.files("incident_server")
            .joinpath("fixtures", "incidents.json")
            .read_text(encoding="utf-8")
        )
        return cls(FixtureData.model_validate_json(content))

    def get(self, source: Source, transaction_id: TransactionId) -> ToolResult:
        """Return found records or an empty result; load/validation errors still raise."""
        checked_id = _TRANSACTION_ID.validate_python(transaction_id)
        result = self._results.get((source, checked_id))
        if result is None:
            return ToolResult(source=source, transaction_id=checked_id, status="empty")
        return result.model_copy(deep=True)


@lru_cache(maxsize=1)
def get_repository() -> FixtureRepository:
    return FixtureRepository.load()
