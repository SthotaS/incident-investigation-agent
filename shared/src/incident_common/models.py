"""The data contract between read-only tools and the investigation client."""

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    field_validator,
    model_validator,
)

from incident_common.hypothesis_models import CauseAssessment
from incident_common.rag_models import RetrievalInfo
from incident_common.tot_models import ThoughtSearch

TransactionId = Annotated[str, Field(pattern=r"^TX[0-9]{6}$")]
IncidentId = Annotated[str, Field(pattern=r"^INC-[0-9]{5}$")]
MAX_TOOL_RESULT_BYTES = 524288
MAX_FACT_BYTES = 32768
_UNSAFE_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _safe_text(value: str) -> str:
    if _UNSAFE_CONTROLS.search(value):
        raise ValueError("Text cannot contain terminal control characters.")
    return value


def _safe_json_text(value: object) -> None:
    if isinstance(value, str):
        _safe_text(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            _safe_text(key)
            _safe_json_text(item)
    elif isinstance(value, list):
        for item in value:
            _safe_json_text(item)


class Source(StrEnum):
    TRANSACTIONS = "transaction_database"
    LOGS = "application_logs"
    TRACES = "distributed_traces"
    MONITORING = "monitoring"
    DEPLOYMENTS = "deployment_history"
    HISTORY = "past_incidents"


class ToolName(StrEnum):
    TRANSACTION = "get_transaction"
    LOGS = "search_logs"
    TRACES = "get_trace"
    MONITORING = "get_monitoring"
    DEPLOYMENTS = "get_deployments"
    HISTORY = "search_incidents"
    SIMILAR = "searchSimilarIncidents"


TOOL_SOURCES = {
    ToolName.TRANSACTION: Source.TRANSACTIONS,
    ToolName.LOGS: Source.LOGS,
    ToolName.TRACES: Source.TRACES,
    ToolName.MONITORING: Source.MONITORING,
    ToolName.DEPLOYMENTS: Source.DEPLOYMENTS,
    ToolName.HISTORY: Source.HISTORY,
    ToolName.SIMILAR: Source.HISTORY,
}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    record_id: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=8192)
    observed_at: str = Field(min_length=1, max_length=64)
    reference: str = Field(min_length=13, max_length=512, pattern=r"^synthetic://.+$")
    facts: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("record_id", "summary", "observed_at", "reference")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _safe_text(value)

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("Source references cannot contain whitespace.")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        observed = datetime.fromisoformat(value)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("Source timestamps must include a time zone.")
        return value

    @field_validator("facts")
    @classmethod
    def validate_facts(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if len(json.dumps(value, allow_nan=False).encode("utf-8")) > MAX_FACT_BYTES:
            raise ValueError("A source record has too much evidence data.")
        _safe_json_text(value)
        return value


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    source: Source
    transaction_id: TransactionId
    status: Literal["found", "empty"]
    records: list[Record] = Field(default_factory=list, max_length=10)
    retrieval: RetrievalInfo | None = None

    @model_validator(mode="after")
    def validate_records(self) -> "ToolResult":
        if (self.status == "found") != bool(self.records):
            raise ValueError("A found result needs records; an empty result must have no records.")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("Source record IDs must be unique within a result.")
        if len({record.reference for record in self.records}) != len(self.records):
            raise ValueError("Source references must be unique within a result.")
        for record in self.records:
            if (
                "transaction_id" in record.facts
                and record.facts["transaction_id"] != self.transaction_id
            ):
                raise ValueError("The source record transaction does not match its result.")
            if self.source != Source.HISTORY and (
                record.facts.get("historical") is True
                or "past_incident_id" in record.facts
                or "past_root_cause" in record.facts
            ):
                raise ValueError("Historical context cannot be used as current evidence.")
        if self.retrieval is not None:
            if self.source != Source.HISTORY:
                raise ValueError("Retrieval metadata belongs only to historical search results.")
            expected = [(m.incident_id, m.reference) for m in self.retrieval.matches]
            actual = [(r.facts.get("past_incident_id"), r.reference) for r in self.records]
            if expected != actual or len({item[0] for item in expected}) != len(expected):
                raise ValueError("Retrieved matches and historical records must agree.")
            if any(r.facts.get("historical") is not True for r in self.records):
                raise ValueError("Retrieved records must be marked as historical.")
        return self


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: IncidentId = "INC-10234"
    transaction_id: TransactionId = "TX784521"
    scope: Literal["full", "status"] = "full"
    mode: Literal["legacy", "without-rag", "with-rag"] = "legacy"
    use_rag: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_rag_control(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "use_rag" in value:
            enabled = value["use_rag"]
            if not isinstance(enabled, bool):
                raise ValueError("use_rag must be True or False.")
            mode = "with-rag" if enabled else "without-rag"
            if "mode" in value and value["mode"] != mode:
                raise ValueError("mode and use_rag must agree.")
            value["mode"] = mode
        else:
            value["use_rag"] = value.get("mode") == "with-rag"
        return value


class SkippedCheck(BaseModel):
    """A policy decision, never an attempted tool call or an empty result."""

    tool: Literal[ToolName.SIMILAR] = ToolName.SIMILAR
    reason: str = "RAG disabled; knowledge database not checked."


class AuditEntry(BaseModel):
    call_number: int
    called_at: str
    tool: ToolName
    source: Source
    arguments: dict[str, str]
    reason: str
    decision_references: list[str] = Field(default_factory=list)
    status: Literal["found", "empty", "error"]
    observation: str
    record_references: list[str] = Field(default_factory=list)
    new_evidence: bool
    elapsed_ms: float


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    cached_input_tokens: int = Field(default=0, ge=0, strict=True)
    total_tokens: int = Field(default=0, ge=0, strict=True)
    measured_responses: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_totals(self) -> "ModelUsage":
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Token totals must equal input plus output tokens.")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("Cached input cannot exceed total input tokens.")
        return self


class InvestigationReport(BaseModel):
    run_id: str
    mode: str = "legacy"
    use_rag: bool = False
    skipped_checks: list[SkippedCheck] = Field(default_factory=list)
    planner: str = "demo"
    reasoning: Literal["linear", "tot"] = "linear"
    thought_searches: list[ThoughtSearch] = Field(default_factory=list, max_length=3)
    model: str | None = None
    model_calls: int = 0
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    model_response_ids: list[str] = Field(default_factory=list)
    planner_error: str | None = None
    retrieval: RetrievalInfo | None = None
    historical_context: list[Record] = Field(default_factory=list)
    incident_id: IncidentId
    transaction_id: TransactionId
    status: Literal["finished", "paused"]
    stop_reason: str
    calls_used: int
    call_limit: int = 8
    supported_failure_point: str | None = None
    possible_failure_point: str | None = None
    likely_cause: str | None = None
    open_possible_causes: list[str] = Field(default_factory=list)
    cause_assessments: list[CauseAssessment] = Field(default_factory=list)
    unsupported_explanations: list[str] = Field(default_factory=list)
    missing_sources: list[Source] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)
    root_cause_confirmed: Literal[False] = False
    knowledge_base_saved: Literal[False] = False
    production_changes_made: Literal[False] = False
