"""Short-term memory for one cycle; a new cycle needs engineer direction."""

from dataclasses import dataclass, field

from incident_common.models import AuditEntry, ToolName, ToolResult


@dataclass
class InvestigationMemory:
    results: dict[ToolName, ToolResult] = field(default_factory=dict)
    attempts: dict[ToolName, int] = field(default_factory=dict)
    evidence_keys: set[tuple[str, str]] = field(default_factory=set)
    audit: list[AuditEntry] = field(default_factory=list)
    no_new_evidence_count: int = 0

    def has_fact(self, tool: ToolName, name: str, value: object) -> bool:
        return self.has_facts(tool, **{name: value})

    def has_facts(self, tool: ToolName, **expected: object) -> bool:
        """Require related facts to come from the same record."""
        result = self.results.get(tool)
        return bool(
            result
            and any(
                all(record.facts.get(name) == value for name, value in expected.items())
                for record in result.records
            )
        )
