"""Evidence-grounded beam search over short investigation plans.

LangChain LCEL routes generation -> evaluation/search -> selection. Only the
existing engine executes tools. Future checks stay plans until observations arrive.
"""

import inspect
from dataclasses import dataclass

from incident_common.models import InvestigationRequest, ModelUsage, Source, ToolName
from incident_common.tot_models import (
    ThoughtBatch,
    ThoughtNode,
    ThoughtProposal,
    ThoughtSearch,
    ToTLimits,
    ToTProgress,
)

from incident_client.hypotheses import assess_causes
from incident_client.llm import PlannerError
from incident_client.memory import InvestigationMemory
from incident_client.policy import Action, available_tools, prioritize_checks

CAUSES = {
    "broad_service_issue": ("Shared service issue", ToolName.MONITORING),
    "transaction_specific_issue": ("Transaction-specific issue", ToolName.TRACES),
    "recent_deployment": ("Recent change", ToolName.DEPLOYMENTS),
}


@dataclass
class SearchContext:
    request: InvestigationRequest
    memory: InvestigationMemory
    allowed: list[ToolName]
    remaining_calls: int
    batch: ThoughtBatch | None = None
    search: ThoughtSearch | None = None


class ToTPlanner:
    reasoning = "tot"

    def __init__(self, backend, *, limits: ToTLimits | None = None) -> None:
        from langchain_core.runnables import RunnableLambda

        self.backend = backend
        self.limits = limits or ToTLimits()
        self.searches: list[ThoughtSearch] = []
        self._events: list[ToTProgress] = []
        self._memory: InvestigationMemory | None = None
        self._pipeline = (
            RunnableLambda(self._generate)
            | RunnableLambda(self._search)
            | RunnableLambda(self._select)
        )

    @property
    def name(self):
        return self.backend.name

    @property
    def model(self):
        return getattr(self.backend, "model", None)

    @property
    def calls(self):
        return getattr(self.backend, "calls", 0)

    @property
    def usage(self):
        return getattr(self.backend, "usage", ModelUsage())

    @property
    def response_ids(self):
        return getattr(self.backend, "response_ids", [])

    async def aclose(self):
        close = getattr(self.backend, "aclose", None)
        if close is not None:
            await close()

    def pop_events(self) -> list[ToTProgress]:
        events, self._events = self._events, []
        return events

    async def choose(self, request: InvestigationRequest, memory: InvestigationMemory):
        if memory is not self._memory:
            self._memory = memory
            self.searches = []
            self._events = []
        allowed = available_tools(request, memory)
        if len(allowed) <= 1:
            # Setup and a sole remaining check need no tree or extra model request.
            action = self.backend.choose(request, memory)
            return await action if inspect.isawaitable(action) else action
        if len(self.searches) >= self.limits.max_searches:
            raise PlannerError("ToT search limit reached. Ask the engineer whether to continue.")
        remaining = memory.call_limit - len(memory.audit)
        if remaining <= 0:
            raise PlannerError("No tool calls remain for a ToT investigation step.")
        return await self._pipeline.ainvoke(SearchContext(request, memory, allowed, remaining))

    async def _generate(self, context: SearchContext) -> SearchContext:
        if getattr(self.backend, "name", None) == "demo":
            references = [
                record.reference
                for result in context.memory.results.values()
                for record in result.records
            ][:12]
            proposals = [
                ThoughtProposal(
                    cause_id=cause_id,
                    summary=f"Check whether {label.lower()} explains the observed failures.",
                    evidence_references=references,
                    candidate_tools=[tool.value for tool in context.allowed],
                )
                for cause_id, (label, _) in CAUSES.items()
            ]
            context.batch = ThoughtBatch(proposals=proposals)
        else:
            generate = getattr(self.backend, "generate_thoughts", None)
            if generate is None:
                raise PlannerError("This planner does not support ToT thought generation.")
            context.batch = await generate(context.request, context.memory)
        # Catch forged model_copy results and mutable nested values at the role boundary.
        try:
            context.batch = ThoughtBatch.model_validate(context.batch.model_dump())
        except (AttributeError, TypeError, ValueError):
            raise PlannerError("ToT received invalid candidate thoughts.") from None
        return context

    def _search(self, context: SearchContext) -> SearchContext:
        memory = context.memory
        number = len(self.searches) + 1
        nodes: list[ThoughtNode] = []
        proposals = {proposal.cause_id: proposal for proposal in context.batch.proposals}
        known = {
            record.reference for result in memory.results.values() for record in result.records
        }
        current = {
            record.reference
            for result in memory.results.values()
            if result.source != Source.HISTORY
            for record in result.records
        }
        assessments = {cause.cause_id: cause for cause in assess_causes(memory)}
        checks = prioritize_checks(context.request, memory)
        check_order = {check.tool.value: index for index, check in enumerate(checks)}
        check_values = {
            check.tool.value: {"high": 18, "medium": 12, "low": 6}[check.priority]
            for check in checks
        }

        def record(**kwargs) -> ThoughtNode:
            if len(nodes) >= self.limits.max_nodes:
                finish("node_limit")
                raise PlannerError("ToT node limit reached. Ask the engineer whether to continue.")
            node = ThoughtNode(**kwargs)
            nodes.append(node)
            return node

        def replace(node: ThoughtNode, **changes) -> ThoughtNode:
            updated = node.model_copy(update=changes)
            nodes[nodes.index(node)] = updated
            return updated

        def finish(reason, selected=None):
            search = ThoughtSearch(
                search_number=number,
                observed_after_call=len(memory.audit),
                limits=self.limits,
                nodes=nodes.copy(),
                selected_node_id=selected.node_id if selected else None,
                selected_tool=selected.planned_tools[0] if selected else None,
                stop_reason=reason,
            )
            context.search = search
            self.searches.append(search)
            self._events.append(ToTProgress(search=search))

        def score(cause_id, steps):
            cause = assessments.get(cause_id)
            supporting = (
                {note.reference for note in cause.supporting_evidence} & current if cause else set()
            )
            opposing = (
                {note.reference for note in cause.opposing_evidence} & current if cause else set()
            )
            parts = {
                "current_support": min(2, len(supporting)) * 12,
                "current_opposition": -min(2, len(opposing)) * 10,
                "state_support": 12 if cause and cause.status == "possible" else 0,
                "local_error_lead": 24
                if cause_id == "recent_deployment"
                and memory.has_fact(ToolName.LOGS, "local_exception", True)
                and ToolName.DEPLOYMENTS in context.allowed
                else 0,
                "historical_lead": 0,
                "planned_check_value": 0,
            }
            target = cause.next_tool if cause and cause.next_tool else CAUSES[cause_id][1].value
            # A past lead adds limited planning value, never current-source support.
            history = memory.results.get(ToolName.SIMILAR) if context.request.use_rag else None
            if (
                history
                and cause_id == "broad_service_issue"
                and not memory.has_fact(ToolName.LOGS, "local_exception", True)
                and not memory.has_facts(
                    ToolName.TRACES,
                    service="settlement",
                    successful_response=False,
                    retries_exhausted=True,
                )
                and ToolName.MONITORING not in memory.results
                and any(
                    item.facts.get("useful_check") == ToolName.MONITORING.value
                    and memory.has_fact(ToolName.LOGS, "service", item.facts.get("service"))
                    for item in history.records
                )
            ):
                parts["historical_lead"] = 6

            def value(tool):
                return check_values[tool] + (16 if tool == target else 8)

            if not steps:
                parts["planned_check_value"] = max(
                    value(tool) for tool in proposals[cause_id].candidate_tools
                )
            else:
                parts["planned_check_value"] = value(steps[0])
                if len(steps) > 1:
                    parts["planned_check_value"] += round(value(steps[1]) * 0.35)
            return max(0, min(100, sum(parts.values()))), parts

        def order(node):
            tools = node.planned_tools or proposals[node.cause_id].candidate_tools
            return (
                -node.score,
                -node.score_parts.get("current_support", 0),
                min(check_order[tool] for tool in tools)
                if not node.planned_tools
                else check_order[tools[0]],
                list(CAUSES).index(node.cause_id),
                tuple(tools),
            )

        def keep(candidates):
            eligible = []
            for node in candidates:
                if node.score < self.limits.minimum_score:
                    replace(
                        node,
                        status="pruned",
                        reason="Planning score below the threshold; cause remains open.",
                    )
                else:
                    eligible.append(node)
            eligible.sort(key=order)
            selected = []
            for index, node in enumerate(eligible):
                if index < self.limits.beam_width:
                    selected.append(replace(node, status="kept"))
                else:
                    replace(node, status="pruned", reason="Outside the beam width; not ruled out.")
            return selected

        root = record(
            node_id=f"s{number}-root",
            parent_id=None,
            cause_id=None,
            depth=0,
            score=0,
            status="root",
            reason="Current observed evidence; future tool results are unknown.",
        )
        candidates = []
        for index, proposal in enumerate(context.batch.proposals):
            safe = set(proposal.evidence_references) <= known and set(
                proposal.candidate_tools
            ) <= set(context.allowed)
            value, parts = score(proposal.cause_id, []) if safe else (0, {})
            node = record(
                node_id=f"s{number}-h{index}",
                parent_id=root.node_id,
                cause_id=proposal.cause_id,
                depth=1,
                evidence_references=list(
                    dict.fromkeys(
                        [
                            *[
                                note.reference
                                for note in (
                                    [
                                        *assessments[proposal.cause_id].supporting_evidence,
                                        *assessments[proposal.cause_id].opposing_evidence,
                                    ]
                                    if proposal.cause_id in assessments
                                    else []
                                )
                                if note.reference in current
                            ],
                            *proposal.evidence_references,
                        ]
                    )
                )[:12]
                if safe
                else [],
                score=value,
                score_parts=parts,
                status="candidate" if safe else "pruned",
                reason=proposal.summary
                if safe
                else "Rejected: unavailable tool or unobserved source.",
            )
            if safe:
                candidates.append(node)
        frontier = keep(candidates)
        # Depth 1 is a cause; depth 2 proposes a check; depth 3 adds one lookahead check.
        depth_limit = min(self.limits.max_depth, context.remaining_calls + 1)
        for depth in range(2, depth_limit + 1):
            children = []
            for parent in frontier:
                proposal = proposals[parent.cause_id]
                for tool in sorted(proposal.candidate_tools, key=lambda name: check_order[name]):
                    if tool in parent.planned_tools:
                        continue
                    steps = [*parent.planned_tools, tool]
                    value, parts = score(parent.cause_id, steps)
                    children.append(
                        record(
                            node_id=f"{parent.node_id}-{tool}",
                            parent_id=parent.node_id,
                            cause_id=parent.cause_id,
                            depth=depth,
                            planned_tools=steps,
                            evidence_references=parent.evidence_references,
                            score=value,
                            score_parts=parts,
                            status="candidate",
                            reason=(
                                "Planned check; future observations and the cause remain unknown."
                            ),
                        )
                    )
            if not children:
                break
            frontier = keep(children)
        selectable = [node for node in frontier if node.planned_tools]
        if not selectable:
            finish("no_valid_path")
            raise PlannerError(
                "No ToT path met the evidence and planning checks. Ask the engineer."
            )
        selected = replace(sorted(selectable, key=order)[0], status="selected")
        finish("selected", selected)
        return context

    def _select(self, context: SearchContext) -> Action:
        search = context.search
        node = next(node for node in search.nodes if node.node_id == search.selected_node_id)
        proposal = next(item for item in context.batch.proposals if item.cause_id == node.cause_id)
        check = next(
            item
            for item in prioritize_checks(context.request, context.memory)
            if item.tool.value == search.selected_tool
        )
        references = tuple(
            dict.fromkeys([*check.evidence_references, *proposal.evidence_references])
        )
        return Action(
            ToolName(search.selected_tool),
            f"ToT compared possible paths. {check.reason}",
            references,
        )
