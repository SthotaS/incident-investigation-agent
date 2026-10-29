"""Exercise real LangChain ToT search using synthetic observations and no API key."""

from collections import Counter

import pytest
from conftest import FakeGateway, result_for
from incident_client.investigation import InvestigationEngine
from incident_client.memory import InvestigationMemory
from incident_client.policy import DemoPlanner, InvestigationLimits
from incident_client.reporting import render_progress, render_summary
from incident_client.tot import ToTPlanner
from incident_common.models import InvestigationRequest, ToolName
from incident_common.tot_models import ToTLimits, ToTProgress
from pydantic import ValidationError
from test_rag_flow import RagGateway, historical_result


def observed(results):
    return InvestigationMemory(
        results={tool: results[tool] for tool in (ToolName.TRANSACTION, ToolName.LOGS)}
    )


@pytest.mark.parametrize("use_rag", [False, True])
async def test_tot_runs_all_checks_with_one_budget_and_real_search_tree(results, use_rag):
    results[ToolName.SIMILAR] = historical_result()
    gateway = RagGateway(results)
    planner = ToTPlanner(DemoPlanner())
    events = []
    report = await InvestigationEngine(gateway, planner=planner).run(
        InvestigationRequest(use_rag=use_rag), on_progress=events.append
    )
    assert report.status == "finished" and report.reasoning == "tot"
    assert report.calls_used == len(gateway.calls) == (6 if use_rag else 5)
    assert report.call_limit == 8 and report.model_calls == 0
    assert report.supported_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed and not report.knowledge_base_saved
    assert not report.production_changes_made
    assert len(report.thought_searches) == 2
    assert sum(isinstance(item, ToTProgress) for item in events) == 2
    expected_first = ToolName.MONITORING if use_rag else ToolName.TRACES
    assert report.thought_searches[0].selected_tool == expected_first.value
    for search in report.thought_searches:
        nodes = {node.node_id: node for node in search.nodes}
        assert len(nodes) == len(search.nodes) <= 16
        assert sum(node.depth == 1 for node in nodes.values()) == 3
        assert max(node.depth for node in nodes.values()) == 3
        assert any(node.status == "pruned" for node in nodes.values())
        assert len([node for node in nodes.values() if node.status == "selected"]) == 1
        for node in nodes.values():
            if node.parent_id is not None:
                assert nodes[node.parent_id].depth + 1 == node.depth
            assert len(node.planned_tools) == len(set(node.planned_tools))
        # The plan's next tool was really executed, using the one original tool counter.
        assert report.audit[search.observed_after_call].tool.value == search.selected_tool
        assert all(
            len(
                [
                    node
                    for node in nodes.values()
                    if node.depth == depth and node.status in {"kept", "selected"}
                ]
            )
            <= 2
            for depth in (1, 2, 3)
        )


async def test_new_observation_updates_scores_without_fabricating_later_evidence(results):
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    first, second = report.thought_searches

    def cause_score(search):
        return next(
            node.score
            for node in search.nodes
            if node.depth == 1 and node.cause_id == "broad_service_issue"
        )

    assert cause_score(second) > cause_score(first)
    for search in report.thought_searches:
        allowed_refs = {
            reference
            for entry in report.audit[: search.observed_after_call]
            for reference in entry.record_references
        }
        assert all(set(node.evidence_references) <= allowed_refs for node in search.nodes)


async def test_pruned_cause_is_still_reported_as_unconfirmed(results):
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    pruned = [
        node
        for node in report.thought_searches[0].nodes
        if node.depth == 1 and node.cause_id == "recent_deployment"
    ]
    assert pruned and pruned[0].status == "pruned" and "not ruled out" in pruned[0].reason
    recent = next(
        cause for cause in report.cause_assessments if cause.cause_id == "recent_deployment"
    )
    assert recent.status == "less_supported"
    assert not report.root_cause_confirmed


@pytest.mark.parametrize("budget", [3, 4])
async def test_search_and_all_branches_share_the_remaining_tool_budget(results, budget):
    gateway = FakeGateway(results)
    report = await InvestigationEngine(
        gateway,
        planner=ToTPlanner(DemoPlanner()),
        limits=InvestigationLimits(max_tool_calls=budget),
    ).run(InvestigationRequest(use_rag=False))
    assert report.status == "paused" and report.stop_reason == "call_budget"
    assert report.calls_used == len(gateway.calls) == budget
    for search in report.thought_searches:
        available = budget - search.observed_after_call
        assert max(len(node.planned_tools) for node in search.nodes) <= available


@pytest.mark.parametrize("failure", ["failed", "empty"])
async def test_no_new_evidence_pause_is_not_reset_by_tree_generation(results, failure):
    gateway = FakeGateway(results, always_fail={ToolName.TRACES} if failure == "failed" else None)
    if failure == "empty":
        for tool in (ToolName.TRACES, ToolName.MONITORING, ToolName.DEPLOYMENTS):
            results[tool] = result_for(tool, empty=True)
    report = await InvestigationEngine(gateway, planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    assert report.stop_reason == "no_new_evidence" and report.calls_used == 4
    assert report.supported_failure_point is None
    if failure == "failed":
        assert Counter(gateway.calls)[ToolName.TRACES] == 2
        assert len(report.thought_searches) == 1


async def test_two_current_sources_remain_required_with_tot(results):
    results[ToolName.TRACES] = result_for(ToolName.TRACES, empty=True)
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    assert report.status == "paused" and report.stop_reason == "insufficient_evidence"
    assert report.supported_failure_point is None
    assert report.possible_failure_point == "Settlement-service call"


async def test_retries_and_all_branches_stop_at_eight_shared_calls(results):
    results[ToolName.SIMILAR] = historical_result()
    gateway = RagGateway(results, fail_first={ToolName.LOGS, ToolName.MONITORING})
    report = await InvestigationEngine(gateway, planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=True)
    )
    assert report.calls_used == len(gateway.calls) == 8
    assert report.status == "paused" and report.stop_reason == "call_budget"
    assert [entry.call_number for entry in report.audit] == list(range(1, 9))
    assert Counter(gateway.calls)[ToolName.LOGS] == 2
    assert Counter(gateway.calls)[ToolName.MONITORING] == 2
    assert report.supported_failure_point == "Settlement-service call"
    assert not report.root_cause_confirmed and not report.knowledge_base_saved


async def test_current_local_error_takes_priority_over_past_monitoring_lead(results):
    results[ToolName.LOGS].records[0].facts["local_exception"] = True
    memory = observed(results)
    memory.results[ToolName.SIMILAR] = historical_result()
    planner = ToTPlanner(DemoPlanner())

    action = await planner.choose(InvestigationRequest(use_rag=True), memory)

    assert action.tool == ToolName.DEPLOYMENTS
    causes = [node for node in planner.searches[0].nodes if node.depth == 1]
    assert all(node.score_parts["historical_lead"] == 0 for node in causes)
    assert not any(
        node.status == "pruned" for node in causes if node.cause_id == "recent_deployment"
    )


async def test_past_bonus_stops_when_new_current_trace_guides_the_next_check(results):
    memory = observed(results)
    memory.results[ToolName.SIMILAR] = historical_result()
    memory.results[ToolName.TRACES] = results[ToolName.TRACES]
    planner = ToTPlanner(DemoPlanner())
    action = await planner.choose(InvestigationRequest(use_rag=True), memory)
    assert action.tool == ToolName.MONITORING
    assert all(
        node.score_parts.get("historical_lead", 0) == 0 for node in planner.searches[0].nodes
    )


async def test_conflicting_current_trace_pauses_before_more_search(results):
    results[ToolName.TRACES] = result_for(
        ToolName.TRACES, service="settlement", successful_response=True
    )
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    assert report.stop_reason == "conflicting_evidence" and report.calls_used == 3
    assert len(report.thought_searches) == 1 and report.supported_failure_point is None


async def test_status_question_bypasses_tot_and_stops_after_one_tool(results):
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(scope="status", use_rag=False)
    )
    assert report.status == "finished" and report.calls_used == 1
    assert report.thought_searches == []


async def test_ties_are_repeatable_across_independent_runs(results):
    reports = []
    for _ in range(2):
        reports.append(
            await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
                InvestigationRequest(use_rag=False)
            )
        )
    assert reports[0].thought_searches == reports[1].thought_searches


@pytest.mark.parametrize(
    "limits",
    [
        {"beam_width": 3},
        {"max_depth": 4},
        {"branching_factor": 4},
        {"max_nodes": 17},
        {"max_searches": 4},
        {"minimum_score": -1},
    ],
)
def test_limits_cannot_expand_search_boundaries(limits):
    with pytest.raises(ValidationError):
        ToTLimits(**limits)


@pytest.mark.parametrize(
    "limits,reason",
    [
        (ToTLimits(max_nodes=1), "node_limit"),
        (ToTLimits(minimum_score=100), "no_valid_path"),
    ],
)
async def test_search_failure_pauses_and_keeps_its_public_audit(results, limits, reason):
    report = await InvestigationEngine(
        FakeGateway(results), planner=ToTPlanner(DemoPlanner(), limits=limits)
    ).run(InvestigationRequest(use_rag=False))
    assert report.status == "paused" and report.stop_reason == "planner_error"
    assert report.calls_used == 2 and report.planner_error
    assert report.thought_searches[-1].stop_reason == reason


async def test_search_round_limit_cannot_start_another_unbounded_search(results):
    report = await InvestigationEngine(
        FakeGateway(results), planner=ToTPlanner(DemoPlanner(), limits=ToTLimits(max_searches=1))
    ).run(InvestigationRequest(use_rag=False))
    assert report.status == "paused" and report.calls_used == 3
    assert "search limit" in report.planner_error


async def test_reusing_wrapper_for_new_investigation_does_not_reuse_old_branch_state(results):
    planner = ToTPlanner(DemoPlanner())
    first = await InvestigationEngine(FakeGateway(results), planner=planner).run(
        InvestigationRequest(use_rag=False)
    )
    second = await InvestigationEngine(FakeGateway(results), planner=planner).run(
        InvestigationRequest(scope="status", use_rag=False)
    )
    assert len(first.thought_searches) == 2 and second.thought_searches == []


async def test_terminal_and_json_expose_public_paths_and_pruning(results):
    report = await InvestigationEngine(FakeGateway(results), planner=ToTPlanner(DemoPlanner())).run(
        InvestigationRequest(use_rag=False)
    )
    event = ToTProgress(search=report.thought_searches[0])
    compact, verbose = render_progress(event), render_progress(event, verbose=True)
    assert "[ToT 1]" in compact and "[3/8]" not in compact
    assert "Planning scores" in compact and "not ruled out" in compact
    assert "Sources:" in verbose and "pruned" in verbose
    assert "s1-root" in report.model_dump_json()
    assert "not cause probabilities" in render_summary(report)
