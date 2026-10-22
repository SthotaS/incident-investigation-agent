"""Possible causes stay separate from current failure evidence and past fixes."""

import pytest
from conftest import result_for
from incident_client.hypotheses import assess_causes
from incident_client.memory import InvestigationMemory
from incident_common.models import ToolName


def _by_cause(memory):
    return {item.cause_id: item for item in assess_causes(memory)}


def _memory(**tools):
    return InvestigationMemory(results={ToolName(name): value for name, value in tools.items()})


def _window(*observations):
    failed = {item[0] for item in observations if item[2] == "failed"}
    succeeded = {item[0] for item in observations if item[2] == "succeeded"}
    return {
        "window_start": "2026-01-15T09:55:00Z",
        "window_end": "2026-01-15T10:10:00Z",
        "error_started_at": "2026-01-15T10:00:00Z",
        "service": "settlement",
        "checked_transaction_count": len(failed | succeeded),
        "affected_transaction_count": len(failed),
        "successful_transaction_count": len(succeeded),
        "observations": [
            {
                "transaction_id": transaction,
                "observed_at": f"2026-01-15T{time}Z",
                "outcome": outcome,
                "http_status": 503 if outcome == "failed" else 200,
                "reference": f"synthetic://monitoring/{transaction}/{time}",
            }
            for transaction, time, outcome in observations
        ],
    }


def _logs():
    return result_for(ToolName.LOGS, service="settlement", http_status=503)


def test_status_only_and_history_only_do_not_invent_current_causes():
    assert assess_causes(_memory(get_transaction=result_for(ToolName.TRANSACTION))) == []
    history = result_for(
        ToolName.SIMILAR,
        service="settlement",
        historical=True,
        past_root_cause="connection pool exhausted",
    )
    assert assess_causes(_memory(searchSimilarIncidents=history)) == []


def test_current_failures_need_health_and_transaction_comparison():
    memory = _memory(search_logs=_logs())
    causes = _by_cause(memory)
    broad = causes["broad_service_issue"]
    assert broad.status == "needs_evidence"
    assert broad.next_tool == "get_monitoring"
    assert "timing" in " ".join(broad.missing_evidence)
    assert causes["transaction_specific_issue"].next_tool == "get_trace"
    assert causes["recent_deployment"].next_tool == "get_deployments"


def test_nearby_failed_transactions_raise_shared_service_priority():
    monitoring = result_for(
        ToolName.MONITORING,
        service="settlement",
        elevated_errors=True,
        application_healthy=True,
        monitoring_window=_window(
            ("TX784521", "10:00:02", "failed"),
            ("TX784522", "10:00:04", "failed"),
            ("TX784523", "10:01:00", "failed"),
            ("TX784519", "09:56:00", "succeeded"),
        ),
    )
    memory = _memory(search_logs=_logs(), get_monitoring=monitoring)
    causes = _by_cause(memory)
    broad = causes["broad_service_issue"]
    assert broad.status == "possible" and broad.priority == "high"
    assert causes["transaction_specific_issue"].status == "less_supported"
    assert assess_causes(memory)[0].cause_id == "broad_service_issue"
    assert broad.next_tool is None
    assert "already returned" in broad.next_check
    note = next(item for item in broad.supporting_evidence if "TX784522" in item.observation)
    assert note.reference == monitoring.records[0].reference
    assert "2026-01-15T10:00:04" in note.observation
    assert "does not prove an outage" in note.observation


def test_isolated_failure_with_nearby_successes_prefers_transaction_check():
    memory = _memory(
        search_logs=_logs(),
        get_monitoring=result_for(
            ToolName.MONITORING,
            service="settlement",
            elevated_errors=False,
            application_healthy=True,
            monitoring_window=_window(
                ("TX784521", "10:00:02", "failed"),
                ("TX784522", "10:00:04", "succeeded"),
                ("TX784523", "10:01:00", "succeeded"),
            ),
        ),
    )
    causes = _by_cause(memory)
    specific = causes["transaction_specific_issue"]
    assert specific.status == "possible" and specific.priority == "high"
    assert specific.next_tool == "get_trace"
    assert causes["broad_service_issue"].status == "less_supported"
    assert "not proof" in specific.supporting_evidence[-1].observation


def test_failures_outside_five_minutes_do_not_establish_shared_timing():
    memory = _memory(
        search_logs=_logs(),
        get_monitoring=result_for(
            ToolName.MONITORING,
            service="settlement",
            monitoring_window=_window(
                ("TX784521", "10:00:02", "failed"),
                ("TX784522", "10:09:00", "failed"),
                ("TX784523", "10:09:01", "failed"),
            ),
        ),
    )
    broad = _by_cause(memory)["broad_service_issue"]
    assert broad.status == "needs_evidence" and broad.priority != "high"
    assert not any("other transactions" in item.observation for item in broad.supporting_evidence)


def test_no_recent_deployment_reduces_support_without_ruling_it_out():
    memory = _memory(
        search_logs=_logs(),
        get_deployments=result_for(
            ToolName.DEPLOYMENTS, relevant_change=False, check_completed=True
        ),
    )
    assessment = _by_cause(memory)["recent_deployment"]
    assert assessment.status == "less_supported" and assessment.next_tool is None
    assert "does not rule it out" in assessment.opposing_evidence[0].observation
    assert assessment.missing_evidence


def test_relevant_change_still_requires_evidence_linking_it_to_failure():
    memory = _memory(
        search_logs=_logs(),
        get_deployments=result_for(ToolName.DEPLOYMENTS, relevant_change=True),
    )
    assessment = _by_cause(memory)["recent_deployment"]
    assert assessment.status == "possible"
    assert "does not prove" in assessment.supporting_evidence[0].observation
    assert "linking" in assessment.missing_evidence[0]


def test_past_connection_pool_cause_is_only_a_manual_check_lead():
    history = result_for(
        ToolName.SIMILAR,
        service="settlement",
        historical=True,
        past_incident_id="INC-00022",
        past_root_cause="Settlement connection pool exhausted",
    )
    pool = _by_cause(_memory(search_logs=_logs(), searchSimilarIncidents=history))[
        "connection_pool_issue"
    ]
    assert pool.status == "needs_evidence" and pool.next_tool is None
    assert "not available" in pool.missing_evidence[0]
    assert pool.supporting_evidence[0].reference == history.records[0].reference
    assert "not evidence of the current cause" in pool.supporting_evidence[0].observation


@pytest.mark.parametrize(
    "facts",
    [
        {"past_root_cause": None},
        {"past_root_cause": "connection pool exhausted", "root_cause_confirmed": False},
    ],
)
def test_unknown_or_unconfirmed_saved_causes_do_not_create_pool_explanation(facts):
    history = result_for(ToolName.SIMILAR, service="settlement", historical=True, **facts)
    assert "connection_pool_issue" not in _by_cause(
        _memory(search_logs=_logs(), searchSimilarIncidents=history)
    )


def test_unrelated_service_results_cannot_support_settlement_causes():
    unrelated = result_for(ToolName.LOGS, service="billing", http_status=503)
    assert assess_causes(_memory(search_logs=unrelated)) == []
    memory = _memory(
        search_logs=_logs(),
        get_monitoring=result_for(ToolName.MONITORING, service="billing", elevated_errors=True),
    )
    assert _by_cause(memory)["broad_service_issue"].status == "needs_evidence"


def test_wrong_fact_type_does_not_turn_into_an_investigation_failure():
    malformed = result_for(ToolName.LOGS, service="settlement", http_status=[503])
    assert assess_causes(_memory(search_logs=malformed)) == []


def test_failed_or_empty_tool_is_not_suggested_again():
    memory = _memory(
        search_logs=_logs(), get_monitoring=result_for(ToolName.MONITORING, empty=True)
    )
    memory.attempts[ToolName.TRACES] = 1
    memory.attempts[ToolName.DEPLOYMENTS] = 1
    assert all(item.next_tool is None for item in assess_causes(memory))
    assert all("already attempted" in item.next_check for item in assess_causes(memory))


def test_bad_monitoring_window_stays_uncertain():
    memory = _memory(
        search_logs=_logs(),
        get_monitoring=result_for(
            ToolName.MONITORING, service="settlement", monitoring_window={"wrong": "field"}
        ),
    )
    broad = _by_cause(memory)["broad_service_issue"]
    assert broad.status == "needs_evidence"
    assert any("valid time window" in text for text in broad.missing_evidence)


def test_successful_current_trace_keeps_conflicting_failure_uncertain():
    memory = _memory(
        search_logs=_logs(),
        get_trace=result_for(ToolName.TRACES, service="settlement", successful_response=True),
        get_monitoring=result_for(ToolName.MONITORING, service="settlement", elevated_errors=True),
    )
    assessments = _by_cause(memory)
    assert assessments["broad_service_issue"].status == "needs_evidence"
    assert assessments["transaction_specific_issue"].status == "needs_evidence"
    assert any("Resolve" in text for text in assessments["broad_service_issue"].missing_evidence)
