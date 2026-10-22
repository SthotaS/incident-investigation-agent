"""Validate timed monitoring evidence without a model, database, or network."""

import json
from copy import deepcopy
from importlib import resources

import pytest
from incident_common.models import Source
from incident_common.monitoring_models import MonitoringWindow, get_monitoring_window
from incident_server.repository import FixtureData, FixtureRepository
from pydantic import ValidationError


def fixture_data():
    return json.loads(
        resources.files("incident_server").joinpath("fixtures", "incidents.json").read_text()
    )


def monitoring_record(data):
    result = next(item for item in data["results"] if item["source"] == "monitoring")
    return result["records"][0]


def window_data():
    return monitoring_record(fixture_data())["facts"]["monitoring_window"]


def test_fixture_returns_timed_sample_alongside_existing_health_evidence():
    repository = FixtureRepository.load()
    result = repository.get(Source.MONITORING, "TX784521")
    facts = result.records[0].facts
    assert facts["service"] == "settlement"
    assert facts["elevated_errors"] is True
    assert facts["application_healthy"] is True
    window = get_monitoring_window(facts)
    assert window is not None
    assert window.checked_transaction_count == 6
    assert window.affected_transaction_count == 4
    assert window.successful_transaction_count == 2
    current = next(item for item in window.observations if item.transaction_id == "TX784521")
    assert current.outcome == "failed"
    assert current.http_status == 503
    other_failures = [
        item
        for item in window.observations
        if item.transaction_id != current.transaction_id and item.outcome == "failed"
    ]
    assert len(other_failures) == 3
    assert all(
        abs((item.observed_at - current.observed_at).total_seconds()) < 120
        for item in other_failures
    )
    successes = [item for item in window.observations if item.outcome == "succeeded"]
    assert all(item.observed_at < window.error_started_at for item in successes)


def test_optional_window_absence_is_distinct_from_malformed_data():
    assert get_monitoring_window({"elevated_errors": True}) is None
    for malformed in (None, {}, [], "not a monitoring sample"):
        with pytest.raises(ValidationError):
            get_monitoring_window({"monitoring_window": malformed})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_start", "2026-01-15T10:10:00Z"),
        ("window_end", "2026-01-15T09:50:00Z"),
        ("window_start", "2026-01-15T09:55:00"),
        ("error_started_at", "2026-01-15T10:30:00Z"),
        ("error_started_at", "2026-01-15T10:01:00Z"),
        ("checked_transaction_count", 7),
        ("affected_transaction_count", 3),
        ("successful_transaction_count", 1),
        ("checked_transaction_count", True),
    ],
)
def test_window_rejects_impossible_times_and_counts(field, value):
    window = window_data()
    window[field] = value
    with pytest.raises(ValidationError):
        MonitoringWindow.model_validate(window)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_at", "2026-01-15T10:30:00Z"),
        ("observed_at", "2026-01-15T09:56:00"),
        ("transaction_id", "121"),
        ("http_status", 503),
        ("http_status", "200"),
        ("reference", "https://example.com/live-data"),
    ],
)
def test_observations_reject_invalid_timestamps_ids_and_statuses(field, value):
    window = window_data()
    window["observations"][0][field] = value
    with pytest.raises(ValidationError):
        MonitoringWindow.model_validate(window)


def test_duplicate_transaction_and_reference_do_not_inflate_comparison():
    for field in ("transaction_id", "reference"):
        window = window_data()
        window["observations"][1][field] = window["observations"][0][field]
        with pytest.raises(ValidationError):
            MonitoringWindow.model_validate(window)


def test_failed_http_observation_cannot_claim_success_status():
    window = window_data()
    window["observations"][2]["http_status"] = 200
    with pytest.raises(ValidationError):
        MonitoringWindow.model_validate(window)


def test_fixture_validation_rejects_window_from_different_service_or_future():
    for field, value in (
        ("service", "authorization"),
        ("observed_at", "2026-01-15T10:05:00Z"),
    ):
        data = fixture_data()
        record = monitoring_record(data)
        if field == "service":
            record["facts"]["monitoring_window"][field] = value
        else:
            record[field] = value
        with pytest.raises(ValidationError):
            FixtureData.model_validate(data)


def test_monitoring_window_is_rejected_on_unrelated_source():
    data = fixture_data()
    data["results"][0]["records"][0]["facts"]["monitoring_window"] = deepcopy(window_data())
    with pytest.raises(ValidationError):
        FixtureData.model_validate(data)
