"""Separate API billing limits from short-term request limits without exposing bodies."""

import httpx
import pytest
from incident_client.api_errors import describe_api_error
from openai import AsyncOpenAI, RateLimitError

PRIVATE_TEXT = "private API body sk-secret-placeholder private-project-name"


def limit_error(body):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return RateLimitError(
        PRIVATE_TEXT,
        response=httpx.Response(429, request=request),
        body=body,
    )


def assert_safe(message):
    assert "429" in message
    for value in (
        PRIVATE_TEXT,
        "sk-secret-placeholder",
        "private-project-name",
        "unrecognized-private-code",
        "unrecognized-private-type",
    ):
        assert value not in message


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("insufficient_quota", ("quota",)),
        ("credit_balance_exhausted", ("credit",)),
        ("organization_spend_limit_exceeded", ("organization", "spend")),
        ("project_spend_limit_exceeded", ("project", "spend")),
        ("organization_usage_limit_exceeded", ("organization", "usage")),
    ],
)
def test_known_api_limit_codes_explain_the_billing_or_account_limit(code, expected):
    error = limit_error({"code": code, "type": "insufficient_quota", "message": PRIVATE_TEXT})
    message = describe_api_error(error)
    assert all(word in message.lower() for word in expected)
    assert "rate limit" not in message.lower()
    assert_safe(message)


def test_slow_down_is_a_short_term_rate_limit_with_wait_guidance():
    error = limit_error({"code": "slow_down", "type": "rate_limit_error", "message": PRIVATE_TEXT})
    message = describe_api_error(error)
    assert "rate" in message.lower() and "wait" in message.lower()
    assert "billing" not in message.lower()
    assert_safe(message)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"type": "insufficient_quota", "message": PRIVATE_TEXT}, "quota"),
        (
            {
                "code": "unrecognized-private-code",
                "type": "insufficient_quota",
                "message": PRIVATE_TEXT,
            },
            "quota",
        ),
        (
            {
                "error": {
                    "code": None,
                    "type": "insufficient_quota",
                    "message": PRIVATE_TEXT,
                }
            },
            "quota",
        ),
        (
            {
                "error": {
                    "code": "unrecognized-private-code",
                    "type": "rate_limit_error",
                    "message": PRIVATE_TEXT,
                }
            },
            "rate",
        ),
    ],
)
def test_safe_error_type_explains_limits_when_code_does_not(body, expected):
    message = describe_api_error(limit_error(body))
    assert expected in message.lower()
    if expected == "rate":
        assert "wait" in message.lower()
    else:
        assert "rate limit" not in message.lower()
    assert_safe(message)


@pytest.mark.parametrize(
    ("code", "error_type", "expected", "unexpected"),
    [
        ("slow_down", "insufficient_quota", "rate", "credit/quota"),
        ("project_spend_limit_exceeded", "rate_limit_error", "project", "rate limit reached"),
    ],
)
def test_specific_limit_code_takes_priority_over_a_broader_type(
    code, error_type, expected, unexpected
):
    error = limit_error({"code": code, "type": error_type, "message": PRIVATE_TEXT})
    message = describe_api_error(error)
    assert expected in message.lower()
    assert unexpected not in message.lower()
    if expected == "rate":
        assert "wait" in message.lower()
    assert_safe(message)


def test_unknown_limit_code_and_type_stay_generic_and_private():
    error = limit_error(
        {
            "code": "unrecognized-private-code",
            "type": "unrecognized-private-type",
            "message": PRIVATE_TEXT,
        }
    )
    message = describe_api_error(error)
    assert "limit" in message.lower()
    assert "rate limit reached" not in message.lower()
    assert "credit/quota limit reached" not in message.lower()
    assert_safe(message)


async def test_installed_sdk_recognizes_current_credit_error_without_a_live_request():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "credit_balance_exhausted",
                    "type": "insufficient_quota",
                    "message": PRIVATE_TEXT,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncOpenAI(
            api_key="sk-secret-placeholder", http_client=http_client, max_retries=0
        ) as client:
            with pytest.raises(RateLimitError) as failure:
                await client.responses.create(model="synthetic-test-model", input="Synthetic")

    message = describe_api_error(failure.value)
    assert "credit" in message.lower()
    assert "rate limit" not in message.lower()
    assert len(requests) == 1
    assert_safe(message)
