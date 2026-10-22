"""Turn API failures into useful messages without showing keys or provider bodies."""

from json import JSONDecodeError

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
)
from pydantic import ValidationError


def _error_field(error: APIStatusError, name: str) -> str | None:
    value = getattr(error, name, None)
    if isinstance(value, str) and value:
        return value
    if isinstance(error.body, dict):
        nested = error.body.get("error")
        body = nested if isinstance(nested, dict) else error.body
        value = body.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _describe_limit(error: APIStatusError) -> str:
    # Specific billing codes take priority over the broader error type.
    messages = {
        "insufficient_quota": (
            "API credit/quota limit reached (HTTP 429). Check API billing and available credit."
        ),
        "credit_balance_exhausted": (
            "API credit balance is empty (HTTP 429). Check API billing and add credit to continue."
        ),
        "organization_spend_limit_exceeded": (
            "API organization spend limit reached (HTTP 429). "
            "Review the organization spending limit."
        ),
        "project_spend_limit_exceeded": (
            "API project spend limit reached (HTTP 429). Review this project's spending limit."
        ),
        "organization_usage_limit_exceeded": (
            "API organization usage limit reached (HTTP 429). Request a higher API usage limit."
        ),
        "rate_limit_exceeded": (
            "API rate limit reached (HTTP 429). Wait briefly before trying again."
        ),
        "slow_down": (
            "API rate limit: requests increased too quickly (HTTP 429). "
            "Wait and reduce the request rate."
        ),
    }
    code = _error_field(error, "code")
    if code in messages:
        return messages[code]
    error_type = _error_field(error, "type")
    if error_type == "insufficient_quota":
        return messages["insufficient_quota"]
    if error_type == "rate_limit_error":
        return messages["rate_limit_exceeded"]
    return "API limit reached (HTTP 429). Check API credits and rate limits."


def describe_api_error(error: Exception) -> str:
    """Use fixed messages and known codes. Never display str(error) or raw bodies.

    The category helps the engineer choose the next step; it does not silently
    retry the API request, switch planners, or change an investigation finding.
    """
    # A timeout is also an APIConnectionError, so check it first.
    if isinstance(error, APITimeoutError):
        return "AI API request timeout. Check your connection and try again."
    if isinstance(error, APIConnectionError):
        return "Could not connect to the AI API. Check your internet, VPN, or proxy settings."
    if isinstance(error, (APIResponseValidationError, ValidationError, JSONDecodeError)):
        return (
            "The AI output was not a valid decision. Check the response format and client version."
        )
    if isinstance(error, APIStatusError):
        status, code = error.status_code, _error_field(error, "code")
        if status == 401:
            return "API key rejected (HTTP 401). Check the key and selected API project."
        if status == 403:
            return "API access denied (HTTP 403). Check project permissions and model access."
        if status == 404:
            return (
                "API model or endpoint not found (HTTP 404). "
                "Check OPENAI_MODEL, model access, and the API base URL."
            )
        if status == 429:
            return _describe_limit(error)
        if status == 400:
            if code in ("model_not_found", "unsupported_model"):
                return "API model rejected (HTTP 400). Check the model name and API access."
            return (
                "API request rejected (HTTP 400). "
                "Check the model, output schema, and request settings."
            )
        if status == 422:
            return "API could not read the request (HTTP 422). Check the request fields."
        if isinstance(status, int) and not isinstance(status, bool) and 500 <= status <= 599:
            return f"AI API service error (HTTP {status}). Wait briefly before trying again."
        if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599:
            return f"API request failed (HTTP {status}). Check API access and request settings."
    return "Unexpected API client error. Check the installed OpenAI package and client settings."
