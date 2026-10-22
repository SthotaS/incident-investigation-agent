"""Local runtime settings. No model key is needed for the dummy-data demo."""

import os
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    tool_timeout_seconds: float = Field(default=10, gt=0, le=120)
    api_timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_api_input_bytes: int = Field(default=131072, ge=4096, le=524288)
    max_api_calls: int = Field(default=8, ge=1, le=8)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            tool_timeout_seconds=os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "10"),
            log_level=os.getenv("LOG_LEVEL", "WARNING"),
            api_timeout_seconds=os.getenv("OPENAI_TIMEOUT_SECONDS", "30"),
            max_api_input_bytes=os.getenv("MAX_API_INPUT_BYTES", "131072"),
            max_api_calls=os.getenv("MAX_API_CALLS", "8"),
        )
