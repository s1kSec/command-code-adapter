from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class AnthropicToolChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["auto", "any", "tool", "none"] = "auto"
    name: str | None = None


class AnthropicToolParam(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class AnthropicThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: int | None = None


class AnthropicRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    max_tokens: int = 4096
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[AnthropicToolParam] | None = None
    tool_choice: AnthropicToolChoice | None = None
    thinking: AnthropicThinkingConfig | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    stream: bool = False


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
