# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Maps Claude Agent SDK content blocks onto util-genai message parts.

The SDK hands back its own block dataclasses. util-genai owns the serialized
shape of ``gen_ai.input.messages`` / ``gen_ai.output.messages``, so blocks are
translated into its part types and never into hand-built dicts.
"""

from __future__ import annotations

from typing import Any, cast

from opentelemetry.util.genai.types import (
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    ServerToolCallPart,
    ServerToolCallResponsePart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
)


def block_to_parts(block: Any) -> list[MessagePart]:
    """Translate one SDK content block into zero or more message parts.

    Blocks are matched on the attributes they carry rather than by importing the
    SDK's classes, so a block type added by a newer SDK degrades to a
    ``GenericPart`` instead of being dropped or raising.
    """
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return [TextPart(content=text)] if text else []

    thinking = getattr(block, "thinking", None)
    if isinstance(thinking, str):
        return [ReasoningPart(content=thinking)] if thinking else []

    tool_use_id = getattr(block, "tool_use_id", None)
    if isinstance(tool_use_id, str):
        content = getattr(block, "content", None)
        # A server tool's result content is a raw dict; a regular tool's is text
        # or a list of blocks.
        if isinstance(content, dict):
            return [
                ServerToolCallResponsePart(
                    server_tool_call_response=content, id=tool_use_id
                )
            ]
        return [ToolCallResponsePart(response=content, id=tool_use_id)]

    name = getattr(block, "name", None)
    block_id = getattr(block, "id", None)
    if isinstance(name, str) and isinstance(block_id, str):
        tool_input = getattr(block, "input", None)
        if _is_server_tool(block):
            return [
                ServerToolCallPart(
                    name=name, server_tool_call=tool_input, id=block_id
                )
            ]
        return [
            ToolCallRequestPart(arguments=tool_input, name=name, id=block_id)
        ]

    return [GenericPart(type=type(block).__name__)]


def _is_server_tool(block: Any) -> bool:
    """True when the API executed this tool server-side on the model's behalf.

    Server tool calls never come back to the caller for a result, so they are
    not tool executions we can span. Recognized by class name because the block
    otherwise looks identical to a regular tool_use.
    """
    return type(block).__name__.startswith("ServerTool")


def blocks_of(content: object) -> list[object]:
    """Normalize a message's ``content`` to a list of blocks.

    Blocks stay ``object`` on the way through: every caller identifies each one
    with ``isinstance`` against the SDK's block classes.
    """
    if isinstance(content, list):
        return cast("list[object]", content)
    if isinstance(content, str) and content:
        return [_TextOnly(content)]
    return []


class _TextOnly:
    """Adapts a plain-string message content to the block protocol."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def parts_of(blocks: list[Any]) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for block in blocks:
        parts.extend(block_to_parts(block))
    return parts


def input_message(blocks: list[Any], role: str) -> InputMessage | None:
    parts = parts_of(blocks)
    return InputMessage(role=role, parts=parts) if parts else None


def output_message(
    blocks: list[Any], role: str, finish_reason: str | None
) -> OutputMessage | None:
    parts = parts_of(blocks)
    if not parts:
        return None
    return OutputMessage(
        role=role, parts=parts, finish_reason=finish_reason or "stop"
    )


def prompt_to_messages(prompt: Any) -> list[InputMessage]:
    """Render the caller's prompt as the agent span's input.

    A prompt is either a string or, in streaming-input mode, an async iterable
    of message dicts. The iterable cannot be read without consuming what the
    SDK needs, so only the string form produces input.
    """
    if isinstance(prompt, str) and prompt:
        return [InputMessage(role="user", parts=[TextPart(content=prompt)])]
    return []


def text_output(text: str) -> list[OutputMessage]:
    return [
        OutputMessage(
            role="assistant",
            parts=[TextPart(content=text)],
            finish_reason="stop",
        )
    ]
