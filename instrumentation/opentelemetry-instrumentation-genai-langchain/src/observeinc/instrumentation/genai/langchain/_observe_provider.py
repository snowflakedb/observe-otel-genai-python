# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The real provider of a chat model built on the OpenAI client, from the base
URL it was constructed with. Specific to this fork."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from observeinc.util.genai.observe_provider import provider_from_host
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

_OPENAI = GenAIAttributes.GenAiProviderNameValues.OPENAI.value


def resolve_provider(provider: str, serialized: dict[str, Any]) -> str:
    """``provider``, with ``openai`` replaced by the provider its base URL names.

    Only a base URL passed to the constructor is in ``serialized``; one taken
    from ``OPENAI_BASE_URL`` is not.
    """
    if provider != _OPENAI:
        return provider
    kwargs: dict[str, Any] = serialized.get("kwargs") or {}
    base_url = kwargs.get("openai_api_base")
    if not isinstance(base_url, str):
        return provider
    return provider_from_host(urlparse(base_url).hostname) or provider
