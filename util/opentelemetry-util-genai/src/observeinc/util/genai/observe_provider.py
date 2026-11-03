# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``gen_ai.provider.name`` from the host a GenAI client sends requests to.
Specific to this fork.

A client built for one provider's API, such as the OpenAI or Anthropic SDK, is
often pointed at another provider serving the same API. The host is then the
only sign of which provider serves the request.
"""

from __future__ import annotations

import re

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

_Provider = GenAIAttributes.GenAiProviderNameValues

# Not a semconv well-known value; the JS SDK reports it too.
_SNOWFLAKE = "snowflake"

# Providers serving OpenAI's or Anthropic's API, by base URL host. Keep in sync
# with PROVIDER_BY_HOST in the JS SDK.
_PROVIDER_BY_HOST: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), provider)
    for pattern, provider in (
        (r"(^|\.)snowflakecomputing\.com$", _SNOWFLAKE),
        (r"\.openai\.azure\.com$", _Provider.AZURE_AI_OPENAI.value),
        (
            r"\.(services|inference)\.ai\.azure\.com$",
            _Provider.AZURE_AI_INFERENCE.value,
        ),
        (
            r"^bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com$",
            _Provider.AWS_BEDROCK.value,
        ),
        (r"^api\.groq\.com$", _Provider.GROQ.value),
        (r"^api\.deepseek\.com$", _Provider.DEEPSEEK.value),
        (r"^api\.x\.ai$", _Provider.X_AI.value),
        (r"^api\.mistral\.ai$", _Provider.MISTRAL_AI.value),
        (
            r"^generativelanguage\.googleapis\.com$",
            _Provider.GCP_GEMINI.value,
        ),
    )
)


def provider_from_host(host: str | None) -> str | None:
    """The provider serving ``host``, or ``None`` for a host not in the table."""
    if host:
        for pattern, provider in _PROVIDER_BY_HOST:
            if pattern.search(host):
                return provider
    return None
