# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ``gen_ai.conversation.id`` on every metric. Specific to this fork.

Semconv keeps the conversation id off metrics because it is high cardinality.
This lets a backend that can afford it break metrics down by conversation.
"""

from __future__ import annotations

import os
from typing import Final

OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID: Final = (
    "OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID"
)


def metrics_include_conversation_id() -> bool:
    return (
        os.environ.get(OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID, "")
        .strip()
        .lower()
        == "true"
    )
