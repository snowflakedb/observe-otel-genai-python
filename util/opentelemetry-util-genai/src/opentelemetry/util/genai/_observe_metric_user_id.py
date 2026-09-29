# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ``user.id`` on every metric. Specific to this fork.

It is high cardinality, so it is kept off metrics unless a backend that can
afford it asks for it.
"""

from __future__ import annotations

import os
from typing import Final

OBSERVE_GENAI_METRICS_INCLUDE_USER_ID: Final = (
    "OBSERVE_GENAI_METRICS_INCLUDE_USER_ID"
)


def metrics_include_user_id() -> bool:
    return (
        os.environ.get(OBSERVE_GENAI_METRICS_INCLUDE_USER_ID, "")
        .strip()
        .lower()
        == "true"
    )
