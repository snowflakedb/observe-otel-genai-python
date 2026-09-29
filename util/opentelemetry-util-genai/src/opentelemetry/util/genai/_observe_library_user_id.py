# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in recording of a user id passed to the instrumented library. Specific
to this fork.

Off by default, so an id an application already passes to a library does not
start showing up in telemetry on upgrade.
"""

from __future__ import annotations

import os
from typing import Final

OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY: Final = (
    "OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY"
)


def capture_user_id_from_library() -> bool:
    return (
        os.environ.get(OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY, "")
        .strip()
        .lower()
        == "true"
    )
