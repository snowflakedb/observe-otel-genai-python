# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from observeinc.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from observeinc.instrumentation.genai.claude_agent_sdk.version import (
    __version__,
)
from observeinc.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = ClaudeAgentSDKInstrumentor
    instrumentation_scope_name = (
        "observeinc.instrumentation.genai.claude_agent_sdk"
    )
    instrumentation_scope_version = __version__
