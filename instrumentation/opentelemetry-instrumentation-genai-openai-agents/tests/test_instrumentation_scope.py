# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from observeinc.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from observeinc.instrumentation.genai.openai_agents.version import (
    __version__,
)
from observeinc.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = OpenAIAgentsInstrumentor
    instrumentation_scope_name = (
        "observeinc.instrumentation.genai.openai_agents"
    )
    instrumentation_scope_version = __version__
