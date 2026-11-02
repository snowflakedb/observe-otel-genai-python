# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from observeinc.instrumentation.genai.openai import OpenAIInstrumentor
from observeinc.instrumentation.genai.openai.version import __version__
from observeinc.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = OpenAIInstrumentor
    instrumentation_scope_name = "observeinc.instrumentation.genai.openai"
    instrumentation_scope_version = __version__
