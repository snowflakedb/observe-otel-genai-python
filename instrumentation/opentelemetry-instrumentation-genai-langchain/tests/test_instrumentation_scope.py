# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from observeinc.instrumentation.genai.langchain import LangChainInstrumentor
from observeinc.instrumentation.genai.langchain.version import __version__
from observeinc.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = LangChainInstrumentor
    instrumentation_scope_name = "observeinc.instrumentation.genai.langchain"
    instrumentation_scope_version = __version__
