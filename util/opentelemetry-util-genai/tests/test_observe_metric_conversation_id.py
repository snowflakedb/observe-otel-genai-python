# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID puts the conversation id on
every metric. Specific to this fork."""

from __future__ import annotations

import os
from unittest import mock

from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.types import LLMInvocation
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.test_base import TestBase

CONVERSATION = GenAI.GEN_AI_CONVERSATION_ID
FLAG = "OBSERVE_GENAI_METRICS_INCLUDE_CONVERSATION_ID"


class MetricConversationIdTest(TestBase):
    def _handler(self, flag: str | None) -> TelemetryHandler:
        with mock.patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            if flag is not None:
                os.environ[FLAG] = flag
            return TelemetryHandler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )

    def _conversation_ids(self) -> dict[str, set[object]]:
        """The conversation ids each metric's data points carry."""
        return {
            m.name: {
                p.attributes.get(CONVERSATION) for p in m.data.data_points
            }
            for m in self.get_sorted_metrics() or []
        }

    def _run_every_operation(self, handler: TelemetryHandler) -> None:
        agent = handler.invoke_local_agent(
            agent_name="support-bot", conversation_id="conv-42"
        )
        workflow = handler.workflow("triage")
        chat = handler.inference("openai", request_model="gpt-5")
        chat.input_tokens, chat.output_tokens = 10, 3
        chat.record_stream_chunk()
        chat.record_stream_chunk()
        chat.stop()
        embedding = handler.embedding("openai", request_model="embed-3")
        embedding.input_tokens = 8
        embedding.stop()
        handler.tool("lookup").stop()
        handler.retrieval(data_source_id="kb").stop()
        handler.fetch_response("openai", response_id="resp_1").stop()
        workflow.stop()
        agent.stop()

    def test_every_metric_carries_it_when_enabled(self) -> None:
        self._run_every_operation(self._handler("true"))
        ids = self._conversation_ids()
        self.assertEqual(
            set(ids),
            {
                "gen_ai.client.operation.duration",
                "gen_ai.client.token.usage",
                "gen_ai.client.operation.time_to_first_chunk",
                "gen_ai.client.operation.time_per_output_chunk",
                "gen_ai.client.inference.usage.input_tokens",
                "gen_ai.client.inference.usage.output_tokens",
                "gen_ai.client.inference.operation.input_tokens",
                "gen_ai.client.inference.operation.output_tokens",
                "gen_ai.execute_tool.duration",
                "gen_ai.invoke_agent.duration",
                "gen_ai.invoke_workflow.duration",
            },
        )
        for name, values in ids.items():
            self.assertEqual(values, {"conv-42"}, name)

    def test_no_metric_carries_it_by_default(self) -> None:
        self._run_every_operation(self._handler(None))
        for name, values in self._conversation_ids().items():
            self.assertEqual(values, {None}, name)

    def test_the_flag_is_case_insensitive_and_only_true_enables_it(
        self,
    ) -> None:
        for flag in ("TRUE", "1", "yes"):
            tool = self._handler(flag).tool("lookup")
            tool.conversation_id = f"conv-{flag}"
            tool.stop()
        self.assertEqual(
            self._conversation_ids()["gen_ai.execute_tool.duration"],
            {"conv-TRUE", None},
        )

    def test_an_id_set_after_creation_is_used(self) -> None:
        chat = self._handler("true").inference("openai", request_model="gpt-5")
        chat.record_stream_chunk()
        chat.conversation_id = "conv-late"
        chat.stop()
        ids = self._conversation_ids()
        self.assertEqual(
            ids["gen_ai.client.operation.duration"], {"conv-late"}
        )
        # Recorded before the id was set.
        self.assertEqual(
            ids["gen_ai.client.operation.time_to_first_chunk"], {None}
        )

    def test_a_cleared_id_is_left_off(self) -> None:
        handler = self._handler("true")
        agent = handler.invoke_local_agent(conversation_id="conv-42")
        tool = handler.tool("lookup")
        tool.conversation_id = None
        tool.stop()
        agent.stop()
        ids = self._conversation_ids()
        self.assertEqual(ids["gen_ai.execute_tool.duration"], {None})
        self.assertEqual(ids["gen_ai.invoke_agent.duration"], {"conv-42"})

    def test_replacing_metric_attributes_keeps_it(self) -> None:
        handler = self._handler("true")
        agent = handler.invoke_local_agent(conversation_id="conv-42")
        chat = handler.inference("openai", request_model="gpt-5")
        chat.metric_attributes = {"custom": "x"}
        chat.stop()
        agent.stop()
        self.assertEqual(
            self._conversation_ids()["gen_ai.client.operation.duration"],
            {"conv-42"},
        )

    def test_the_start_stop_llm_api_carries_it(self) -> None:
        handler = self._handler("true")
        agent = handler.invoke_local_agent(conversation_id="conv-42")
        llm = LLMInvocation(provider="openai", request_model="gpt-5")
        handler.start_llm(llm)
        handler.stop_llm(llm)
        agent.stop()
        self.assertEqual(
            self._conversation_ids()["gen_ai.client.operation.duration"],
            {"conv-42"},
        )
