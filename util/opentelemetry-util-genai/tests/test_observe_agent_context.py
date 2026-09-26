# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Operations started under an agent record that agent's ``gen_ai.agent.name``,
on their span and on their metrics. Specific to this fork."""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, LLMInvocation

AGENT = GenAI.GEN_AI_AGENT_NAME


class AgentNameFromContextTest(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _span(self, name: str) -> Any:
        spans = [s for s in self.get_finished_spans() if s.name == name]
        self.assertEqual(
            len(spans), 1, [s.name for s in self.get_finished_spans()]
        )
        return spans[0]

    def _points(self, metric: str) -> list[Any]:
        return [
            point
            for m in self.get_sorted_metrics() or []
            if m.name == metric
            for point in m.data.data_points
        ]

    def _chat(self, **kwargs: Any) -> None:
        invocation = self.handler.inference(
            "openai", request_model="gpt-5", **kwargs
        )
        invocation.input_tokens, invocation.output_tokens = 10, 3
        invocation.stop()

    def test_chat_under_an_agent_records_its_name(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self._chat()
        agent.stop()

        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "support-bot"
        )
        for metric in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            points = self._points(metric)
            self.assertTrue(points, metric)
            for point in points:
                self.assertEqual(point.attributes[AGENT], "support-bot")

    def test_streamed_chat_timing_metrics_record_it(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        invocation = self.handler.inference("openai", request_model="gpt-5")
        invocation.record_stream_chunk()
        invocation.record_stream_chunk()
        invocation.stop()
        agent.stop()

        for metric in (
            "gen_ai.client.operation.time_to_first_chunk",
            "gen_ai.client.operation.time_per_output_chunk",
        ):
            points = self._points(metric)
            self.assertTrue(points, metric)
            self.assertEqual(points[0].attributes[AGENT], "support-bot")

    def test_a_nested_agent_names_the_calls_inside_it(self) -> None:
        outer = self.handler.invoke_local_agent(agent_name="coordinator")
        self._chat()
        inner = self.handler.invoke_local_agent(agent_name="researcher")
        self._chat()
        inner.stop()
        self._chat()
        outer.stop()

        chats = sorted(
            (s for s in self.get_finished_spans() if s.name == "chat gpt-5"),
            key=lambda s: s.start_time,
        )
        self.assertEqual(
            [s.attributes[AGENT] for s in chats],
            ["coordinator", "researcher", "coordinator"],
        )

    def test_a_remote_agent_names_the_calls_inside_it(self) -> None:
        agent = self.handler.invoke_remote_agent(
            "openai", agent_name="remote-bot"
        )
        self._chat()
        agent.stop()
        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "remote-bot"
        )

    def test_a_chat_outside_any_agent_records_none(self) -> None:
        self._chat()
        self.assertNotIn(AGENT, self._span("chat gpt-5").attributes)
        for point in self._points("gen_ai.client.operation.duration"):
            self.assertNotIn(AGENT, point.attributes)

    def test_the_name_does_not_outlive_the_agent(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        agent.stop()
        self._chat()
        self.assertNotIn(AGENT, self._span("chat gpt-5").attributes)

    def test_an_explicit_parent_context_carries_the_name(self) -> None:
        """How an instrumentation that does not keep the agent current, like
        claude-agent-sdk, parents its calls."""
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        agent.suspend()
        self._chat(context=agent.context)
        agent.stop()
        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "support-bot"
        )

    def test_a_failed_chat_still_records_it(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        invocation = self.handler.inference("openai", request_model="gpt-5")
        invocation.fail(Error(message="boom", type="RuntimeError"))
        agent.stop()

        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "support-bot"
        )
        point = self._points("gen_ai.client.operation.duration")[0]
        self.assertEqual(point.attributes[AGENT], "support-bot")
        self.assertEqual(point.attributes["error.type"], "RuntimeError")

    def test_embeddings_and_retrieval_record_it(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        embedding = self.handler.embedding("openai", request_model="embed-3")
        embedding.input_tokens = 4
        embedding.stop()
        self.handler.retrieval(data_source_id="kb").stop()
        self.handler.fetch_response("openai", response_id="resp_1").stop()
        agent.stop()

        for span in self.get_finished_spans():
            if span.attributes.get(GenAI.GEN_AI_OPERATION_NAME) in (
                "embeddings",
                "retrieval",
                "fetch_response",
            ):
                self.assertEqual(span.attributes[AGENT], "support-bot")
        operations = {
            p.attributes[GenAI.GEN_AI_OPERATION_NAME]: p.attributes.get(AGENT)
            for p in self._points("gen_ai.client.operation.duration")
        }
        self.assertEqual(
            operations,
            {
                "embeddings": "support-bot",
                "retrieval": "support-bot",
                "fetch_response": "support-bot",
            },
        )

    def test_a_tool_falls_back_to_the_enclosing_agent(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self.handler.tool("lookup_order").stop()
        agent.stop()

        self.assertEqual(
            self._span("execute_tool lookup_order").attributes[AGENT],
            "support-bot",
        )
        point = self._points("gen_ai.execute_tool.duration")[0]
        self.assertEqual(point.attributes[AGENT], "support-bot")

    def test_a_tool_keeps_the_agent_name_it_was_given(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self.handler.tool("lookup_order", agent_name="order-agent").stop()
        agent.stop()

        self.assertEqual(
            self._span("execute_tool lookup_order").attributes[AGENT],
            "order-agent",
        )
        point = self._points("gen_ai.execute_tool.duration")[0]
        self.assertEqual(point.attributes[AGENT], "order-agent")

    def test_a_workflow_inside_an_agent_is_not_named_after_it(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self.handler.workflow("triage").stop()
        agent.stop()
        self.assertNotIn(
            AGENT, self._span("invoke_workflow triage").attributes
        )

    def test_an_agent_records_only_its_own_name(self) -> None:
        outer = self.handler.invoke_local_agent(agent_name="coordinator")
        self.handler.invoke_local_agent(agent_name="researcher").stop()
        outer.stop()
        self.assertEqual(
            self._span("invoke_agent researcher").attributes[AGENT],
            "researcher",
        )

    def test_an_attribute_the_caller_sets_wins(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        invocation = self.handler.inference("openai", request_model="gpt-5")
        invocation.attributes[AGENT] = "caller-named"
        invocation.metric_attributes[AGENT] = "caller-named"
        invocation.stop()
        agent.stop()

        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "caller-named"
        )
        point = self._points("gen_ai.client.operation.duration")[0]
        self.assertEqual(point.attributes[AGENT], "caller-named")

    def test_an_unnamed_agent_hides_the_outer_agent_name(self) -> None:
        """The nearest agent owns the call, named or not."""
        outer = self.handler.invoke_local_agent(agent_name="coordinator")
        inner = self.handler.invoke_local_agent()
        self._chat()
        self.handler.tool("lookup").stop()
        inner.stop()
        outer.stop()

        self.assertNotIn(AGENT, self._span("chat gpt-5").attributes)
        self.assertNotIn(AGENT, self._span("execute_tool lookup").attributes)

    def test_the_deprecated_start_llm_api_records_it(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        invocation = LLMInvocation(request_model="legacy-model")
        self.handler.start_llm(invocation)
        self.handler.stop_llm(invocation)
        agent.stop()

        span = self._span("chat legacy-model")
        self.assertEqual(span.attributes[AGENT], "support-bot")
        point = self._points("gen_ai.client.operation.duration")[0]
        self.assertEqual(point.attributes[AGENT], "support-bot")

    def test_an_explicit_agent_name_overrides_the_enclosing_one(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self._chat(agent_name="billing-bot")
        agent.stop()

        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "billing-bot"
        )
        point = self._points("gen_ai.client.operation.duration")[0]
        self.assertEqual(point.attributes[AGENT], "billing-bot")

    def test_an_explicit_agent_name_is_accepted_by_every_factory(self) -> None:
        self.handler.embedding("openai", agent_name="bot").stop()
        self.handler.retrieval(data_source_id="kb", agent_name="bot").stop()
        self.handler.fetch_response(
            "openai", response_id="resp_1", agent_name="bot"
        ).stop()
        for span in self.get_finished_spans():
            self.assertEqual(span.attributes[AGENT], "bot", span.name)

    def test_an_empty_agent_name_is_kept_rather_than_inherited(self) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self._chat(agent_name="")
        agent.stop()
        self.assertEqual(self._span("chat gpt-5").attributes[AGENT], "")

    def test_a_chat_inside_a_workflow_inside_an_agent_records_the_agent(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        workflow = self.handler.workflow("triage")
        self._chat()
        workflow.stop()
        agent.stop()
        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "support-bot"
        )

    def test_an_explicit_agent_name_reaches_the_calls_under_it(self) -> None:
        """Passed on like an explicit conversation id."""
        agent = self.handler.invoke_local_agent(agent_name="coordinator")
        tool = self.handler.tool("delegate", agent_name="specialist")
        self._chat()
        tool.stop()
        agent.stop()
        self.assertEqual(
            self._span("chat gpt-5").attributes[AGENT], "specialist"
        )

    def test_an_unnamed_agent_under_a_named_tool_hides_the_tools_name(
        self,
    ) -> None:
        outer = self.handler.invoke_local_agent(agent_name="outer")
        tool = self.handler.tool("delegate", agent_name="specialist")
        inner = self.handler.invoke_local_agent()
        self._chat()
        inner.stop()
        tool.stop()
        outer.stop()

        self.assertNotIn(AGENT, self._span("chat gpt-5").attributes)
        self.assertEqual(
            self._span("execute_tool delegate").attributes[AGENT], "specialist"
        )
