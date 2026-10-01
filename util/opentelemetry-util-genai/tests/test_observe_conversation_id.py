# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tools, embeddings, retrieval and fetched responses carry the conversation id
their context holds, which semconv leaves off them. Specific to this fork."""

from __future__ import annotations

from typing import Any

from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.invocation import (
    RetrievalInvocation,
    ToolInvocation,
)
from observeinc.util.genai.types import Error
from opentelemetry.context import attach, detach, get_current, set_value
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.test_base import TestBase

CONVERSATION = GenAI.GEN_AI_CONVERSATION_ID


class ConversationIdOnEverySpanTest(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _by_operation(self) -> dict[str, Any]:
        return {
            s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s
            for s in self.get_finished_spans()
        }

    def _run_each_operation(self) -> None:
        self.handler.tool("lookup").stop()
        self.handler.embedding("openai", request_model="embed-3").stop()
        self.handler.retrieval(data_source_id="kb").stop()
        self.handler.fetch_response("openai", response_id="resp_1").stop()

    def test_operations_under_a_conversation_carry_its_id(self) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="support-bot", conversation_id="conv-42"
        )
        self._run_each_operation()
        agent.stop()

        spans = self._by_operation()
        for operation in (
            "execute_tool",
            "embeddings",
            "retrieval",
            "fetch_response",
        ):
            self.assertEqual(
                spans[operation].attributes[CONVERSATION], "conv-42"
            )

    def test_operations_outside_a_conversation_carry_none(self) -> None:
        self._run_each_operation()
        for span in self.get_finished_spans():
            self.assertNotIn(CONVERSATION, span.attributes)

    def test_an_id_set_after_creation_is_used(self) -> None:
        tool = self.handler.tool("lookup")
        tool.conversation_id = "conv-late"
        tool.stop()
        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "conv-late",
        )

    def test_a_value_the_caller_sets_wins(self) -> None:
        agent = self.handler.invoke_local_agent(conversation_id="conv-42")
        tool = self.handler.tool("lookup")
        tool.attributes[CONVERSATION] = "caller-set"
        tool.stop()
        agent.stop()
        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "caller-set",
        )

    def test_a_failed_operation_still_carries_it(self) -> None:
        agent = self.handler.invoke_local_agent(conversation_id="conv-42")
        tool = self.handler.tool("lookup")
        tool.fail(Error(message="boom", type="RuntimeError"))
        agent.stop()
        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "conv-42",
        )

    def test_operations_semconv_covers_are_unchanged(self) -> None:
        agent = self.handler.invoke_local_agent(conversation_id="conv-42")
        self.handler.inference("openai", request_model="gpt-5").stop()
        self.handler.workflow("triage").stop()
        agent.stop()
        spans = self._by_operation()
        for operation in ("chat", "invoke_workflow", "invoke_agent"):
            self.assertEqual(
                spans[operation].attributes[CONVERSATION], "conv-42"
            )

    def test_a_child_parented_explicitly_carries_its_parents_id(self) -> None:
        """How claude-agent-sdk parents its calls, with the agent not current."""
        agent = self.handler.invoke_local_agent(conversation_id="conv-42")
        agent.suspend()
        self.handler.tool("lookup", context=agent.context).stop()
        agent.stop()
        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "conv-42",
        )

    def test_an_id_set_on_the_parent_later_reaches_explicit_children(
        self,
    ) -> None:
        """How the langchain instrumentation sets it on its agents and workflows."""
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        agent.conversation_id = "conv-late"
        self.handler.tool("lookup", context=agent.context).stop()
        agent.stop()
        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "conv-late",
        )

    def test_an_id_set_on_the_current_parent_later_reaches_its_children(
        self,
    ) -> None:
        before = get_current()
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        agent.conversation_id = "conv-late"
        self.handler.tool("lookup").stop()
        agent.stop()

        self.assertEqual(
            self._by_operation()["execute_tool"].attributes[CONVERSATION],
            "conv-late",
        )
        self.assertIs(get_current(), before)

    def test_a_later_id_leaves_a_context_attached_on_top_alone(self) -> None:
        """Swapping the parent's context then would detach out of order."""
        before = get_current()
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        on_top = attach(set_value("unrelated", 1))
        current = get_current()
        agent.conversation_id = "conv-late"
        self.assertIs(get_current(), current)
        detach(on_top)
        agent.stop()
        self.assertIs(get_current(), before)

    def test_setting_the_id_it_already_has_leaves_the_context_alone(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(conversation_id="conv-42")
        tool = self.handler.tool("lookup")
        current = get_current()
        tool.conversation_id = "conv-42"
        self.assertIs(get_current(), current)
        tool.stop()
        agent.stop()


class ConversationIdAtCreationTest(TestBase):
    """``tool()`` and ``retrieval()`` take a ``conversation_id`` like
    ``inference()`` and ``workflow()`` do."""

    def setUp(self) -> None:
        super().setUp()
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _start(
        self, operation: str, conversation_id: str | None
    ) -> ToolInvocation | RetrievalInvocation:
        if operation == "execute_tool":
            return self.handler.tool("lookup", conversation_id=conversation_id)
        return self.handler.retrieval(
            data_source_id="kb", conversation_id=conversation_id
        )

    def _ids(self) -> dict[str, Any]:
        return {
            s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s.attributes.get(
                CONVERSATION
            )
            for s in self.get_finished_spans()
        }

    def test_the_id_is_on_the_span_and_its_children(self) -> None:
        for operation in ("execute_tool", "retrieval"):
            with self.subTest(operation):
                self.memory_exporter.clear()
                invocation = self._start(operation, "conv-42")
                self.handler.inference("openai").stop()
                invocation.stop()
                self.assertEqual(
                    self._ids(), {operation: "conv-42", "chat": "conv-42"}
                )

    def test_none_inherits_the_ambient_id(self) -> None:
        for operation in ("execute_tool", "retrieval"):
            with self.subTest(operation):
                self.memory_exporter.clear()
                agent = self.handler.invoke_local_agent(
                    conversation_id="conv-42"
                )
                self._start(operation, None).stop()
                agent.stop()
                self.assertEqual(self._ids()[operation], "conv-42")

    def test_an_explicit_id_overrides_the_ambient_one(self) -> None:
        for operation in ("execute_tool", "retrieval"):
            with self.subTest(operation):
                self.memory_exporter.clear()
                agent = self.handler.invoke_local_agent(
                    conversation_id="conv-42"
                )
                invocation = self._start(operation, "conv-7")
                self.handler.inference("openai").stop()
                invocation.stop()
                agent.stop()
                self.assertEqual(
                    self._ids(),
                    {
                        operation: "conv-7",
                        "chat": "conv-7",
                        "invoke_agent": "conv-42",
                    },
                )
