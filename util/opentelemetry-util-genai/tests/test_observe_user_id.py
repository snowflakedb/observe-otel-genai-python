# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``user.id`` set on the Context reaches every GenAI span, and a library's own
user id is added behind a flag. Specific to this fork."""

from __future__ import annotations

import os
import threading
from unittest import mock

from observeinc.util.genai._observe_library_user_id import (
    OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY,
)
from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.types import LLMInvocation
from observeinc.util.genai.user_id import (
    get_user_id,
    using_user_id,
    with_library_user_id,
    with_user_id,
)
from opentelemetry import context as otel_context
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
)
from opentelemetry.test.test_base import TestBase


class UserIdTest(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _span_user_ids(self) -> dict[str, object]:
        return {
            span.name: span.attributes.get(USER_ID)
            for span in self.get_finished_spans()
        }

    def _run_every_operation(self) -> None:
        handler = self.handler
        agent = handler.invoke_local_agent(agent_name="support-bot")
        workflow = handler.workflow("triage")
        handler.inference("openai", request_model="gpt-5").stop()
        handler.embedding("openai", request_model="embed-3").stop()
        handler.tool("lookup").stop()
        handler.retrieval(data_source_id="kb").stop()
        handler.fetch_response("openai", response_id="resp_1").stop()
        workflow.stop()
        agent.stop()

    def test_every_span_carries_it(self) -> None:
        with using_user_id("user-123"):
            self._run_every_operation()
        user_ids = self._span_user_ids()
        self.assertEqual(len(user_ids), 7, user_ids)
        for name, value in user_ids.items():
            self.assertEqual(value, "user-123", name)
            self.assertIsInstance(value, str)

    def test_no_span_carries_it_without_one(self) -> None:
        self._run_every_operation()
        for name, value in self._span_user_ids().items():
            self.assertIsNone(value, name)

    def test_an_empty_id_is_left_off(self) -> None:
        with using_user_id(""):
            self.handler.tool("lookup").stop()
        self.assertEqual(self._span_user_ids(), {"execute_tool lookup": None})

    def test_using_user_id_restores_the_context_on_exit(self) -> None:
        with using_user_id("user-123"):
            self.assertEqual(get_user_id(), "user-123")
        self.assertIsNone(get_user_id())

    def test_the_innermost_id_wins(self) -> None:
        with using_user_id("outer"):
            with using_user_id("inner"):
                self.handler.tool("inner-tool").stop()
            self.handler.tool("outer-tool").stop()
        self.assertEqual(
            self._span_user_ids(),
            {
                "execute_tool inner-tool": "inner",
                "execute_tool outer-tool": "outer",
            },
        )

    def test_a_context_passed_to_the_factory_reaches_its_children(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="support-bot",
            context=with_user_id("user-123"),
            _attach_to_context=False,
        )
        self.handler.inference(
            "openai", request_model="gpt-5", context=agent.context
        ).stop()
        agent.stop()
        self.handler.tool("unrelated").stop()
        self.assertEqual(
            self._span_user_ids(),
            {
                "invoke_agent support-bot": "user-123",
                "chat gpt-5": "user-123",
                "execute_tool unrelated": None,
            },
        )

    def test_a_failed_operation_carries_it(self) -> None:
        with using_user_id("user-123"):
            chat = self.handler.inference("openai", request_model="gpt-5")
            chat.fail(RuntimeError("boom"))
        (span,) = self.get_finished_spans()
        self.assertEqual(span.attributes[USER_ID], "user-123")
        self.assertEqual(span.attributes["error.type"], "RuntimeError")

    def test_it_does_not_cross_threads(self) -> None:
        with using_user_id("user-123"):
            thread = threading.Thread(
                target=lambda: self.handler.tool("t").stop()
            )
            thread.start()
            thread.join()
        self.assertEqual(self._span_user_ids(), {"execute_tool t": None})

    def test_an_explicit_attribute_wins(self) -> None:
        with using_user_id("user-123"):
            tool = self.handler.tool("lookup")
            tool.attributes[USER_ID] = "set-by-instrumentation"
            tool.stop()
        self.assertEqual(
            self._span_user_ids()["execute_tool lookup"],
            "set-by-instrumentation",
        )

    def test_the_start_stop_llm_api_carries_it(self) -> None:
        llm = LLMInvocation(provider="openai", request_model="gpt-5")
        with using_user_id("user-123"):
            self.handler.start_llm(llm)
            self.handler.stop_llm(llm)
        self.assertEqual(self._span_user_ids()["chat gpt-5"], "user-123")

    def test_it_stays_off_metrics(self) -> None:
        with using_user_id("user-123"):
            self._run_every_operation()
        for metric in self.get_sorted_metrics() or []:
            for point in metric.data.data_points:
                self.assertNotIn(USER_ID, point.attributes, metric.name)


class WithLibraryUserIdTest(TestBase):
    def _set_flag(self, value: str | None) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY, None)
        if value is not None:
            os.environ[OBSERVE_GENAI_CAPTURE_USER_ID_FROM_LIBRARY] = value

    def test_off_by_default(self) -> None:
        self._set_flag(None)
        parent = otel_context.get_current()
        self.assertIs(with_library_user_id("user-123", parent), parent)
        self.assertIsNone(with_library_user_id("user-123"))

    def test_the_flag_is_case_insensitive_and_only_true_enables_it(
        self,
    ) -> None:
        for flag in ("1", "yes"):
            self._set_flag(flag)
            self.assertIsNone(with_library_user_id("user-123"))
        self._set_flag(" TRUE ")
        self.assertEqual(
            get_user_id(with_library_user_id("user-123")), "user-123"
        )

    def test_an_empty_id_leaves_the_context_alone(self) -> None:
        self._set_flag("true")
        parent = with_user_id("from-helper")
        for empty in (None, ""):
            self.assertIs(with_library_user_id(empty, parent), parent)

    def test_it_overrides_using_user_id(self) -> None:
        self._set_flag("true")
        with using_user_id("from-helper"):
            self.assertEqual(
                get_user_id(with_library_user_id("from-library")),
                "from-library",
            )
