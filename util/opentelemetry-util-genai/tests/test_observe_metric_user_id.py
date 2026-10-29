# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OBSERVE_GENAI_METRICS_INCLUDE_USER_ID puts the user id on every metric.
Specific to this fork."""

from __future__ import annotations

import os
from unittest import mock

from observeinc.util.genai.handler import TelemetryHandler
from observeinc.util.genai.user_id import using_user_id
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
)
from opentelemetry.test.test_base import TestBase

FLAG = "OBSERVE_GENAI_METRICS_INCLUDE_USER_ID"


class MetricUserIdTest(TestBase):
    def _handler(self, flag: str | None) -> TelemetryHandler:
        with mock.patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            if flag is not None:
                os.environ[FLAG] = flag
            return TelemetryHandler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
            )

    def _user_ids(self) -> dict[str, set[object]]:
        """The user ids each metric's data points carry."""
        return {
            m.name: {p.attributes.get(USER_ID) for p in m.data.data_points}
            for m in self.get_sorted_metrics() or []
        }

    @staticmethod
    def _run_every_operation(handler: TelemetryHandler) -> None:
        agent = handler.invoke_local_agent(agent_name="support-bot")
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
        with using_user_id("user-123"):
            self._run_every_operation(self._handler("true"))
        user_ids = self._user_ids()
        self.assertEqual(
            set(user_ids),
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
        for name, values in user_ids.items():
            self.assertEqual(values, {"user-123"}, name)

    def test_no_metric_carries_it_by_default(self) -> None:
        with using_user_id("user-123"):
            self._run_every_operation(self._handler(None))
        for name, values in self._user_ids().items():
            self.assertEqual(values, {None}, name)

    def test_the_flag_is_case_insensitive_and_only_true_enables_it(
        self,
    ) -> None:
        for flag in (" TRUE ", "1", "yes"):
            with using_user_id(f"user-{flag.strip()}"):
                self._handler(flag).tool("lookup").stop()
        self.assertEqual(
            self._user_ids()["gen_ai.execute_tool.duration"],
            {"user-TRUE", None},
        )

    def test_a_failed_operation_carries_it(self) -> None:
        with using_user_id("user-123"):
            chat = self._handler("true").inference(
                "openai", request_model="gpt-5"
            )
            chat.fail(RuntimeError("boom"))
        (point,) = [
            p
            for m in self.get_sorted_metrics() or []
            if m.name == "gen_ai.client.operation.duration"
            for p in m.data.data_points
        ]
        self.assertEqual(point.attributes[USER_ID], "user-123")
        self.assertEqual(point.attributes["error.type"], "RuntimeError")

    def test_replacing_metric_attributes_keeps_it(self) -> None:
        with using_user_id("user-123"):
            chat = self._handler("true").inference(
                "openai", request_model="gpt-5"
            )
            chat.metric_attributes = {"custom": "x"}
            chat.stop()
        self.assertEqual(
            self._user_ids()["gen_ai.client.operation.duration"],
            {"user-123"},
        )
