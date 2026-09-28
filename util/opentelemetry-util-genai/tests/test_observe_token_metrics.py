# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics of semantic-conventions-genai#374: usage
counters by modality, and per-operation histograms. Specific to this fork."""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.metrics.export import Histogram, Sum
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, Modality

USAGE = "gen_ai.client.inference.usage."
OPERATION = "gen_ai.client.inference.operation."
MODALITY = "gen_ai.token.modality"


class InferenceTokenMetricsTest(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _metrics(self) -> dict[str, Any]:
        return {m.name: m for m in self.get_sorted_metrics() or []}

    def _counts(self, name: str) -> dict[str, int]:
        """Counter value per modality."""
        metric = self._metrics()[USAGE + name]
        return {
            p.attributes[MODALITY]: p.value for p in metric.data.data_points
        }

    def _chat(self, **tokens: Any) -> Any:
        invocation = self.handler.inference("openai", request_model="gpt-5")
        for field, value in tokens.items():
            setattr(invocation, field, value)
        return invocation

    def test_counters_split_by_modality_and_histograms_by_operation(
        self,
    ) -> None:
        chat = self._chat(
            input_tokens=1000,
            output_tokens=300,
            cache_read_input_tokens=800,
            cache_write_input_tokens=100,
            thinking_tokens=200,
        )
        chat.set_input_tokens([(Modality.TEXT, 700), (Modality.IMAGE, 300)])
        chat.stop()

        self.assertEqual(
            self._counts("input_tokens"), {"text": 700, "image": 300}
        )
        # Nothing attributed output or the subsets to a modality.
        self.assertEqual(self._counts("output_tokens"), {"unknown": 300})
        self.assertEqual(
            self._counts("cache_read.input_tokens"), {"unknown": 800}
        )
        self.assertEqual(
            self._counts("cache_write.input_tokens"), {"unknown": 100}
        )
        self.assertEqual(
            self._counts("reasoning.output_tokens"), {"unknown": 200}
        )
        metrics = self._metrics()
        for name, total in (("input_tokens", 1000), ("output_tokens", 300)):
            (point,) = metrics[OPERATION + name].data.data_points
            self.assertEqual((point.count, point.sum), (1, total))
            self.assertNotIn(MODALITY, point.attributes)

    def test_the_part_no_modality_accounts_for_is_unknown(self) -> None:
        chat = self._chat(input_tokens=1000, cache_read_input_tokens=500)
        chat.set_input_tokens([(Modality.TEXT, 600)])
        chat.set_cache_read_input_tokens([(Modality.AUDIO, 500)])
        chat.stop()
        self.assertEqual(
            self._counts("input_tokens"), {"text": 600, "unknown": 400}
        )
        self.assertEqual(
            self._counts("cache_read.input_tokens"), {"audio": 500}
        )

    def test_output_is_split_by_modality(self) -> None:
        chat = self._chat(output_tokens=50)
        chat.set_output_tokens([(Modality.AUDIO, 30)])
        chat.stop()
        self.assertEqual(
            self._counts("output_tokens"), {"audio": 30, "unknown": 20}
        )

    def test_a_breakdown_without_a_total_is_not_recorded(self) -> None:
        chat = self._chat(output_tokens=5)
        chat.set_input_tokens([(Modality.TEXT, 600)])
        chat.stop()
        self.assertNotIn(USAGE + "input_tokens", self._metrics())

    def test_a_breakdown_larger_than_the_total_counts_as_unknown(
        self,
    ) -> None:
        chat = self._chat(input_tokens=100)
        chat.set_input_tokens([(Modality.TEXT, 150)])
        chat.stop()
        self.assertEqual(self._counts("input_tokens"), {"unknown": 100})

    def test_instrument_types_and_units(self) -> None:
        self._chat(
            input_tokens=10,
            output_tokens=3,
            cache_read_input_tokens=4,
            cache_write_input_tokens=2,
            thinking_tokens=1,
        ).stop()
        metrics = self._metrics()
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read.input_tokens",
            "cache_write.input_tokens",
            "reasoning.output_tokens",
        ):
            metric = metrics[USAGE + name]
            self.assertEqual(metric.unit, "{token}")
            self.assertIsInstance(metric.data, Sum)
            self.assertTrue(metric.data.is_monotonic)
            (point,) = metric.data.data_points
            self.assertIsInstance(point.value, int)
        for name in ("input_tokens", "output_tokens"):
            metric = metrics[OPERATION + name]
            self.assertEqual(metric.unit, "{token}")
            self.assertIsInstance(metric.data, Histogram)

    def test_token_metrics_carry_the_operations_metric_attributes(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(agent_name="support-bot")
        self._chat(input_tokens=10, output_tokens=3).stop()
        agent.stop()

        for name in (USAGE + "input_tokens", OPERATION + "output_tokens"):
            (point,) = self._metrics()[name].data.data_points
            self.assertEqual(
                point.attributes[GenAI.GEN_AI_OPERATION_NAME], "chat"
            )
            self.assertEqual(
                point.attributes[GenAI.GEN_AI_PROVIDER_NAME], "openai"
            )
            self.assertEqual(
                point.attributes[GenAI.GEN_AI_REQUEST_MODEL], "gpt-5"
            )
            self.assertEqual(
                point.attributes[GenAI.GEN_AI_AGENT_NAME], "support-bot"
            )

    def test_a_failed_chat_records_what_it_reported(self) -> None:
        chat = self._chat(input_tokens=10)
        chat.fail(Error(message="boom", type="RuntimeError"))
        (point,) = self._metrics()[USAGE + "input_tokens"].data.data_points
        self.assertEqual(point.value, 10)
        self.assertEqual(point.attributes["error.type"], "RuntimeError")

    def test_nothing_is_recorded_without_token_counts(self) -> None:
        self._chat().stop()
        metrics = self._metrics()
        self.assertFalse(
            [n for n in metrics if n.startswith((USAGE, OPERATION))]
        )

    def test_the_old_token_usage_histogram_is_still_recorded(self) -> None:
        self._chat(input_tokens=10, output_tokens=3).stop()
        points = self._metrics()["gen_ai.client.token.usage"].data.data_points
        self.assertEqual(
            {p.attributes[GenAI.GEN_AI_TOKEN_TYPE]: p.sum for p in points},
            {"input": 10, "output": 3},
        )

    def test_embeddings_record_input_tokens_only(self) -> None:
        embedding = self.handler.embedding("openai", request_model="embed-3")
        embedding.input_tokens = 8
        embedding.stop()

        metrics = self._metrics()
        self.assertEqual(self._counts("input_tokens"), {"unknown": 8})
        (point,) = metrics[OPERATION + "input_tokens"].data.data_points
        self.assertEqual(
            point.attributes[GenAI.GEN_AI_OPERATION_NAME], "embeddings"
        )
        self.assertNotIn(USAGE + "output_tokens", metrics)
        self.assertNotIn(OPERATION + "output_tokens", metrics)

    def test_an_agent_records_no_inference_token_metrics(self) -> None:
        """A remote agent reports its tokens on the old histogram only: the
        inference metrics describe inference operations."""
        agent = self.handler.invoke_remote_agent(
            "openai", agent_name="support-bot"
        )
        agent.input_tokens, agent.output_tokens = 100, 20
        agent.stop()
        metrics = self._metrics()
        self.assertFalse(
            [n for n in metrics if n.startswith((USAGE, OPERATION))]
        )
        self.assertIn("gen_ai.client.token.usage", metrics)

    def test_histograms_use_the_specified_bucket_boundaries(self) -> None:
        self._chat(input_tokens=10).stop()
        (point,) = self._metrics()[OPERATION + "input_tokens"].data.data_points
        self.assertEqual(
            list(point.explicit_bounds),
            [
                1,
                4,
                16,
                64,
                256,
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                67108864,
            ],
        )
