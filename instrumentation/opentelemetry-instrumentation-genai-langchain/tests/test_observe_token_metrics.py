# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics add up to the token counts on the spans, for
LangChain model responses. Specific to this fork."""

import pytest
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from opentelemetry.test_util_genai._observe_token_metrics import (
    assert_token_metrics_match_spans,
)

from .test_llm_call import (
    _openai_cassette_name,
    _supports_reasoning_token_details,
)

MESSAGES = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="What is the capital of France?"),
]


def test_chat_openai(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_openai_gpt_3_5_turbo_model,
    vcr,
):
    with vcr.use_cassette(
        _openai_cassette_name(
            chat_openai_gpt_3_5_turbo_model,
            "test_chat_openai_gpt_3_5_turbo_model_llm_call",
        )
    ):
        chat_openai_gpt_3_5_turbo_model.invoke(MESSAGES)
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_chat_openai_streaming(
    span_exporter, metric_reader, start_instrumentation, vcr
):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_sequences=["\n", "Human:", "AI:"],
        seed=100,
        stream_usage=True,
    )
    if "max_completion_tokens" not in llm._get_request_payload([], stop=None):
        pytest.skip("the cassette was recorded with a newer langchain-openai")
    with vcr.use_cassette("inference_streaming_conformance.yaml"):
        for _ in llm.stream(MESSAGES):
            pass
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


@pytest.mark.skipif(
    not _supports_reasoning_token_details,
    reason="langchain-openai < 0.2.1 does not surface reasoning token details",
)
def test_chat_openai_reasoning_tokens(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_openai_reasoning,
    vcr,
):
    messages = [
        SystemMessage(content="You are a careful mathematical reasoner."),
        HumanMessage(
            content=(
                "A snail climbs a 12 meter well. Each day it climbs up 3 "
                "meters, and each night it slides back 2 meters. On which day "
                "does it first reach the top? Think through it step by step."
            )
        ),
    ]
    payload = chat_openai_reasoning._get_request_payload(messages, stop=None)
    suffix = "_old" if "n" in payload else ""
    with vcr.use_cassette(
        f"test_chat_openai_reasoning_token_details{suffix}.yaml"
    ):
        chat_openai_reasoning.invoke(messages)
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["reasoning.output_tokens"]


def test_chat_anthropic_cache_tokens(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_anthropic_claude_sonnet,
    vcr,
):
    with vcr.use_cassette(
        "test_chat_anthropic_claude_sonnet_cache_token_details.yaml"
    ):
        chat_anthropic_claude_sonnet.invoke(MESSAGES)
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["cache_read.input_tokens"]
    assert usage["cache_write.input_tokens"]


def test_chat_bedrock(
    span_exporter,
    metric_reader,
    start_instrumentation,
    us_amazon_nova_lite_v1_0,
    vcr,
):
    old = not getattr(us_amazon_nova_lite_v1_0, "beta_use_converse_api", False)
    with vcr.use_cassette(
        "test_us_amazon_nova_lite_v1_0_bedrock_llm_call"
        f"{'_old' if old else ''}.yaml"
    ):
        us_amazon_nova_lite_v1_0.invoke(MESSAGES)
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_tokens_split_by_modality(
    span_exporter, metric_reader, start_instrumentation
):
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="A cat.",
                    usage_metadata={
                        "input_tokens": 120,
                        "output_tokens": 50,
                        "total_tokens": 170,
                        "input_token_details": {
                            "text": 70,
                            "image": 20,
                            "audio": 10,
                        },
                        "output_token_details": {"text": 35, "audio": 5},
                    },
                )
            ]
        )
    )
    # The instrumentation skips a call it can't name a model for.
    model.invoke(MESSAGES, config={"metadata": {"ls_model_name": "fake"}})
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["input_tokens"] == {
        "text": 70,
        "image": 20,
        "audio": 10,
        "unknown": 20,
    }
    assert usage["output_tokens"] == {"text": 35, "audio": 5, "unknown": 10}
