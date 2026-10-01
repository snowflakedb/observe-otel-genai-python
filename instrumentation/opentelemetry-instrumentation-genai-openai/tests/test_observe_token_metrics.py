# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics add up to the token counts on the spans, for
real OpenAI responses. Specific to this fork."""

import pytest

from observeinc.test_util_genai._observe_token_metrics import (
    assert_token_metrics_match_spans,
)
from observeinc.util.genai.utils import is_experimental_mode

from .test_utils import (
    CACHEABLE_MESSAGES,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    REASONING_MODEL,
    REASONING_PROMPT,
    USER_ONLY_PROMPT,
)


def test_chat_completion(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_chat_completion_metrics.yaml"):
        openai_client.chat.completions.create(
            messages=USER_ONLY_PROMPT, model=DEFAULT_MODEL, stream=False
        )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_chat_completion_cached_tokens(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_chat_completion_cached_tokens.yaml"):
        for _ in range(2):
            openai_client.chat.completions.create(
                messages=CACHEABLE_MESSAGES,
                model=DEFAULT_MODEL,
                max_tokens=8,
                stream=False,
            )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["cache_read.input_tokens"]


def test_chat_completion_streaming_cached_tokens(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_chat_completion_streaming_cached_tokens.yaml"):
        openai_client.chat.completions.create(
            messages=CACHEABLE_MESSAGES,
            model=DEFAULT_MODEL,
            max_tokens=8,
            stream=False,
        )
        stream = openai_client.chat.completions.create(
            messages=CACHEABLE_MESSAGES,
            model=DEFAULT_MODEL,
            max_tokens=8,
            stream=True,
            stream_options={"include_usage": True},
        )
        for _ in stream:
            pass
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    # 1408 from each call, so the streamed one has to contribute.
    assert usage["cache_read.input_tokens"] == {"unknown": 2816}


def test_chat_completion_reasoning_tokens(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_chat_completion_reasoning_tokens.yaml"):
        openai_client.chat.completions.create(
            messages=REASONING_PROMPT,
            model=REASONING_MODEL,
            stream=False,
            extra_body={
                "max_completion_tokens": 1024,
                "reasoning_effort": "medium",
            },
        )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["reasoning.output_tokens"]


def test_embeddings(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_embeddings_token_metrics.yaml"):
        openai_client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input="This is a test for embeddings token metrics",
        )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["input_tokens"]
    assert not usage["output_tokens"]


def test_responses(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    if not is_experimental_mode() or not hasattr(openai_client, "responses"):
        pytest.skip("needs the Responses API and the latest semconv")
    with vcr.use_cassette(
        "test_responses_create_aggregates_cache_tokens[content_mode0].yaml"
    ):
        openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions="You are a helpful assistant.",
            input=USER_ONLY_PROMPT[0]["content"],
        )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["input_tokens"]
    assert usage["output_tokens"]
