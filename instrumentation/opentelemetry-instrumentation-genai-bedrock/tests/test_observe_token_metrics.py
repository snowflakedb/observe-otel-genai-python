# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference token metrics add up to the token counts on the spans, for
Bedrock responses. Specific to this fork."""

from __future__ import annotations

import io
import json

from botocore.response import StreamingBody
from botocore.stub import Stubber

from opentelemetry.test_util_genai._observe_token_metrics import (
    assert_token_metrics_match_spans,
)

MESSAGES = [{"role": "user", "content": [{"text": "Say this is a test"}]}]
INFERENCE_CONFIG = {
    "maxTokens": 10,
    "temperature": 0.8,
    "topP": 1,
    "stopSequences": ["|"],
}


def test_converse(
    bedrock_client, instrument_no_content, vcr, span_exporter, metric_reader
):
    with vcr.use_cassette("test_converse_no_content.yaml"):
        bedrock_client.converse(
            messages=MESSAGES,
            modelId="amazon.nova-micro-v1:0",
            inferenceConfig=INFERENCE_CONFIG,
        )
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_converse_stream(
    bedrock_client, instrument_no_content, vcr, span_exporter, metric_reader
):
    with vcr.use_cassette("test_converse_stream_no_content.yaml"):
        response = bedrock_client.converse_stream(
            messages=MESSAGES,
            modelId="amazon.nova-micro-v1:0",
            inferenceConfig=INFERENCE_CONFIG,
        )
        list(response["stream"])
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_invoke_model_stream(
    span_exporter, metric_reader, bedrock_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_invoke_model_stream_titan_conformance.yaml"):
        response = bedrock_client.invoke_model_with_response_stream(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps(
                {
                    "inputText": "Say this is a test",
                    "textGenerationConfig": {
                        "maxTokenCount": 10,
                        "temperature": 0.8,
                        "topP": 1.0,
                        "stopSequences": ["|"],
                    },
                }
            ),
        )
        list(response["body"])
    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["output_tokens"]


def test_invoke_model_anthropic_cache_tokens(
    bedrock_client, instrument_no_content, span_exporter, metric_reader
):
    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
    )
    response_body = json.dumps(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 4,
            },
        }
    ).encode("utf-8")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    stubber = Stubber(bedrock_client)
    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(response_body), len(response_body)
            ),
        },
    )
    with stubber:
        bedrock_client.invoke_model(modelId=model_id, body=request_body)

    usage = assert_token_metrics_match_spans(span_exporter, metric_reader)
    assert usage["cache_read.input_tokens"]
    assert usage["cache_write.input_tokens"]
