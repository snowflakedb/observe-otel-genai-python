# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from observeinc.util.genai.observe_provider import provider_from_host


@pytest.mark.parametrize(
    "host, expected",
    [
        ("myorg-myaccount.snowflakecomputing.com", "snowflake"),
        ("myaccount.privatelink.snowflakecomputing.com", "snowflake"),
        ("my-resource.openai.azure.com", "azure.ai.openai"),
        ("my-resource.services.ai.azure.com", "azure.ai.inference"),
        ("models.inference.ai.azure.com", "azure.ai.inference"),
        ("bedrock-runtime.us-west-2.amazonaws.com", "aws.bedrock"),
        ("api.groq.com", "groq"),
        ("api.deepseek.com", "deepseek"),
        ("api.x.ai", "x_ai"),
        ("api.mistral.ai", "mistral_ai"),
        ("generativelanguage.googleapis.com", "gcp.gemini"),
        ("api.openai.com", None),
        ("localhost", None),
        ("snowflakecomputing.com.example.com", None),
        ("notsnowflakecomputing.com", None),
        (None, None),
    ],
)
def test_provider_from_host(host, expected):
    assert provider_from_host(host) == expected
