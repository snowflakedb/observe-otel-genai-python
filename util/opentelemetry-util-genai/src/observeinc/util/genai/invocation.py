# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Public re-export of all GenAI invocation types.

Users can import everything from this single module:

    from observeinc.util.genai.invocation import (
        Error,
        GenAIInvocation,
        FetchResponseInvocation,
        InferenceInvocation,
        EmbeddingInvocation,
        RetrievalInvocation,
        ToolInvocation,
        WorkflowInvocation,
    )
"""

from observeinc.util.genai._agent_invocation import (
    AgentInvocation,
    LocalAgentInvocation,
    RemoteAgentInvocation,
)
from observeinc.util.genai._embedding_invocation import (
    CLIENT_EMBEDDING_CONTEXT_KEY,
    EmbeddingData,
    EmbeddingInvocation,
)
from observeinc.util.genai._fetch_response_invocation import (
    CLIENT_FETCH_RESPONSE_CONTEXT_KEY,
    FetchResponseData,
    FetchResponseInvocation,
)
from observeinc.util.genai._inference_invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    InferenceData,
    InferenceInvocation,
)
from observeinc.util.genai._invocation import (
    ContextToken,
    Error,
    GenAIInvocation,
)
from observeinc.util.genai._retrieval_invocation import (
    CLIENT_RETRIEVAL_CONTEXT_KEY,
    RetrievalData,
    RetrievalInvocation,
)
from observeinc.util.genai._tool_invocation import (
    ToolInvocation,
)
from observeinc.util.genai._workflow_invocation import (
    WorkflowInvocation,
)

__all__ = [
    "CLIENT_EMBEDDING_CONTEXT_KEY",
    "CLIENT_FETCH_RESPONSE_CONTEXT_KEY",
    "CLIENT_INFERENCE_CONTEXT_KEY",
    "CLIENT_RETRIEVAL_CONTEXT_KEY",
    "AgentInvocation",
    "ContextToken",
    "EmbeddingData",
    "EmbeddingInvocation",
    "Error",
    "FetchResponseData",
    "FetchResponseInvocation",
    "GenAIInvocation",
    "InferenceData",
    "InferenceInvocation",
    "LocalAgentInvocation",
    "RemoteAgentInvocation",
    "RetrievalData",
    "RetrievalInvocation",
    "ToolInvocation",
    "WorkflowInvocation",
]
