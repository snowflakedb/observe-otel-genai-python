# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks the fields LLM Explorer needs, separately from the upstream tests.

The upstream test files check that the instrumentation follows the GenAI
semantic conventions. This file checks the narrower thing our product depends
on: that the specific fields LLM Explorer reads are actually present on the
spans we emit. Keeping it in its own file means a rebase never has to
reconcile our product requirements with upstream's test expectations.

Add a test here when LLM Explorer starts depending on a field, and a
``test_observe_*`` file next to the upstream tests when we carry a fix or an
intentional difference from upstream.

Still to fill in, from the design docs — check each against the contract page
before asserting on it, these are the fields the docs call out rather than a
verified list:

- ``gen_ai.conversation.id`` on chat spans, not just on the agent span
- ``gen_ai.agent.name`` where LLM Explorer expects to read it
- cost from ``SDKResultMessage.modelUsage`` (``costUSD``)
- time to first token, from ``ttft_ms`` on the result and stream events
- the span hierarchy: one ``invoke_agent`` per turn, one ``chat`` per API call,
  ``execute_tool`` children, nested ``invoke_agent`` for subagents

Contract: LLM Explorer Critical Trace Field Structure Contract
https://app.notion.com/p/observeinc/LLM-Explorer-Critical-Trace-Field-Structure-Contract-3b9770a806fc8194a45de79ed2e66222
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="placeholder: fill in from the LLM Explorer contract page"
)
def test_llm_explorer_contract_fields() -> None:
    raise AssertionError("not implemented")
