"""Offline tests: an httpx mock transport replaces the chat endpoint; retrieval runs for real on sample_docs.

The hybrid test downloads the BAAI/bge-small-en-v1.5 ONNX model, so it only runs with RAG_TEST_HYBRID=1.
"""
import json
import os
import sys

import httpx
import pytest
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.llms.openai_like import OpenAILike

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import rag  # noqa: E402

DOCS = os.path.join(ROOT, "sample_docs")


@pytest.fixture
def endpoint():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        if request.url.path.endswith("/completions") and "messages" not in body:
            return httpx.Response(200, json={"id": "t", "object": "text_completion", "created": 0, "model": body["model"],
                                             "choices": [{"index": 0, "text": "legacy", "finish_reason": "stop", "logprobs": None}]})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Within 30 days of delivery."}}],
            "usage": {"prompt_tokens": 600, "completion_tokens": 8, "total_tokens": 608}})

    return seen, httpx.Client(transport=httpx.MockTransport(handler))


def test_llm_posts_chat_completions_with_reasoning_effort(endpoint):
    seen, client = endpoint
    rag.make_llm("gpt-6-sol", api_key="test", http_client=client).complete("ping")
    url, body = seen[-1]
    assert url == "https://argolink.io/v1/chat/completions"
    assert body["reasoning_effort"] == "low" and body["model"] == "gpt-6-sol"


def test_plain_openailike_would_hit_the_legacy_completions_route(endpoint):
    seen, client = endpoint
    OpenAILike(model="gpt-6-sol", api_base="https://argolink.io/v1", api_key="test", http_client=client).complete("ping")
    assert seen[-1][0] == "https://argolink.io/v1/completions"  # why make_llm sets is_chat_model=True


def test_claude_gets_its_default_temperature(endpoint):
    seen, client = endpoint
    rag.make_llm("claude-sonnet-5", api_key="test", http_client=client).complete("ping")
    assert seen[-1][1]["temperature"] == 1.0  # OpenAILike's own default, 0.1, returns 400 on this model


def test_bm25_query_answers_from_the_returns_policy(endpoint):
    seen, client = endpoint
    engine = rag.build_engine(DOCS, rag.make_llm("gpt-6-sol", api_key="test", http_client=client), top_k=2)
    response = engine.query("How many days do I have to return an order?")
    assert response.source_nodes[0].node.metadata["file_name"] == "returns.md"
    prompt = json.dumps(seen[-1][1]["messages"])
    assert "within 30 days of delivery" in prompt and len(seen) == 1


def test_cost_uses_counted_tokens():
    counter = TokenCountingHandler()
    counter.llm_token_counts.append(type("Event", (), {"prompt_token_count": 2_400, "completion_token_count": 250,
                                                       "total_token_count": 2_650})())
    assert rag.query_cost("gpt-6-sol", counter) == pytest.approx((2_400 * 0.6 + 250 * 3.0) / 1e6)
    assert rag.query_cost("some-other-model", counter) is None


@pytest.mark.skipif(os.environ.get("RAG_TEST_HYBRID") != "1", reason="downloads a local embedding model")
def test_hybrid_retrieval_makes_one_llm_call_per_question(endpoint):
    seen, client = endpoint
    engine = rag.build_engine(DOCS, rag.make_llm("gpt-6-sol", api_key="test", http_client=client), top_k=2, hybrid=True)
    response = engine.query("How many days do I have to return an order?")
    assert response.source_nodes[0].node.metadata["file_name"] == "returns.md"
    assert len(seen) == 1  # num_queries=1: no extra query-generation calls
