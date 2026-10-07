"""Answer questions over a folder of documents with LlamaIndex and an OpenAI-compatible chat model.

The endpoint serves chat models but no embedding models, so retrieval runs on
your machine: BM25 keyword search by default (no model download), or BM25 plus
local FastEmbed vectors with --hybrid.

    pip install -r requirements.txt
    export OPENAI_API_KEY=...                      # key for the endpoint
    export OPENAI_BASE_URL=https://argolink.io/v1  # optional, this is the default
    python rag.py "How long do I have to return an order?"
    python rag.py --model gpt-5.6-luna --docs ./my_docs "What does the warranty cover?"
"""
from __future__ import annotations

import argparse
import os

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever

DEFAULT_BASE_URL = "https://argolink.io/v1"

# USD per 1M input / output tokens on the default endpoint, checked 2026-10-07.
PRICES = {
    "gpt-6-sol": (0.60, 3.00),
    "gpt-5.6-luna": (0.06, 0.36),
    "deepseek-v4.1-flash": (0.12, 0.48),
    "glm-5.3-flash": (0.12, 0.40),
    "claude-sonnet-5": (0.80, 4.00),
}

# Per-model request settings. temperature=1.0 is Claude Sonnet 5's default; it rejects
# any other value, and OpenAILike would otherwise send 0.1.
MODEL_SETTINGS = {
    "gpt-6-sol": {"reasoning_effort": "low"},
    "gpt-5.6-luna": {"reasoning_effort": "low"},
    "deepseek-v4.1-flash": {"reasoning_effort": "none"},
    "glm-5.3-flash": {"reasoning_effort": "high"},
    "claude-sonnet-5": {"temperature": 1.0},
}


def make_llm(model: str = "gpt-6-sol", **overrides) -> OpenAILike:
    settings = {**MODEL_SETTINGS.get(model, {}), **overrides}
    return OpenAILike(
        model=model,
        api_base=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=settings.pop("api_key", None) or os.environ.get("OPENAI_API_KEY"),
        is_chat_model=True,       # the default (False) posts to the legacy /v1/completions
        context_window=128_000,   # the default 3,900 makes LlamaIndex trim retrieved context;
                                  # stay under 272K on GPT models to avoid the long-context price step
        max_tokens=1024,
        max_retries=2,
        **settings,
    )


def build_engine(docs_dir: str, llm: OpenAILike, top_k: int = 4, hybrid: bool = False) -> RetrieverQueryEngine:
    documents = SimpleDirectoryReader(docs_dir).load_data()
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(documents)
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    if hybrid:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        vectors = VectorStoreIndex(nodes, embed_model=FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5"))
        retriever = QueryFusionRetriever(
            [retriever, vectors.as_retriever(similarity_top_k=top_k)],
            llm=llm,
            similarity_top_k=top_k,
            num_queries=1,            # the default 4 asks the LLM for 3 extra queries on every question
            mode="reciprocal_rerank",
            use_async=False,
        )
    return RetrieverQueryEngine.from_args(retriever, llm=llm)


def query_cost(model: str, counter: TokenCountingHandler) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (counter.prompt_llm_token_count * price_in + counter.completion_llm_token_count * price_out) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--docs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs"))
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hybrid", action="store_true", help="BM25 + local FastEmbed vectors")
    args = parser.parse_args()

    counter = TokenCountingHandler()  # uses the usage the API returns when present
    Settings.callback_manager = CallbackManager([counter])
    llm = make_llm(args.model)
    Settings.llm = llm                # keep every LlamaIndex component on this model
    engine = build_engine(args.docs, llm, top_k=args.top_k, hybrid=args.hybrid)

    response = engine.query(args.question)
    print(response.response, "\n")
    for source in response.source_nodes:
        print(f"  [{source.score:.2f}] {source.node.metadata.get('file_name')}")
    cost = query_cost(args.model, counter)
    tokens = f"{counter.prompt_llm_token_count} in / {counter.completion_llm_token_count} out"
    print(f"\n{tokens}" + (f", ${cost:.6f}" if cost is not None else ""))


if __name__ == "__main__":
    main()
