# RAG with LlamaIndex on an OpenAI-compatible model API

A working retrieval-augmented generation setup for [LlamaIndex](https://github.com/run-llama/llama_index): `OpenAILike` sends the answer step to any OpenAI-compatible endpoint, retrieval runs locally (BM25, or BM25 plus FastEmbed vectors), and every query prints the tokens and dollars it used. Unofficial; not affiliated with LlamaIndex.

[20-line version](#the-20-line-version) · [OpenAILike settings](#five-openailike-settings-that-decide-whether-it-works) · [Retrieval without embeddings](#retrieval-without-an-embeddings-endpoint) · [Which model](#choosing-the-answer-model) · [FAQ](#faq)

<sub>Last checked: 2026-10-07, with llama-index-core 0.14.25, llama-index-llms-openai-like 0.8.1, llama-index-retrievers-bm25 0.8.0, llama-index-embeddings-fastembed 0.7.0 and fastembed 0.8.1.</sub>

## The 20-line version

```python
import os
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever

Settings.llm = OpenAILike(
    model="gpt-6-sol",
    api_base="https://argolink.io/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    is_chat_model=True,        # default False posts to the legacy /v1/completions
    context_window=128_000,    # default 3,900 makes LlamaIndex trim the retrieved context
    reasoning_effort="low",
)
documents = SimpleDirectoryReader("sample_docs").load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(documents)
retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)
engine = RetrieverQueryEngine.from_args(retriever, llm=Settings.llm)
print(engine.query("How long do customers have to return an order?"))
```

No embedding model, no vector database, one chat call per question. The model in this example is GPT-6 Sol on [ArgoLink](https://argolink.io/gh-rag-model-integration), which serves GPT, Claude, DeepSeek and GLM models through one OpenAI-compatible base URL; any endpoint that speaks `/v1/chat/completions` works the same way.

## Run the full version

```bash
git clone https://github.com/augustinepemberton9/rag-model-integration
cd rag-model-integration
pip install -r requirements.txt                 # Python 3.10+
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://argolink.io/v1   # the default

python rag.py "How many days do I have to return an order?"
python rag.py --model gpt-5.6-luna --top-k 6 "What does the warranty cover?"
python rag.py --docs ./my_docs --hybrid "Which payment methods are accepted?"
```

`rag.py` prints the answer, the retrieved files with their scores, and the prompt and completion tokens with their dollar cost. The token counts come from `TokenCountingHandler`, which reads the `usage` the API returns. For the first question on `sample_docs` with `--top-k 2`, BM25 ranks `returns.md` first (score 0.83) and `warranty.md` second (0.48).

`sample_docs/` holds four short policy pages for a fictional store; point `--docs` at your own folder. Markdown and plain text load out of the box; PDFs need `pip install llama-index-readers-file` (it brings `pypdf`).

## Five OpenAILike settings that decide whether it works

| Setting | Default | Set it to | What goes wrong otherwise |
|---|---|---|---|
| `is_chat_model` | `False` | `True` | requests go to `/v1/completions`, a legacy route chat endpoints usually don't serve |
| `context_window` | `3900` | the budget you want per query, e.g. `128_000` | LlamaIndex trims or splits retrieved context to fit 3,900 tokens |
| `temperature` | `0.1` | `1.0` for `claude-sonnet-5` | Claude Sonnet 5 returns 400 for any non-default `temperature`, `top_p` or `top_k` |
| `reasoning_effort` | not sent | the model's documented value, e.g. `"low"` for `gpt-6-sol`, `"none"` for `deepseek-v4.1-flash` | the model thinks at its default depth and bills those tokens as output |
| `max_retries` | `3` | `0`–`2` | a failing call is retried three times before you see the error |

Keep `context_window` under 272,000 on GPT models: a prompt above 272K tokens bills the whole request at 2× input and 1.5× output.

Two behaviours we saw while pinning the requests in the tests: LlamaIndex sends `temperature: 1.0` and a `developer`-role system prompt for model names it knows as OpenAI reasoning models (`gpt-6-sol`, `gpt-5.4`, `gpt-5.6-*` are on its list), and plain `system` for everything else.

## Retrieval without an embeddings endpoint

The model catalog behind the default base URL had no embedding models on 2026-10-07, so a vector index cannot call it for embeddings. Retrieval runs on your side:

| Retrieval | Install | Model download | Best for |
|---|---|---|---|
| BM25 (default) | `llama-index-retrievers-bm25` | none | exact terms, product names, error codes, small to medium corpora |
| BM25 + FastEmbed vectors (`--hybrid`) | `llama-index-embeddings-fastembed fastembed` | `BAAI/bge-small-en-v1.5` (ONNX, runs on CPU) | paraphrased questions, mixed vocabulary |
| Vectors from another provider | that provider's LlamaIndex embedding package | none locally | large corpora you already embed elsewhere |

The hybrid mode fuses both rankings with reciprocal rank fusion:

```python
retriever = QueryFusionRetriever(
    [bm25, vectors.as_retriever(similarity_top_k=4)],
    llm=llm,
    similarity_top_k=4,
    num_queries=1,              # default 4: three extra LLM calls per question to rewrite the query
    mode="reciprocal_rerank",
    use_async=False,
)
```

Two install traps: `llama-index-embeddings-fastembed` does not install `fastembed` itself, and `QueryFusionRetriever` quietly calls the LLM to generate extra queries unless `num_queries=1`. The hybrid test checks that a question costs exactly one chat call.

## Choosing the answer model

A retrieval query here is about 2,400 input tokens (4 chunks of up to 512 tokens plus the prompt template) and ~250 output tokens. Per 1,000 queries:

| Model | Input / output per 1M | Official input / output | 1,000 queries | Notes |
|---|---:|---:|---:|---|
| `gpt-5.6-luna` | $0.06 / $0.36 | $0.20 / $1.20 | $0.23 | cheapest; fine for direct lookups |
| `glm-5.3-flash` | $0.12 / $0.40 | $0.15 / $0.50 | $0.39 | always thinks; thinking bills as output |
| `deepseek-v4.1-flash` | $0.12 / $0.48 | $0.15 / $0.60 | $0.41 off-peak, $0.82 peak | 2× on weekdays 09:00–12:00 and 14:00–18:00 Beijing time |
| [`gpt-6-sol`](https://argolink.io/gh-rag-model-integration) | $0.60 / $3.00 | $2 / $10 | $2.19 | stronger synthesis across several chunks |
| `claude-sonnet-5` | $0.80 / $4.00 | $2 / $10 | $2.92 | send `temperature: 1.0` or none |

<sub>Rates from the public catalog (`GET https://argolink.io/api/catalog/v1/models?page_size=100`) and the model pages, checked 2026-10-07; official columns are the vendors' list prices as the catalog records them. Reasoning tokens are billed as output and come on top of the 250 assumed here. Re-check before you budget.</sub>

## FAQ

**Can I use `llama_index.llms.openai.OpenAI` instead of `OpenAILike`?**
Only for model names it already knows: `OpenAI(model="deepseek-v4.1-flash", ...)` raises `ValueError: Unknown model`. `OpenAILike` accepts any model ID.

**Why does the very first query fail?**
Most often `is_chat_model` was left at `False`, which sends `POST /v1/completions` instead of `/v1/chat/completions`. The test `test_plain_openailike_would_hit_the_legacy_completions_route` shows the request.

**Why are answers missing facts that are in my documents?**
Check retrieval before the model: print `response.source_nodes`. If the right chunk is not there, raise `--top-k`, lower `chunk_size`, or switch on `--hybrid` for paraphrased questions. If it is there but the answer still misses it, check `context_window`.

**Does the hybrid mode send my documents anywhere?**
No. FastEmbed computes the vectors on your CPU; only the retrieved chunks and the question go to the chat endpoint.

**How do I persist the index?**
`BM25Retriever` has `persist(path)` and `BM25Retriever.from_persist_dir(path)`. For the vector part use `vectors.storage_context.persist("./storage")` and `load_index_from_storage` from `llama_index.core`.

## Run the tests

```bash
pip install -r requirements.txt pytest
pytest tests/                       # RAG_TEST_HYBRID=1 also runs the FastEmbed test (downloads the model)
```

An `httpx` mock transport replaces the chat endpoint, so the tests need no key. Retrieval runs for real on `sample_docs`.

## License

MIT
