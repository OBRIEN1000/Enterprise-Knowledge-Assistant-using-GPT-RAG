# Enterprise Knowledge Assistant

A retrieval-augmented generation (RAG) application that lets teams query
their internal documentation in natural language and receive answers
grounded in the exact passages of the source material.

The project is intentionally minimal — a single Streamlit process backed
by Google's Gemini API for both retrieval embeddings and answer
generation, with OpenRouter as an automatic fallback. There is no
database to provision and no native dependency to compile, so the app
deploys cleanly on Streamlit Community Cloud.

## Features

- Multi-format ingestion: PDF, DOCX, TXT, Markdown
- Paragraph-aware chunking with configurable size and overlap
- 768-dimensional embeddings via `gemini-embedding-001`
- In-memory cosine similarity search (no external vector DB)
- Streamed answers from `gemini-2.5-flash`
- Automatic fallback to OpenRouter if the primary provider fails
- Citations: every answer references the originating document and page

## Architecture

```
            ┌────────────────────────────┐
            │      Streamlit UI          │
            └─────────────┬──────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │            RAG pipeline            │
        │                                    │
        │  Documents → chunks → embeddings   │
        │             │                      │
        │             ▼                      │
        │      In-memory index               │
        │             │                      │
        │             ▼                      │
        │  Query → top-K context → LLM       │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │  Gemini (primary) · OpenRouter     │
        └────────────────────────────────────┘
```

## Project layout

```
.
├── app.py                  Streamlit entry point
├── requirements.txt        Pinned Python dependencies
├── .streamlit/
│   └── config.toml         Theme & server configuration
├── sample_docs/            Optional demo corpus
└── src/
    ├── config.py           Environment / secrets loader
    ├── document_loader.py  PDF · DOCX · TXT · MD extraction
    ├── chunker.py          Paragraph-aware text splitter
    ├── embeddings.py       Gemini text-embedding-004 client
    ├── vector_store.py     Numpy-backed cosine similarity index
    ├── llm_client.py       Gemini + OpenRouter streaming client
    └── rag_pipeline.py     Orchestration
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env           # then fill in your API keys
streamlit run app.py
```

Open http://localhost:8501.

### Required keys

| Variable               | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `GEMINI_API_KEY`       | Embeddings and primary LLM (required)            |
| `OPENROUTER_API_KEY`   | Fallback LLM provider (optional but recommended) |
| `LLM_PROVIDER`         | `gemini` (default) or `openrouter`               |
| `GEMINI_MODEL`         | Defaults to `gemini-2.5-flash`                   |
| `OPENROUTER_MODEL`     | Defaults to `google/gemini-2.5-flash`            |
| `EMBEDDING_MODEL`      | Defaults to `gemini-embedding-001`               |

Obtain the keys from:
- Gemini — https://aistudio.google.com/apikey
- OpenRouter — https://openrouter.ai/settings/keys

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and connect the repository.
3. In **Settings → Secrets**, paste:

   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   OPENROUTER_API_KEY = "your-openrouter-key"
   LLM_PROVIDER = "gemini"
   GEMINI_MODEL = "gemini-2.5-flash"
   OPENROUTER_MODEL = "google/gemini-2.5-flash"
   EMBEDDING_MODEL = "gemini-embedding-001"
   ```

4. Deploy. The app boots in ~60 seconds.

## How it works

1. **Loading.** Uploaded files are parsed page by page so citations can
   reference a precise location. Plain-text formats are treated as a
   single logical page.
2. **Chunking.** Text is split at paragraph boundaries and capped at
   ~900 characters with 150-character overlap so sentences are never
   cut in the middle.
3. **Embedding.** Each chunk is sent in batches to Gemini's
   `gemini-embedding-001` endpoint and projected to 768 dimensions.
   Vectors are L2-normalised so cosine similarity reduces to a dot product.
4. **Retrieval.** A user question is embedded with the same model and
   compared against every chunk vector. The top-K hits become context.
5. **Generation.** The selected context is passed to the LLM with a
   strict system prompt that asks for citations in `[n]` form.

## License

MIT.
