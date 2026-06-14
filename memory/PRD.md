# Enterprise Knowledge Assistant — PRD

## Original problem statement

User had built an Enterprise Knowledge Assistant (RAG) at
https://github.com/OBRIEN1000/Enterprise-Knowledge-Assistant-using-GPT-RAG
that did not work well on share.streamlit.io. They asked for a complete
rebuild of the codebase, free of "vibe-coded" markers, suitable for use
as a portfolio piece. They provided their own Gemini and OpenRouter API
keys and asked that the Gemini provider be prioritised. They explicitly
asked not to push the rebuild to GitHub.

## Why the previous version failed on Streamlit Cloud

- `sentence-transformers` (~500 MB) blew past the free-tier build limit
  and made cold-starts unusable.
- ChromaDB requires a recent SQLite, which the Streamlit Cloud base
  image does not ship — a common gotcha causing import errors.
- The previous code mixed Gemini API and OpenRouter without a robust
  fallback, so any provider hiccup broke the chat.

## Architecture of the rebuild

- **Streamlit** (single `app.py`) with custom CSS for a dark, restrained
  enterprise look.
- **Google Gen AI SDK** (`google-genai`) for both embeddings
  (`gemini-embedding-001`, 768-dim) and answer generation
  (`gemini-2.5-flash`, streaming).
- **OpenRouter** via the OpenAI-compatible SDK as automatic fallback.
- **Numpy** in-memory cosine similarity index — no native deps, no SQLite.
- **PyPDF** + **python-docx** + plain text loaders.

## What's implemented (2026-01)

- Multi-format ingestion: PDF, DOCX, TXT, Markdown
- Paragraph-aware chunking with overlap
- Gemini batch embeddings (output_dimensionality=768, L2 normalised)
- In-memory cosine search with per-session isolation
- Streamed Gemini answers + automatic OpenRouter fallback
- Citations rendered as `[n]` markers plus a Sources accordion
- Demo corpus loader (2 sample Acme Corp policy excerpts)
- KPI strip (documents, chunks, conversation turns)
- Robust secrets/env loader that works both locally and on Streamlit Cloud
- Theme via `.streamlit/config.toml`
- Verified end-to-end against the user's live Gemini key

## Files of interest

- `/app/enterprise_knowledge_assistant/app.py` — Streamlit entry point
- `/app/enterprise_knowledge_assistant/src/config.py` — env/secrets loader
- `/app/enterprise_knowledge_assistant/src/document_loader.py` — file parsing
- `/app/enterprise_knowledge_assistant/src/chunker.py` — splitting logic
- `/app/enterprise_knowledge_assistant/src/embeddings.py` — Gemini embed client
- `/app/enterprise_knowledge_assistant/src/vector_store.py` — numpy index
- `/app/enterprise_knowledge_assistant/src/llm_client.py` — Gemini + OpenRouter
- `/app/enterprise_knowledge_assistant/src/rag_pipeline.py` — orchestration
- `/app/enterprise_knowledge_assistant/sample_docs/` — demo corpus
- `/app/enterprise_knowledge_assistant/README.md` — portfolio-ready readme

## Backlog

- P1: Persist the vector index across sessions (e.g. on-disk pickle or
  SQLite) so users do not have to re-upload between visits.
- P1: OCR fallback for scanned PDFs (e.g. via `pytesseract`).
- P2: Per-document filtering in the chat (scope a question to a subset).
- P2: Conversation memory in the prompt (currently each turn is
  independent for predictability).
- P2: Token / cost meter in the sidebar.

## Next action items

1. User pushes the contents of `/app/enterprise_knowledge_assistant/` to
   a fresh GitHub repository.
2. On https://share.streamlit.io: connect the repo, set the secrets
   listed in the README, deploy.
3. Optional: add a screenshot to the README once deployed.
