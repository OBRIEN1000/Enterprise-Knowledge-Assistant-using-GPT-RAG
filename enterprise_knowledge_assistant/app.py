"""Enterprise Knowledge Assistant — Streamlit entry point."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import streamlit as st

from src.chunker import Chunk, build_chunks
from src.config import Settings, load_settings
from src.document_loader import SUPPORTED_EXTENSIONS, load_document
from src.embeddings import EmbeddingsClient
from src.llm_client import LLMClient, LLMUnavailableError
from src.rag_pipeline import RagPipeline
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --accent: #7c5cff;
    --accent-soft: rgba(124, 92, 255, 0.16);
    --surface: #141821;
    --surface-2: #1a1f2b;
    --border: rgba(255, 255, 255, 0.06);
    --muted: #8b91a1;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

.app-hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.app-hero .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px; height: 46px;
    border-radius: 12px;
    background: linear-gradient(135deg, #7c5cff, #4aa3ff);
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
    box-shadow: 0 8px 24px rgba(124, 92, 255, 0.35);
}
.app-hero h1 {
    margin: 0;
    font-size: 1.7rem;
    letter-spacing: -0.01em;
}
.app-hero p {
    margin: 0;
    color: var(--muted);
    font-size: 0.95rem;
}

.kpi-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.kpi {
    flex: 1 1 180px;
    padding: 0.85rem 1rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
}
.kpi .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi .value { font-size: 1.3rem; font-weight: 600; margin-top: 0.2rem; }

.source-card {
    border: 1px solid var(--border);
    background: var(--surface-2);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
}
.source-card .meta {
    color: var(--accent);
    font-weight: 600;
    font-size: 0.82rem;
    margin-bottom: 0.35rem;
}
.source-card .body {
    color: #c8ccd6;
    line-height: 1.45;
    white-space: pre-wrap;
}

.chat-empty {
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: var(--muted);
}

.provider-pill {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    background: var(--accent-soft);
    color: var(--accent);
    margin-left: 0.4rem;
    font-weight: 600;
}

div[data-testid="stSidebar"] { background: #0f1219; }
div[data-testid="stSidebar"] .stFileUploader > label { font-weight: 600; }

.stChatMessage { border-radius: 12px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resource initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_embeddings_client(api_key: str, model: str) -> EmbeddingsClient:
    return EmbeddingsClient(api_key=api_key, model=model)


def build_llm_client(settings: Settings) -> LLMClient:
    return LLMClient(
        gemini_api_key=settings.gemini_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        provider=settings.llm_provider,
        gemini_model=settings.gemini_model,
        openrouter_model=settings.openrouter_model,
    )


def init_session_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = {}  # filename -> file hash


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _file_signature(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


def ingest_uploaded_files(files, settings: Settings, pipeline: RagPipeline) -> List[str]:
    added: List[str] = []
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            st.warning(f"Skipped {uploaded.name}: unsupported file type.")
            continue

        content = uploaded.getvalue()
        signature = _file_signature(content)
        if st.session_state.indexed_files.get(uploaded.name) == signature:
            continue

        # Replace the previous version of a file with the same name
        if uploaded.name in st.session_state.indexed_files:
            st.session_state.vector_store.remove_document(uploaded.name)

        pages = load_document(uploaded.name, content)
        if not pages:
            st.warning(f"Could not extract any text from {uploaded.name}.")
            continue

        chunks: List[Chunk] = build_chunks(
            uploaded.name, pages, settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:
            st.warning(f"No usable chunks produced from {uploaded.name}.")
            continue

        pipeline.index(chunks)
        st.session_state.indexed_files[uploaded.name] = signature
        added.append(uploaded.name)
    return added


def load_sample_documents(settings: Settings, pipeline: RagPipeline) -> List[str]:
    sample_dir = Path(__file__).parent / "sample_docs"
    if not sample_dir.exists():
        return []

    added: List[str] = []
    for path in sorted(sample_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name in st.session_state.indexed_files:
            continue
        content = path.read_bytes()
        pages = load_document(path.name, content)
        if not pages:
            continue
        chunks = build_chunks(path.name, pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue
        pipeline.index(chunks)
        st.session_state.indexed_files[path.name] = _file_signature(content)
        added.append(path.name)
    return added


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def render_header(settings: Settings) -> None:
    provider_label = "Gemini" if settings.llm_provider == "gemini" else "OpenRouter"
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="badge">EK</div>
            <div>
                <h1>Enterprise Knowledge Assistant</h1>
                <p>Retrieval-augmented answers over your private documents.
                <span class="provider-pill">{provider_label} · {settings.gemini_model if settings.llm_provider == "gemini" else settings.openrouter_model}</span>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis() -> None:
    store: VectorStore = st.session_state.vector_store
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi"><div class="label">Documents</div><div class="value">{len(store.documents)}</div></div>
            <div class="kpi"><div class="label">Indexed chunks</div><div class="value">{len(store)}</div></div>
            <div class="kpi"><div class="label">Conversation turns</div><div class="value">{len(st.session_state.messages)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(settings: Settings, pipeline: RagPipeline) -> None:
    with st.sidebar:
        st.markdown("### Knowledge base")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            help="PDF, DOCX, TXT or Markdown. Up to 50 MB per file.",
        )

        if uploaded_files:
            with st.spinner("Indexing documents..."):
                added = ingest_uploaded_files(uploaded_files, settings, pipeline)
            if added:
                st.success(f"Indexed: {', '.join(added)}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Load demo", use_container_width=True):
                with st.spinner("Loading demo corpus..."):
                    added = load_sample_documents(settings, pipeline)
                if added:
                    st.success(f"Loaded {len(added)} sample file(s).")
                else:
                    st.info("Demo corpus already loaded or unavailable.")
        with col_b:
            if st.button("Reset", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.indexed_files = {}
                st.session_state.messages = []
                st.rerun()

        store: VectorStore = st.session_state.vector_store
        if store.documents:
            st.markdown("#### Indexed documents")
            for document in store.documents:
                st.markdown(f"- {document}")

        st.markdown("---")
        st.markdown("### Settings")
        st.caption(f"LLM provider: **{settings.llm_provider}**")
        st.caption(f"Embedding model: **{settings.embedding_model}**")
        st.caption(f"Top-K retrieval: **{settings.top_k}**")

        if not settings.has_gemini and not settings.has_openrouter:
            st.error(
                "No API keys configured. Set `GEMINI_API_KEY` or "
                "`OPENROUTER_API_KEY` via `.env` or Streamlit secrets."
            )
        elif not settings.has_gemini:
            st.warning(
                "Gemini key missing — embeddings require it. Add `GEMINI_API_KEY`."
            )


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for position, hit in enumerate(sources, start=1):
            snippet = hit.chunk.text.strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rstrip() + "…"
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="meta">[{position}] {hit.chunk.document} — page {hit.chunk.page} · score {hit.score:.2f}</div>
                    <div class="body">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat(pipeline: RagPipeline) -> None:
    if not st.session_state.messages:
        st.markdown(
            '<div class="chat-empty">Upload a document or load the demo corpus, '
            "then ask a question to see grounded answers with citations.</div>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask anything about your documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            answer = pipeline.answer(prompt)
            collected = ""
            for token in answer.iterator:
                collected += token
                placeholder.markdown(collected + "▍")
            placeholder.markdown(collected if collected else "_No answer returned._")
            render_sources(answer.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": collected, "sources": answer.sources}
            )
        except LLMUnavailableError as exc:
            placeholder.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {exc}", "sources": []}
            )
        except Exception as exc:  # noqa: BLE001
            placeholder.error(f"Unexpected error: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {exc}", "sources": []}
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    init_session_state()

    if not settings.has_gemini:
        render_header(settings)
        st.error(
            "GEMINI_API_KEY is required for embeddings. Add it to your "
            "`.env` file locally, or to **Settings → Secrets** on Streamlit Cloud."
        )
        st.stop()

    embeddings = get_embeddings_client(settings.gemini_api_key, settings.embedding_model)
    llm = build_llm_client(settings)
    pipeline = RagPipeline(
        store=st.session_state.vector_store,
        embeddings=embeddings,
        llm=llm,
        top_k=settings.top_k,
    )

    render_header(settings)
    render_kpis()
    render_sidebar(settings, pipeline)
    render_chat(pipeline)


if __name__ == "__main__":
    main()
