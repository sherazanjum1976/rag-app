"""
app.py

Streamlit frontend for the RAG PDF Question-Answering application.

Run locally with:
    streamlit run app.py
"""

import streamlit as st

import config
from rag.pipeline import PipelineError, RAGPipeline
from utils.helpers import format_sources_markdown, validate_uploaded_file

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title=config.APP_TITLE, page_icon="📄", layout="wide")


def init_session_state() -> None:
    defaults = {
        "pipeline": None,
        "processing_stats": None,
        "chat_history": [],  # list of dicts: {question, answer, sources}
        "processed_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: configuration & document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if config.is_groq_configured():
        st.success("Groq API key detected.")
    else:
        st.error(
            "No Groq API key found. Set `GROQ_API_KEY` in a local `.env` "
            "file, or in Streamlit Secrets when deployed."
        )

    with st.expander("Retrieval settings", expanded=False):
        top_k = st.slider(
            "Chunks to retrieve (TOP_K)", min_value=1, max_value=10, value=config.TOP_K
        )
        chunk_size = st.number_input(
            "Chunk size (approx. tokens)",
            min_value=100,
            max_value=2000,
            value=config.CHUNK_SIZE,
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (approx. tokens)",
            min_value=0,
            max_value=500,
            value=config.CHUNK_OVERLAP,
            step=25,
        )

    st.divider()
    st.header("📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    process_clicked = st.button(
        "Process document", type="primary", use_container_width=True
    )

    if st.session_state.processed_filename:
        st.caption(f"Currently indexed: **{st.session_state.processed_filename}**")
        if st.button("Clear document & start over", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.processing_stats = None
            st.session_state.processed_filename = None
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main area: title & description
# ---------------------------------------------------------------------------
st.title(config.APP_TITLE)
st.write(
    "Upload a PDF, and ask questions about its content. Answers are "
    "generated using **Retrieval-Augmented Generation (RAG)**: relevant "
    "excerpts are retrieved from your document with a local embedding "
    "model + FAISS, then passed to Groq's `openai/gpt-oss-120b` model to "
    "produce a grounded answer with page references."
)

# ---------------------------------------------------------------------------
# Handle document processing
# ---------------------------------------------------------------------------
if process_clicked:
    validation_error = validate_uploaded_file(uploaded_file)
    if validation_error:
        st.sidebar.error(validation_error)
    else:
        with st.status("Processing document...", expanded=True) as status:
            try:
                st.write("Loading embedding model...")
                pipeline = RAGPipeline()

                st.write("Extracting and cleaning text from PDF...")
                st.write("Chunking document and generating embeddings...")
                stats = pipeline.process_document(
                    uploaded_file,
                    filename=uploaded_file.name,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )

                st.session_state.pipeline = pipeline
                st.session_state.processing_stats = stats
                st.session_state.processed_filename = stats.filename
                st.session_state.chat_history = []

                status.update(
                    label=f"Done! Indexed {stats.num_pages} pages "
                    f"into {stats.num_chunks} chunks.",
                    state="complete",
                )
            except PipelineError as exc:
                status.update(label="Processing failed", state="error")
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                status.update(label="Unexpected error", state="error")
                st.error(f"An unexpected error occurred while processing: {exc}")

# ---------------------------------------------------------------------------
# Document status
# ---------------------------------------------------------------------------
if st.session_state.processing_stats:
    stats = st.session_state.processing_stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Document", stats.filename)
    col2.metric("Pages indexed", stats.num_pages)
    col3.metric("Chunks indexed", stats.num_chunks)
else:
    st.info("👈 Upload a PDF and click **Process document** to get started.")

st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.subheader("💬 Ask a question about the document")

# Replay chat history
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander("📎 Sources"):
            st.markdown(format_sources_markdown(turn["sources"]))

question = st.chat_input(
    "Ask a question about the uploaded document...",
    disabled=st.session_state.pipeline is None,
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not config.is_groq_configured():
            st.error(
                "Groq API key is not configured. Please set GROQ_API_KEY "
                "before asking questions."
            )
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = st.session_state.pipeline.answer_question(
                        question, groq_api_key=config.GROQ_API_KEY, top_k=int(top_k)
                    )
                    st.markdown(result.answer)
                    with st.expander("📎 Sources"):
                        st.markdown(format_sources_markdown(result.sources))

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": result.answer,
                            "sources": result.sources,
                        }
                    )
                except PipelineError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"An unexpected error occurred: {exc}")

st.divider()
st.caption(
    "Note: this app keeps the FAISS index and chat history in memory for "
    "the current session only. Re-processing a document or refreshing the "
    "app will clear them. See README.md for details."
)
