# 📄 RAG Document Q&A

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you upload a PDF and ask questions about it. Answers are generated with **Groq's `openai/gpt-oss-120b`** model, grounded in content retrieved from your document using an open-source embedding model and a **FAISS** vector index.

---

## 1. Architecture Overview

The app is split into two layers:

- **`rag/`** — a self-contained RAG pipeline (no Streamlit imports except where caching is needed): PDF processing, chunking, embeddings, vector storage, retrieval, and LLM calls.
- **`app.py`** — the Streamlit UI, which drives the pipeline and renders results.

Everything runs **in memory for the duration of the browser session** (via `st.session_state`) — there is no database and no file persisted to disk, which matches Streamlit Community Cloud's ephemeral filesystem.

### RAG Workflow

```
PDF Upload
   │
   ▼
PDF Extraction (pypdf, per-page)              rag/pdf_processor.py
   │
   ▼
Text Cleaning & Normalization                 rag/pdf_processor.py
   │
   ▼
Chunking (sentence-aware, with overlap)       rag/text_chunker.py
   │
   ▼
Token/Length Control (~4 chars/token est.)    rag/text_chunker.py
   │
   ▼
Embeddings (all-MiniLM-L6-v2)                 rag/embeddings.py
   │
   ▼
FAISS Vector Store (IndexFlatIP, cosine)      rag/vector_store.py
   │
   ▼
[ User types a question ]
   │
   ▼
Query Embedding                               rag/embeddings.py
   │
   ▼
Similarity Search (top-K + score threshold)   rag/vector_store.py / retriever.py
   │
   ▼
Relevant Context Retrieval (with page nums)   rag/retriever.py
   │
   ▼
Prompt Construction (grounded system prompt)  rag/llm.py
   │
   ▼
Groq LLM (openai/gpt-oss-120b)                rag/llm.py
   │
   ▼
Final Answer + Source/Page References         app.py
```

### Why `all-MiniLM-L6-v2` for embeddings?

- Produces compact **384-dimensional** vectors → small FAISS index, low memory use.
- ~90MB download, runs fast on CPU → well suited to Streamlit Community Cloud's free tier (no GPU, limited RAM).
- Strong quality-for-size trade-off; a widely used default for general-purpose semantic search.

### Why estimate tokens instead of using a real tokenizer?

Chunk sizing uses a simple `len(text) / 4` heuristic instead of a model-specific tokenizer (e.g. `tiktoken`). This avoids an extra dependency and an extra network call to fetch tokenizer files at runtime, which improves reliability on Streamlit Cloud. It's an approximation, not an exact token count — `CHUNK_SIZE` and `CHUNK_OVERLAP` should be read as "approximate tokens."

---

## 2. Project Structure

```text
rag-app/
│
├── app.py                  # Streamlit UI — entry point
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── config.py                # Central config: env vars / st.secrets / defaults
│
├── rag/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF extraction + text cleaning
│   ├── text_chunker.py       # Sentence-aware chunking + length control
│   ├── embeddings.py         # Sentence-Transformers wrapper (cached)
│   ├── vector_store.py       # FAISS index wrapper
│   ├── retriever.py          # Query embedding + similarity search
│   ├── llm.py                # Groq client + grounded prompt construction
│   └── pipeline.py           # Orchestrates the full RAG workflow
│
└── utils/
    ├── __init__.py
    └── helpers.py            # UI-facing formatting/validation helpers
```

---

## 3. Requirements

See [`requirements.txt`](./requirements.txt). Key packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `groq` | LLM inference client |
| `faiss-cpu` | Open-source vector database |
| `sentence-transformers` | Open-source embedding model |
| `pypdf` | PDF text extraction |
| `numpy` | Vector math |
| `python-dotenv` | Load `.env` for local development |

All imports in the codebase have been verified against this list.

---

## 4. Environment & API Key Configuration

**Never commit your API key.** The app reads `GROQ_API_KEY` from, in order of precedence:

1. Streamlit Secrets (`st.secrets`) — used automatically on Streamlit Community Cloud.
2. Environment variables — used for local development via a `.env` file.

### Local development

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set your real key:
   ```
   GROQ_API_KEY=gsk_your_real_key_here
   ```
3. Get a key at <https://console.groq.com/keys>.

### Streamlit Community Cloud

Go to **your app → Settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```

---

## 5. Local Installation & Testing

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Groq API key
cp .env.example .env
# then edit .env and paste in your real GROQ_API_KEY

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Test PDF upload & processing:**
1. In the sidebar, upload any text-based PDF.
2. Click **Process document**.
3. Confirm the status panel shows "Done! Indexed N pages into M chunks."

**Test document retrieval & question answering:**
1. Type a question you know the document answers, e.g. *"What is this document about?"*
2. Confirm an answer appears with a **Sources** expander showing page references.
3. Ask something clearly **not** in the document (e.g. *"What is the capital of Mars?"*) and confirm the app responds that it couldn't find the information, rather than making something up.

---

## 6. GitHub Upload Instructions

```bash
# 1. Create a new repository on GitHub (via the website): 
#    github.com -> New repository -> name it e.g. "rag-app" -> Create

# 2. Initialize Git locally, inside the rag-app/ folder
git init

# 3. Stage all project files
git add .

# 4. Commit
git commit -m "Initial commit: RAG PDF Q&A app"

# 5. Connect your local repo to GitHub (replace with your repo URL)
git remote add origin https://github.com/<your-username>/rag-app.git

# 6. Push
git branch -M main
git push -u origin main
```

**Verify secrets are not committed:**
```bash
git ls-files | grep -E "\.env$"
```
This should print **nothing**. If it prints `.env`, run `git rm --cached .env` and confirm `.env` is listed in `.gitignore` before committing again.

---

## 7. Streamlit Community Cloud Deployment

1. Go to <https://share.streamlit.io/> and sign in with GitHub.
2. Click **New app** → select your `rag-app` repository.
3. Choose the branch you pushed (e.g. `main`).
4. Set **Main file path** to `app.py`.
5. Before deploying, go to **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
6. Click **Deploy**.
7. Once deployed, open the app URL, upload a test PDF, process it, and ask a question to confirm everything works end-to-end in the cloud.

### Troubleshooting common deployment errors

| Symptom | Likely cause / fix |
|---|---|
| "No Groq API key configured" | Add `GROQ_API_KEY` under **Settings → Secrets**, then reboot the app. |
| App crashes on first load with a memory error | The free tier has limited RAM; avoid uploading very large PDFs, and consider redeploying after a reboot. |
| `ModuleNotFoundError` for any package | Confirm `requirements.txt` was committed and pushed; check the deployment logs for the exact missing package. |
| Very slow first response after deploy/reboot | Expected — the embedding model is downloaded and loaded once per app restart; subsequent requests are fast due to `st.cache_resource`. |
| "No readable text could be extracted" | The PDF is likely scanned/image-only. Run OCR on it first (e.g. with a tool like `ocrmypdf`) and re-upload. |
| Groq "authentication failed" | Double check the key was pasted correctly into Secrets, with no extra quotes or whitespace. |
| App works locally but not on Cloud | Make sure `.env` values you rely on locally are also added to Streamlit Secrets — `.env` is never uploaded (it's git-ignored). |

---

## 8. Important Limitations & Possible Future Improvements

### Limitations

- **No persistent storage.** The FAISS index and processed chunks live only in `st.session_state` for the current browser session. Refreshing the page, restarting the app, or Streamlit Cloud recycling the container will clear the index — the user must re-upload and re-process their PDF.
- **Token estimates are approximate**, not exact tokenizer counts.
- **Single document at a time.** Uploading a new PDF replaces the previous index rather than merging documents.
- **English-oriented sentence splitting** — the regex-based sentence splitter is tuned for English punctuation conventions.

### Possible future improvements

- Persist the FAISS index and chunk metadata to an external store (e.g. a database, cloud object storage, or a persistent vector DB) so documents survive restarts.
- Support multiple simultaneously loaded documents with per-document filtering.
- Swap the fixed token-estimate heuristic for a proper tokenizer count for more precise chunk sizing.
- Add hybrid search (keyword + vector) for improved retrieval on documents with rare terms/acronyms.
- Add streaming responses from Groq for a more responsive chat experience.
- Add support for multi-file batch uploads and cross-document Q&A.
