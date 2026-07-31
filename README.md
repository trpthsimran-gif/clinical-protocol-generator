# GenAI Clinical Protocol Generator

An AI-powered tool that drafts evidence-grounded clinical protocol outlines
by retrieving real biomedical literature from PubMed and user-uploaded
reference documents, then generating structured drafts using GPT — all
through a simple, interactive web interface.

This is a Retrieval-Augmented Generation (RAG) system: instead of relying
on the AI's memory alone, it retrieves real, relevant documents first and
grounds every response in that retrieved evidence — reducing hallucination
and keeping every claim traceable back to a source.

## Architecture (full RAG pipeline)

```
Clinical Protocol PDFs  ─┐
PubMed Articles          ├─► Create Embeddings ─► Vector Database (ChromaDB)
Other Medical Documents ─┘                              │
                                                          ▼
                          User Question ─► Retrieve Relevant Chunks
                                                          │
                                                          ▼
                                                     OpenAI GPT
                                                          │
                                                          ▼
                                                    Final Answer ─► Export as PDF or Word (.docx)
```

### How the pieces fit together

- **PubMed articles** are retrieved via keyword search (`pubmed_client.py` + `etl.py`), then
  automatically embedded into the vector database too, so future searches can find them by meaning.
- **Uploaded PDFs** (clinical protocols, other medical documents) are extracted, chunked, and
  embedded via `pdf_ingest.py`.
- **Vector Database**: `vector_store.py` uses ChromaDB, a real persistent vector database, to store embeddings and run
  semantic similarity search — this finds relevant text by *meaning*, not just exact keyword
  overlap. E.g. searching "heart attack" can match a stored chunk about "myocardial infarction."
- Both the PubMed abstracts AND the semantically retrieved chunks are combined into the final
  grounded prompt sent to GPT (`prompts.py`).

## Project structure

```
clinical-protocol-generator/
├── app.py              # Streamlit UI (entry point)
├── pubmed_client.py    # Extract: search + fetch PubMed abstracts
├── etl.py              # Transform + Load: cleaning + SQLite cache + vector embedding
├── pdf_ingest.py        # Extract text from uploaded PDFs for the vector database
├── vector_store.py      # ChromaDB vector database: chunking, embeddings, semantic search
├── pdf_export.py         # Converts the final draft into a downloadable PDF
├── docx_export.py        # Converts the final draft into a downloadable Word (.docx) doc
├── prompts.py           # Prompt engineering (system prompt, templates)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install Python 3.10+** if you don't already have it.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI API key**:
   - Copy `.env.example` to `.env`
   - Get a key from https://platform.openai.com/api-keys
   - Paste it into `.env` as `OPENAI_API_KEY=sk-...`

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   This opens a browser tab at `http://localhost:8501`.

## How it works (walkthrough for interviews)

1. **(Optional) Upload reference documents** — in the sidebar, upload clinical
   protocol PDFs or other medical documents. Click "Ingest uploaded documents"
   to extract their text, split it into chunks, embed each chunk, and store
   it in the local ChromaDB vector database.
2. **You type a clinical topic** (e.g. "post-operative sepsis monitoring").
3. **Click "Retrieve literature (ETL)"** — this runs the ETL pipeline:
   - *Extract*: searches PubMed and fetches the top N abstracts.
   - *Transform*: strips whitespace, truncates overly long text.
   - *Load*: caches results into a local SQLite file (`protocol_cache.db`)
     so repeat queries don't re-hit the PubMed API, AND embeds each abstract
     into the vector database so it's semantically searchable later too.
4. **Click "Draft protocol"** — this:
   - Runs a semantic search over the vector database (PubMed abstracts +
     any uploaded PDFs) to find the most meaning-relevant chunks for your topic.
   - Builds a prompt that grounds GPT in both the fresh PubMed abstracts
     and these semantically retrieved chunks.
   - Forces a consistent section-by-section protocol structure.
   - Sends it to GPT and displays the structured draft.
5. **Refine** — type notes ("add pediatric dosing", "shorten this section")
   and the app re-prompts GPT with the previous draft plus your notes.
6. **Export** — click **"📄 Export as PDF"** for a clean, read-only PDF, or
   **"📝 Export as Word (.docx)"** for an editable document (e.g. to apply
   your institution's letterhead or track changes). Both include a review
   disclaimer footer on every export.

## Why two kinds of search? (good interview talking point)

- **PubMed's own search** (used for the "Retrieve literature" button) is
  *keyword-based* — it matches literal words in your query against its index.
- **The vector database (ChromaDB)** is *semantic* — it matches by meaning,
  using embeddings. So it can find a relevant chunk even if it uses
  different terminology than your search (e.g. "heart attack" vs.
  "myocardial infarction").
- Combining both gives broader, more relevant coverage than either alone —
  this is the standard **RAG (Retrieval-Augmented Generation)** pattern used
  in real production AI systems.

## Things you could extend later (good talking points)

- Let users manually add typed notes/guidelines directly into the knowledge
  base, as a third content source alongside PubMed and uploaded PDFs.
- Deploy `app.py` to Streamlit Community Cloud (free) for a shareable demo link.
- Add metadata filtering (e.g. search only uploaded PDFs, or only PubMed)
  using ChromaDB's built-in filtering support.

## Safety note

This tool drafts protocol **outlines** grounded in retrieved literature.
It is not validated for clinical use and every output requires review by
a qualified clinician before any real-world application.
