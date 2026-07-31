"""
app.py

Interactive web interface for the Clinical Protocol Generator.
Maps to the resume bullet:
  "Developed an interactive web-based interface enabling clinicians
   to query, review, and refine protocols efficiently."

This version implements the full pipeline:

    Clinical Protocol PDFs
        +
    PubMed Articles
        +
    Other Medical Documents
        v
    Create Embeddings
        v
    Vector Database (FAISS)
        v
    User Question
        v
    Retrieve Relevant Chunks
        v
    OpenAI GPT
        v
    Final Answer

Run with:
    streamlit run app.py
"""

import os
import re
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from etl import run_etl, get_cached_queries
from prompts import SYSTEM_PROMPT, build_user_prompt
from pdf_ingest import ingest_pdf
from vector_store import search as vector_search, index_stats, reset_index

load_dotenv()


def linkify_pmids(text: str) -> str:
    """
    Turn any 'PMID: 12345678' or 'PMID 12345678' mentions in the AI's
    draft into clickable PubMed links.
    """
    def _replace(match):
        pmid = match.group(1)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return f"[PMID: {pmid}]({url})"

    return re.sub(r"PMID:?\s*(\d{4,9})", _replace, text)


st.set_page_config(page_title="Clinical Protocol Generator", layout="wide")

# --- OpenAI client ---
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.title("🩺 GenAI Clinical Protocol Generator")
st.caption(
    "Draft evidence-grounded clinical protocol outlines from PubMed literature "
    "and your own uploaded reference documents. Not a substitute for clinical "
    "judgement — all output requires review."
)

if not api_key:
    st.warning("No OPENAI_API_KEY found. Add it to a .env file before generating protocols.")

# --- Sidebar: settings + knowledge base (vector DB) management ---
with st.sidebar:
    st.header("Settings")
    max_results = st.slider("Number of PubMed articles to retrieve", 1, 10, 5)
    top_k_chunks = st.slider("Number of semantic chunks to retrieve", 1, 10, 5)

    st.divider()
    st.header("📁 Knowledge Base")
    st.caption(
        "Upload clinical protocol PDFs or other medical reference documents. "
        "They'll be embedded and searchable alongside PubMed results."
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("📥 Ingest uploaded documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        elif not client:
            st.error("Set OPENAI_API_KEY to create embeddings.")
        else:
            total_chunks = 0
            with st.spinner("Extracting text, chunking, and creating embeddings..."):
                for f in uploaded_files:
                    total_chunks += ingest_pdf(f, source_name=f.name)
            st.success(f"Ingested {len(uploaded_files)} file(s) as {total_chunks} chunks.")

    stats = index_stats()
    st.caption(
        f"📊 Knowledge base: {stats['total_chunks']} chunks from "
        f"{stats['unique_sources']} source(s)."
    )
    if stats["sources"]:
        with st.expander("View sources"):
            for s in stats["sources"]:
                st.text(f"• {s}")
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        reset_index()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.subheader("Previously researched topics")
    for q in get_cached_queries():
        st.text(f"• {q}")

# --- Session state ---
if "articles" not in st.session_state:
    st.session_state.articles = []
if "vector_chunks" not in st.session_state:
    st.session_state.vector_chunks = []
if "draft" not in st.session_state:
    st.session_state.draft = ""

# --- Step 1: Query input ---
topic_raw = st.text_input(
    "Clinical topic to draft a protocol for",
    placeholder="e.g. Post-operative sepsis monitoring in ICU patients",
)
# Strip stray quote characters (e.g. if pasted as "topic") — PubMed treats
# quotes as an exact-phrase search, which often returns weak/unrelated results.
topic = topic_raw.strip("\"'“”‘’").strip()

col1, col2 = st.columns([1, 1])
with col1:
    fetch_clicked = st.button("🔎 Retrieve literature (ETL)", use_container_width=True)
with col2:
    generate_clicked = st.button("✍️ Draft protocol", use_container_width=True)

# --- Step 2: ETL - retrieve & clean PubMed literature (also embeds into vector DB) ---
if fetch_clicked and topic:
    with st.spinner("Running ETL: extracting & cleaning PubMed abstracts..."):
        st.session_state.articles = run_etl(topic, max_results=max_results)
    st.success(f"Retrieved {len(st.session_state.articles)} articles.")

if st.session_state.articles:
    best_score = max(a.get("relevance_score", 1.0) for a in st.session_state.articles)
    if best_score < 0.35:
        st.warning(
            "⚠️ The retrieved articles don't look closely related to your topic. "
            "This can happen with very rare/narrow topics, or if quote marks were "
            "typed into the search box. Review the abstracts below before drafting, "
            "or try rephrasing the topic without quotes."
        )

    with st.expander("📚 Retrieved literature (from PubMed)", expanded=False):
        for art in st.session_state.articles:
            score = art.get("relevance_score")
            score_label = f" — relevance: {int(score * 100)}%" if score is not None else ""
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/"
            st.markdown(
                f"**[PMID {art['pmid']}]({pubmed_url})** {art['title']}{score_label}"
            )
            st.write(art["abstract"])
            st.divider()

# --- Step 3: Generate protocol draft ---
if generate_clicked:
    if not topic:
        st.error("Enter a clinical topic first.")
    elif not client:
        st.error("Set OPENAI_API_KEY to generate a draft.")
    else:
        if not st.session_state.articles:
            with st.spinner("No literature retrieved yet — running ETL first..."):
                st.session_state.articles = run_etl(topic, max_results=max_results)

        with st.spinner("Searching knowledge base (semantic search)..."):
            try:
                st.session_state.vector_chunks = vector_search(topic, top_k=top_k_chunks)
            except Exception as e:
                st.session_state.vector_chunks = []
                st.info(f"Semantic search skipped ({e}).")

        with st.spinner("Drafting protocol with GPT..."):
            user_prompt = build_user_prompt(
                topic, st.session_state.articles, st.session_state.vector_chunks
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            st.session_state.draft = response.choices[0].message.content

if st.session_state.vector_chunks:
    with st.expander("🧠 Semantically retrieved chunks (from vector database)", expanded=False):
        for chunk in st.session_state.vector_chunks:
            source = chunk.get("source", "unknown")
            score = chunk.get("score")
            score_label = f" — similarity: {score:.2f}" if score is not None else ""
            st.markdown(f"**{source}**{score_label}")
            st.write(chunk.get("text", "")[:800] + ("..." if len(chunk.get("text", "")) > 800 else ""))
            st.divider()

# --- Step 4: Display + refine ---
if st.session_state.draft:
    st.subheader("Draft Protocol")
    st.markdown(linkify_pmids(st.session_state.draft))

    st.subheader("🔁 Refine this draft")
    refine_notes = st.text_area(
        "Add clinician notes to refine the draft (e.g. 'add pediatric dosing', "
        "'shorten the precautions section')"
    )
    if st.button("Refine"):
        if not client:
            st.error("Set OPENAI_API_KEY to refine the draft.")
        else:
            with st.spinner("Refining protocol..."):
                refine_prompt = build_user_prompt(
                    topic,
                    st.session_state.articles,
                    st.session_state.vector_chunks,
                    extra_notes=refine_notes,
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "assistant", "content": st.session_state.draft},
                        {"role": "user", "content": refine_prompt},
                    ],
                    temperature=0.3,
                )
                st.session_state.draft = response.choices[0].message.content
            st.rerun()
