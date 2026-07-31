"""
etl.py

Simple ETL pipeline replacing the Cloud Run / Cloud Functions version
from the original project. Same three stages, just running locally:

  Extract  -> pull abstracts from PubMed (pubmed_client.py)
  Transform-> clean/normalize text (strip whitespace, dedupe, truncate)
  Load     -> cache results in a local SQLite DB so repeat queries
              don't re-hit the PubMed API

This keeps the "ETL workflow" concept from the resume bullet, but
swaps managed cloud services for a single local file, so it's easy
to run and easy to explain in an interview.
"""

import sqlite3
import re
from datetime import datetime
from typing import List, Dict

from pubmed_client import get_literature_context
from vector_store import chunk_text, add_documents

DB_PATH = "protocol_cache.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS literature_cache (
            pmid TEXT PRIMARY KEY,
            query TEXT,
            title TEXT,
            abstract TEXT,
            fetched_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def _clean_text(text: str) -> str:
    """Transform step: basic cleaning of PubMed text."""
    text = re.sub(r"\s+", " ", text).strip()
    # Strip stray quote characters a user might paste in (e.g. "topic")
    text = text.strip("\"'“”‘’")
    # Truncate very long abstracts to keep prompt size manageable
    return text[:1500]


# Common words that don't help judge topical relevance
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "is", "are", "was", "were", "by", "at", "as", "be", "this", "that",
    "treatment", "management", "monitoring", "protocol", "patients",
    "patient", "adult", "adults", "care",
}


def _keywords(text: str) -> set:
    """Extract meaningful lowercase keywords from a string (transform helper)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _relevance_score(query: str, title: str, abstract: str) -> float:
    """
    Rough relevance check: what fraction of the query's meaningful keywords
    actually show up in the article's title+abstract.

    This doesn't replace PubMed's own ranking, but catches cases where a
    narrow/rare query falls back to loosely-matched, unrelated results
    (e.g. a query in quotes not matching any exact phrase).
    """
    query_kw = _keywords(query)
    if not query_kw:
        return 1.0  # nothing meaningful to check against

    haystack_kw = _keywords(f"{title} {abstract}")
    overlap = query_kw & haystack_kw
    return len(overlap) / len(query_kw)


def run_etl(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Full ETL run for a given clinical topic/query.
    Returns cleaned articles (each with a 'relevance_score' 0-1),
    sorted most-relevant first, ready to feed into prompt construction.
    """
    # Clean the query itself too (strips stray quotes users may paste in)
    query = query.strip("\"'“”‘’").strip()

    conn = _init_db()

    # EXTRACT
    raw_articles = get_literature_context(query, max_results=max_results)

    cleaned_articles = []
    for art in raw_articles:
        # TRANSFORM
        title = _clean_text(art["title"])
        abstract = _clean_text(art["abstract"])
        cleaned = {
            "pmid": art["pmid"],
            "title": title,
            "abstract": abstract,
            "relevance_score": round(_relevance_score(query, title, abstract), 2),
        }
        cleaned_articles.append(cleaned)

        # LOAD
        conn.execute(
            """
            INSERT OR REPLACE INTO literature_cache
            (pmid, query, title, abstract, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cleaned["pmid"],
                query,
                cleaned["title"],
                cleaned["abstract"],
                datetime.utcnow().isoformat(),
            ),
        )

    conn.commit()
    conn.close()

    # Also embed these abstracts into the vector database, so future semantic
    # searches (across ALL ingested sources - PubMed + uploaded PDFs) can find
    # them by meaning, not just by exact PubMed keyword match.
    if cleaned_articles:
        texts = [f"{a['title']}\n{a['abstract']}" for a in cleaned_articles]
        metadatas = [
            {"source": f"PubMed PMID {a['pmid']}", "type": "pubmed", "pmid": a["pmid"]}
            for a in cleaned_articles
        ]
        try:
            add_documents(texts, metadatas)
        except Exception:
            # Embedding requires a valid OpenAI key; don't let this break
            # the core PubMed retrieval if embeddings fail for any reason.
            pass

    # Most relevant first
    cleaned_articles.sort(key=lambda a: a["relevance_score"], reverse=True)
    return cleaned_articles


def get_cached_queries() -> List[str]:
    """List distinct queries already run, for a 'history' dropdown in the UI."""
    conn = _init_db()
    rows = conn.execute("SELECT DISTINCT query FROM literature_cache").fetchall()
    conn.close()
    return [r[0] for r in rows]
