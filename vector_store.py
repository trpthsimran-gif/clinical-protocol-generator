"""
vector_store.py

A local semantic search layer using ChromaDB - a real, purpose-built
vector database (unlike FAISS, which is just a similarity-search
library). Chroma handles persistence, metadata storage, and querying
natively, so this module is simpler than a hand-rolled FAISS wrapper.

Matches the diagram:

    Documents -> Create Embeddings -> Vector Database (ChromaDB) ->
    Retrieve Relevant Chunks -> feed into GPT

Everything runs locally and persists to a folder on disk:
    chroma_db/   - ChromaDB's own persistent storage (created automatically)

Requires an OpenAI API key (used only for the embeddings call, which
is very cheap - a fraction of a cent per document).
"""

import os
import uuid
from typing import List, Dict, Optional

import chromadb
from openai import OpenAI

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "clinical_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"

_client: Optional[OpenAI] = None
_chroma_client = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_collection():
    """
    Get (or create) the persistent Chroma collection.
    Using cosine similarity as the distance space, so scores are easy
    to interpret the same way as before (higher = more similar).
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split long text into overlapping word-based chunks.

    Overlap matters: without it, a sentence that spans a chunk boundary
    could get cut in half and lose meaning. A small overlap (e.g. 50
    words) keeps context intact across chunk edges.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Call OpenAI's embedding model and return raw embedding vectors."""
    client = _get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def add_documents(texts: List[str], metadatas: List[Dict]) -> int:
    """
    Embed and add a batch of text chunks to the vector database.
    Each chunk needs matching metadata (e.g. {"source": "sepsis.pdf", "type": "pdf"}).
    Returns the number of chunks added.
    """
    if not texts:
        return 0

    collection = _get_collection()
    embeddings = _embed_texts(texts)
    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return len(texts)


def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic search: find the top_k most meaning-similar chunks to the query,
    regardless of exact word overlap.

    Returns a list of dicts: {"text", "score", ...original metadata}
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    top_k = min(top_k, collection.count())
    query_embedding = _embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        entry = dict(meta)
        entry["text"] = doc
        # Chroma returns cosine DISTANCE (lower = more similar);
        # convert to similarity (higher = more similar) to match the
        # scoring convention used elsewhere in this app.
        entry["score"] = 1 - distance
        output.append(entry)
    return output


def index_stats() -> Dict:
    """Quick stats for displaying in the UI (how many chunks are stored, from how many sources)."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "unique_sources": 0, "sources": []}

    all_records = collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in all_records["metadatas"]}
    return {"total_chunks": count, "unique_sources": len(sources), "sources": sorted(sources)}


def reset_index():
    """Delete the local vector database (for a 'clear knowledge base' button)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        _chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet
