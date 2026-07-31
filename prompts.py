"""
prompts.py

Prompt engineering strategies for the clinical protocol generator.
This is the piece that maps to the resume bullet:
  "Preprocessed large datasets and applied prompt engineering
   strategies to boost LLM accuracy."

Strategies used here:
  1. Role/system priming        - fixes the model's persona & constraints
  2. Grounding with retrieved literature (RAG-lite) - reduces hallucination
  3. Structured output format   - forces a consistent protocol template
  4. Few-shot style instruction - shows the model the expected shape
  5. Explicit safety disclaimer instruction (clinical content)
"""

from typing import List, Dict

SYSTEM_PROMPT = """You are a clinical protocol drafting assistant used by \
healthcare professionals. You help draft structured clinical protocol \
DRAFTS based on current biomedical literature. You are not a substitute \
for clinical judgement, institutional guidelines, or regulatory approval.

Rules you must follow:
- Ground every recommendation in the provided literature excerpts when possible.
- If the literature does not cover something, say so explicitly rather than guessing.
- Always output in the structured section format requested.
- Always include a disclaimer that this is a draft requiring clinician review.
"""

PROTOCOL_TEMPLATE = """Return the protocol in exactly this structure:

## Protocol Title
## Purpose / Scope
## Patient Population
## Step-by-Step Procedure
## Key Precautions & Contraindications
## Supporting Evidence (cite PMIDs given below)
## Review Disclaimer
"""


def build_context_block(articles: List[Dict[str, str]], vector_chunks: List[Dict] = None) -> str:
    """Turn cleaned PubMed articles + semantically retrieved chunks into a grounding context block."""
    blocks = []

    for art in articles or []:
        blocks.append(
            f"[PMID: {art['pmid']}] {art['title']}\n{art['abstract']}"
        )

    for chunk in vector_chunks or []:
        source = chunk.get("source", "unknown source")
        blocks.append(f"[From: {source}]\n{chunk.get('text', '')}")

    if not blocks:
        return "No literature was retrieved for this topic. Note this limitation clearly."

    return "\n\n".join(blocks)


def build_user_prompt(
    topic: str,
    articles: List[Dict[str, str]],
    vector_chunks: List[Dict] = None,
    extra_notes: str = "",
) -> str:
    """
    Assemble the final user-turn prompt combining:
      - the clinical topic
      - retrieved & cleaned PubMed literature (grounding)
      - semantically retrieved chunks from the vector database, if any
        (uploaded PDFs + previously cached PubMed abstracts)
      - the required output structure
      - any clinician-provided refinement notes
    """
    context_block = build_context_block(articles, vector_chunks)

    prompt = f"""Draft a clinical protocol for the following topic:

TOPIC: {topic}

RELEVANT LITERATURE (from PubMed and any uploaded reference documents):
{context_block}

{PROTOCOL_TEMPLATE}
"""
    if extra_notes:
        prompt += f"\nAdditional instructions from the reviewing clinician:\n{extra_notes}\n"

    return prompt
