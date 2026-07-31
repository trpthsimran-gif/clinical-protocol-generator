"""
pubmed_client.py

Thin wrapper around NCBI's free E-utilities API to search PubMed and
fetch article abstracts. No API key required for light usage.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed_ids(query: str, max_results: int = 5) -> List[str]:
    """
    Search PubMed for a query and return a list of PubMed IDs (PMIDs).
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """
    Given a list of PMIDs, fetch title + abstract text for each.
    Returns a list of dicts: {"pmid", "title", "abstract"}
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=20)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = title_el.text if title_el is not None else "No title"
        abstract = " ".join(
            part.text.strip() for part in abstract_parts if part.text
        ) or "No abstract available."

        articles.append({"pmid": pmid, "title": title, "abstract": abstract})

    return articles


def get_literature_context(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Convenience function: search + fetch in one call.
    This is the main entry point used by the ETL step.
    """
    pmids = search_pubmed_ids(query, max_results=max_results)
    return fetch_abstracts(pmids)
