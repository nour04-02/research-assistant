"""Find and download papers directly from arXiv (free API, no key needed)."""
from typing import Dict, List

import arxiv
import requests


def search_arxiv(query: str, max_results: int = 8) -> List[Dict]:
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for r in client.results(search):
        short_id = r.get_short_id()
        results.append({
            "id": short_id,
            "title": " ".join(r.title.split()),
            "authors": [a.name for a in r.authors],
            "year": str(r.published.year),
            "summary": " ".join(r.summary.split()),
            "pdf_url": r.pdf_url or f"https://arxiv.org/pdf/{short_id}",
            "url": r.entry_id,
            "categories": r.categories,
        })
    return results


def download_pdf(pdf_url: str) -> bytes:
    res = requests.get(pdf_url, timeout=90, headers={"User-Agent": "ai-research-assistant/1.0"})
    res.raise_for_status()
    return res.content
