"""PDF parsing, section-aware chunking and multi-paper HYBRID retrieval.

Retrieval pipeline for every query:
    1. Dense search   : FAISS + multilingual embeddings (bge-m3), so Arabic questions match English papers
    2. Sparse search  : BM25 keyword search (great for exact names like "ResNet-50" or "ImageNet")
    3. Fusion         : Reciprocal Rank Fusion (RRF) merges both ranked lists
    4. Reranking      : a multilingual cross-encoder re-scores the best candidates (optional)
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Section headings we try to recognise in papers (matched after removing numbering).
SECTION_ALIASES = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background"],
    "related_work": ["related work", "literature review", "prior work", "previous work"],
    "methodology": ["method", "methods", "methodology", "materials and methods", "proposed method",
                    "proposed methods", "approach", "proposed approach", "model", "architecture",
                    "proposed model", "system design", "framework", "proposed framework"],
    "data": ["data", "dataset", "datasets", "data collection", "study population", "participants",
             "data and preprocessing", "dataset and preprocessing"],
    "experiments": ["experiments", "experimental setup", "experimental results", "implementation details",
                    "evaluation", "experiments and results"],
    "results": ["results", "results and discussion", "findings"],
    "discussion": ["discussion"],
    "limitations": ["limitations", "limitations and future work", "limitation"],
    "conclusion": ["conclusion", "conclusions", "conclusion and future work", "future work"],
    "references": ["references", "bibliography", "acknowledgments", "acknowledgements"],
}
_ALIAS_LOOKUP = {alias: sec for sec, aliases in SECTION_ALIASES.items() for alias in aliases}
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+")
# "Abstract: text..." or "Abstract—text..." written on the same line
_ABSTRACT_INLINE = re.compile(r"^abstract\s*[:.\-–—]\s*(.*)$", re.IGNORECASE)
_TOKEN = re.compile(r"\w+", re.UNICODE)

# For each field we want to extract: which sections to look in, and a search query.
FIELD_HINTS = {
    "methodology": (["methodology", "experiments"], "proposed method approach model architecture training procedure"),
    "dataset": (["data", "experiments", "methodology"], "dataset data collection samples patients images split preprocessing"),
    "results": (["results", "experiments", "discussion"], "results accuracy performance metrics outperforms comparison table"),
    "limitations": (["limitations", "discussion", "conclusion"], "limitations weaknesses future work drawbacks"),
}


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def detect_heading(line: str) -> Optional[str]:
    s = line.strip()
    if not s or len(s) > 70:
        return None
    s = _NUMBERING.sub("", s).strip(" :.").lower()
    return _ALIAS_LOOKUP.get(s)


@dataclass
class Paper:
    paper_id: str
    title: str
    n_pages: int
    sections: List[dict] = field(default_factory=list)  # {"section", "text", "page"}
    n_chunks: int = 0
    authors: List[str] = field(default_factory=list)
    year: str = ""
    url: str = ""
    source: str = "upload"  # "upload" or "arxiv"

    @property
    def detected_sections(self) -> List[str]:
        out = []
        for s in self.sections:
            if s["section"] not in out:
                out.append(s["section"])
        return out

    @property
    def short_title(self) -> str:
        return self.title if len(self.title) <= 45 else self.title[:42] + "..."


def _guess_title(reader: PdfReader, fallback: str) -> str:
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    bad = ("microsoft word", "untitled", ".pdf", ".doc")
    if len(meta_title) > 10 and not any(b in meta_title.lower() for b in bad):
        return meta_title.strip()
    first = (reader.pages[0].extract_text() or "") if reader.pages else ""
    for line in first.splitlines():
        line = line.strip()
        if 15 <= len(line) <= 200 and not line.lower().startswith(("arxiv", "http", "doi", "vol")):
            return line
    return fallback


def parse_pdf(file, paper_id: str) -> Paper:
    """Read a PDF and split its text into (section, page) blocks. Stops at References."""
    reader = PdfReader(file)
    paper = Paper(paper_id=paper_id, title=_guess_title(reader, paper_id), n_pages=len(reader.pages))

    current, buf, buf_page = "front", [], 1

    def flush():
        text = " ".join(buf).strip()
        if text:
            paper.sections.append({"section": current, "text": text, "page": buf_page})

    for pno, page in enumerate(reader.pages, start=1):
        flush()                     # keep page numbers accurate for citations
        buf, buf_page = [], pno
        stop = False
        for raw in (page.extract_text() or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            inline = _ABSTRACT_INLINE.match(line)
            if inline and current == "front":
                flush()
                current, buf, buf_page = "abstract", [], pno
                if inline.group(1):
                    buf.append(inline.group(1))
                continue
            sec = detect_heading(line)
            if sec:
                flush()
                current, buf, buf_page = sec, [], pno
                if sec == "references":
                    stop = True
                    break
                continue
            buf.append(line)
        if stop:
            buf = []
            break
    flush()
    return paper


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['title']} | {d.metadata['section']} | p. {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


class Library:
    """All papers in one FAISS index + one BM25 index, with paper metadata on every chunk."""

    def __init__(self, embeddings, reranker=None, embedding_name: str = "",
                 chunk_size: int = 1000, chunk_overlap: int = 150):
        self.embeddings = embeddings
        self.embedding_name = embedding_name
        self.reranker = reranker          # anything with .predict([(query, text), ...]) -> scores
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.papers: Dict[str, Paper] = {}
        self.chunks: List[Document] = []
        self.store: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None

    # ---------- indexing ----------
    def add_pdf(self, file, paper_id: str, meta: Optional[dict] = None) -> Paper:
        if paper_id in self.papers:
            return self.papers[paper_id]
        paper = parse_pdf(file, paper_id)
        if meta:  # metadata from arXiv is cleaner than guessing from the PDF
            paper.title = meta.get("title") or paper.title
            paper.authors = meta.get("authors", [])
            paper.year = str(meta.get("year", ""))
            paper.url = meta.get("url", "")
            paper.source = "arxiv"

        docs = []
        for s in paper.sections:
            for piece in self.splitter.split_text(s["text"]):
                docs.append(Document(
                    page_content=piece,
                    metadata={"paper": paper_id, "title": paper.title, "section": s["section"],
                              "page": s["page"], "chunk_id": len(self.chunks) + len(docs)},
                ))
        if not docs:
            raise ValueError(f"No text could be extracted from {paper_id} (is it a scanned PDF?)")
        if self.store is None:
            self.store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.store.add_documents(docs)
        paper.n_chunks = len(docs)
        self.papers[paper_id] = paper
        self.chunks.extend(docs)
        self.bm25 = BM25Okapi([tokenize(d.page_content) or ["_"] for d in self.chunks])
        return paper

    # ---------- retrieval ----------
    def search(self, query: str, k: int = 4, paper: Optional[str] = None, rerank: bool = True) -> List[Document]:
        """Hybrid search: dense (FAISS) + sparse (BM25) fused with RRF, then optional reranking."""
        if self.store is None:
            return []
        pool = max(k * 4, 12)

        # 1) dense
        if paper:
            dense = self.store.similarity_search(query, k=pool, filter={"paper": paper}, fetch_k=max(100, pool * 10))
        else:
            dense = self.store.similarity_search(query, k=pool)
        dense_ids = [d.metadata["chunk_id"] for d in dense]

        # 2) sparse
        scores = self.bm25.get_scores(tokenize(query))
        candidates = [i for i, d in enumerate(self.chunks) if paper is None or d.metadata["paper"] == paper]
        sparse_ids = [i for i in sorted(candidates, key=lambda i: scores[i], reverse=True)[:pool] if scores[i] > 0]

        # 3) Reciprocal Rank Fusion
        fused: Dict[int, float] = {}
        for ranked in (dense_ids, sparse_ids):
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:pool]

        # 4) cross-encoder reranking
        if rerank and self.reranker is not None and ranked_ids:
            rr = self.reranker.predict([(query, self.chunks[i].page_content) for i in ranked_ids])
            ranked_ids = [cid for _, cid in sorted(zip(rr, ranked_ids), key=lambda x: -float(x[0]))]

        return [self.chunks[i] for i in ranked_ids[:k]]

    def search_each_paper(self, query: str, k_per_paper: int = 3, papers: Optional[List[str]] = None) -> List[Document]:
        """Multi-paper retrieval: top-k from EVERY paper, so no paper is ignored."""
        out = []
        for pid in papers or list(self.papers):
            out.extend(self.search(query, k=k_per_paper, paper=pid))
        return out

    def by_section(self, paper_id: str, sections: List[str], limit: int) -> List[Document]:
        hits = [d for d in self.chunks if d.metadata["paper"] == paper_id and d.metadata["section"] in sections]
        return hits[:limit]

    def paper_context(self, paper_id: str, max_chars: int = 14000) -> str:
        """Context for one paper: abstract + best chunks for each field (round robin)."""
        groups = [self.by_section(paper_id, ["abstract"], 2) or self.by_section(paper_id, ["front", "introduction"], 2)]
        for sections, query in FIELD_HINTS.values():
            groups.append(self.by_section(paper_id, sections, 1) + self.search(query, k=2, paper=paper_id))

        picked, seen, total = [], set(), 0
        for i in range(max(len(g) for g in groups)):
            for g in groups:
                if i >= len(g):
                    continue
                d = g[i]
                if d.metadata["chunk_id"] in seen:
                    continue
                if total + len(d.page_content) > max_chars:
                    return format_docs(picked)
                seen.add(d.metadata["chunk_id"])
                picked.append(d)
                total += len(d.page_content)
        return format_docs(picked)
