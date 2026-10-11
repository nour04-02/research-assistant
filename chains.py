"""All prompts and chains of the assistant.

Chains:
    extract       : 6 structured fields per paper (StructuredOutputParser)
    facts         : methods / datasets / tasks + numeric results per paper (for chart + graph)
    compare       : comparison of several papers
    differences   : key differences + research gaps
    ask           : multi-paper Q&A with citations
    verify        : fact-checks every sentence of an answer against the sources (hallucination check)
    related_work  : academic "Related Work" section with numbered citations
    route         : the agent's router, picks the right tool for a free-text request
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

try:
    from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:  # older LangChain
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from rag import Library, format_docs

FIELDS = [
    ("paper", "Full title of the paper."),
    ("methodology", "The proposed method, model or approach and how it works."),
    ("dataset", "Datasets used: name, size, source, preprocessing."),
    ("results", "Main quantitative results with exact numbers and metrics."),
    ("advantages", "Strengths and contributions of the work."),
    ("limitations", "Weaknesses, assumptions and limitations."),
]
FIELD_NAMES = [f for f, _ in FIELDS]
NOT_REPORTED = "Not reported"
TOOLS = ["qa", "compare", "chart", "related_work", "graph", "arxiv"]

output_parser = StructuredOutputParser.from_response_schemas(
    [ResponseSchema(name=n, description=d) for n, d in FIELDS]
)

LANGUAGE_RULES = {
    "English": "Write your answer in English.",
    "Arabic": "Write your answer in Modern Standard Arabic, but keep technical terms, model names, "
              "dataset names and metric names in English. Keep JSON keys in English.",
}

# ======================= PROMPTS =======================
EXTRACT_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant helping a researcher review scientific papers.

Below are excerpts from ONE paper, titled: "{title}".
Each excerpt is labeled with [title | section | page].

Extract the following information using ONLY these excerpts:
- paper: the full paper title
- methodology: the proposed method/model/approach and how it works (2-4 sentences)
- dataset: datasets used, their size, source and preprocessing (1-3 sentences)
- results: main quantitative results with the exact numbers and metrics (2-4 sentences)
- advantages: strengths and contributions of the work (2-3 short points)
- limitations: weaknesses, assumptions or limitations, stated by the authors or clearly evident (2-3 short points)

Rules:
- If something is not in the excerpts, write "Not reported". Never invent numbers.
- Every value must be a single string.
- {language_rule}

{format_instructions}

Excerpts:
{context}
"""
)

FACTS_PROMPT = PromptTemplate.from_template(
    """You are an information extraction system. Read the excerpts from the paper "{title}"
and extract structured FACTS. Return ONLY a JSON object inside a ```json block:
{{
  "methods": ["short names of the models or techniques the paper proposes or uses (max 6)"],
  "datasets": ["dataset names (max 6)"],
  "tasks": ["research tasks, e.g. image classification (max 3)"],
  "results": [{{"model": "model name", "dataset": "dataset name", "metric": "metric name", "value": 0.0, "unit": "%"}}]
}}
Rules:
- Use short canonical names, e.g. "ImageNet", "ResNet-50", "Top-1 Accuracy".
- results: only numbers that literally appear in the excerpts, focus on the main results of THIS paper (max 8).
- "value" must be a plain number. "unit" is "%" or "".
- Do not write anything outside the JSON block.

Excerpts:
{context}
"""
)

COMPARE_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant. Below are structured summaries of {n} papers.

{summaries}

Write a comparison of these papers in Markdown with these parts:
## Overview
2-3 sentences on what problem the papers address together.
## Common ground
2-4 bullets on what the papers share (task, data, methods, metrics).
## Methodology comparison
A short paragraph comparing HOW each paper approaches the problem.
## Which to use when
One bullet per paper on the situation it suits best.

Base everything ONLY on the summaries. {language_rule}
"""
)

DIFFERENCES_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant. Below are structured summaries of {n} papers.

{summaries}

Identify the key DIFFERENCES between these papers. Use Markdown.
For each aspect below write a ### heading, then one bullet per paper stating its choice,
then one line starting with **Why it matters:**.
Aspects: Methodology, Data, Evaluation, Results, Limitations.

Finish with a ### Research gaps section: 2-3 bullets on questions none of the papers address.

Base everything ONLY on the summaries. {language_rule}
"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant. Answer the researcher's question using ONLY the excerpts
below. They come from several papers and are labeled with [title | section | page].
Cite the source after each claim like [title, p. X]. If several papers are relevant, cover each of them.
If the excerpts do not contain the answer, say so clearly.
{language_rule}

Excerpts:
{context}

Question: {question}
Answer:"""
)

VERIFY_PROMPT = PromptTemplate.from_template(
    """You are a strict fact-checker for a research assistant.
Below are SOURCE excerpts and a numbered list of CLAIMS taken from an answer.
For each claim decide:
- "supported": clearly stated in the sources
- "partial": only partly supported, or the numbers/details are imprecise
- "unsupported": not in the sources, or contradicts them
A claim that only says the information is not available counts as "supported".

Return ONLY a JSON list inside a ```json block, one object per claim:
[{{"id": 1, "verdict": "supported", "source": "paper title, p. X"}}]

SOURCES:
{context}

CLAIMS:
{claims}
"""
)

RELATED_WORK_PROMPT = PromptTemplate.from_template(
    """You are an academic writer. Using ONLY the paper summaries below, write a "Related Work" section
for a research paper. Each paper has a citation number like [1].

{summaries}

Requirements:
- 2 to 3 well-connected paragraphs in formal academic style.
- Group the papers by approach (do not just list them one by one), and show how they relate or differ.
- Cite papers with their numbers, e.g. "... residual connections [1], while transformer-based models [2] ...".
- End with one sentence about the gap that motivates new research.
- Do not write a references list and do not add a title. {language_rule}
"""
)

ROUTER_PROMPT = PromptTemplate.from_template(
    """You are the routing agent of a research assistant. Choose the ONE best tool for the user's request.

Tools:
- qa: answer a question using the content of the loaded papers (default choice)
- compare: compare or contrast the loaded papers, or find their differences
- chart: visually compare the numeric results (accuracy, F1, ...) of the loaded papers
- related_work: write a related work / literature review section from the loaded papers
- graph: show how the papers are connected through shared methods and datasets
- arxiv: find NEW papers on the internet about a topic

Loaded papers: {papers}

User request: {request}

Return ONLY a JSON object inside a ```json block:
{{"tool": "one of qa, compare, chart, related_work, graph, arxiv", "query": "search keywords if tool is arxiv, else empty", "reason": "one short sentence"}}
"""
)


# ======================= PARSING =======================
def _json_candidates(text: str) -> List[str]:
    cands = re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            cands.append(text[s:e + 1])
    return cands


def parse_json_any(text: str):
    """Return the first valid JSON object/list found in the LLM output, else None."""
    for c in _json_candidates(text):
        for attempt in (c, re.sub(r",\s*([}\]])", r"\1", c)):  # also try without trailing commas
            try:
                return json.loads(attempt, strict=False)
            except json.JSONDecodeError:
                continue
    return None


def _to_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {str(v).strip()}" for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip()


def parse_json_output(text: str, keys: List[str]) -> Dict[str, str]:
    """Parse the 6-field extraction (code block, bare JSON, or per-key regex as last resort)."""
    data = parse_json_any(text)
    if not isinstance(data, dict):
        data = {}
        for k in keys:
            m = re.search(rf'"{k}"\s*:\s*"(.*?)"\s*(?:,\s*"|\}})', text, re.DOTALL)
            if m:
                data[k] = m.group(1)
    out = {k: _to_text(data.get(k, "")) or NOT_REPORTED for k in keys}
    out["_parsed_ok"] = any(v != NOT_REPORTED for v in out.values())
    return out


def _to_number(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def parse_facts(text: str) -> Dict:
    data = parse_json_any(text)
    data = data if isinstance(data, dict) else {}
    clean = lambda xs: [str(x).strip() for x in (xs or []) if str(x).strip()][:6]
    results = []
    for r in data.get("results") or []:
        if not isinstance(r, dict):
            continue
        val = _to_number(r.get("value"))
        if val is None:
            continue
        unit = str(r.get("unit", "")).strip()
        if "%" in str(r.get("value", "")):
            unit = "%"
        results.append({
            "model": str(r.get("model", "")).strip(),
            "dataset": str(r.get("dataset", "")).strip() or "N/A",
            "metric": str(r.get("metric", "")).strip() or "Score",
            "value": val,
            "unit": unit,
        })
    return {"methods": clean(data.get("methods")), "datasets": clean(data.get("datasets")),
            "tasks": clean(data.get("tasks"))[:3], "results": results[:8], "_parsed_ok": bool(data)}


def split_claims(answer: str, max_claims: int = 12) -> List[str]:
    """Split an answer into sentence-level claims (works for English and Arabic)."""
    text = re.sub(r"[*#`>|]", " ", answer)
    parts = re.split(r"(?<=[.!?؟])\s+(?=[A-Z؀-ۿ\"(])|\n+", text)
    claims = [re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", p).strip() for p in parts]
    return [c for c in claims if len(c) > 20][:max_claims]


# ======================= HELPERS =======================
def summaries_to_text(summaries: List[Dict[str, str]], numbered: bool = False) -> str:
    blocks = []
    for i, s in enumerate(summaries, start=1):
        head = f"[{i}] {s['paper']}" if numbered else f"### Paper {i}: {s['paper']}"
        lines = [head] + [f"- {name.capitalize()}: {s[name]}" for name in FIELD_NAMES[1:]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _authors_short(authors: List[str]) -> str:
    if not authors:
        return ""
    last = authors[0].split()[-1]
    return last if len(authors) == 1 else (f"{last} and {authors[1].split()[-1]}" if len(authors) == 2 else f"{last} et al.")


def build_references(library: Library, ids: List[str]) -> str:
    lines = []
    for i, pid in enumerate(ids, start=1):
        p = library.papers[pid]
        who = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        parts = [x for x in (who, f"({p.year})" if p.year else "", f"*{p.title}*", p.url) if x]
        lines.append(f"[{i}] " + ". ".join(parts))
    return "\n\n".join(lines)


def build_bibtex(library: Library, ids: List[str]) -> str:
    entries = []
    for pid in ids:
        p = library.papers[pid]
        first = re.sub(r"\W", "", (p.authors[0].split()[-1] if p.authors else "unknown")).lower()
        word = re.sub(r"\W", "", p.title.split()[0]).lower() if p.title else "paper"
        key = f"{first}{p.year}{word}"
        fields = [f"  title = {{{p.title}}}",
                  f"  author = {{{' and '.join(p.authors) or 'Unknown'}}}"]
        if p.year:
            fields.append(f"  year = {{{p.year}}}")
        if p.url:
            fields.append(f"  url = {{{p.url}}}")
        if p.source == "arxiv":
            fields.append(f"  journal = {{arXiv preprint arXiv:{pid.split(':')[-1]}}}")
        entries.append(f"@article{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


# ======================= CHAINS =======================
class ResearchChains:
    def __init__(self, llm, library: Library, language: str = "English",
                 extractions: Optional[Dict] = None, facts: Optional[Dict] = None):
        self.library = library
        self.language_rule = LANGUAGE_RULES.get(language, LANGUAGE_RULES["English"])
        self.extractions = extractions if extractions is not None else {}
        self.facts_cache = facts if facts is not None else {}
        self.extract_chain = EXTRACT_PROMPT | llm
        self.facts_chain = FACTS_PROMPT | llm
        self.compare_chain = COMPARE_PROMPT | llm
        self.differences_chain = DIFFERENCES_PROMPT | llm
        self.qa_chain = QA_PROMPT | llm
        self.verify_chain = VERIFY_PROMPT | llm
        self.related_chain = RELATED_WORK_PROMPT | llm
        self.router_chain = ROUTER_PROMPT | llm

    # ----- per paper -----
    def extract(self, paper_id: str, force: bool = False) -> Dict[str, str]:
        if paper_id in self.extractions and not force:
            return self.extractions[paper_id]
        paper = self.library.papers[paper_id]
        raw = self.extract_chain.invoke({
            "title": paper.title,
            "context": self.library.paper_context(paper_id),
            "language_rule": self.language_rule,
            "format_instructions": output_parser.get_format_instructions(),
        })
        data = parse_json_output(raw, FIELD_NAMES)
        if data["paper"] == NOT_REPORTED:
            data["paper"] = paper.title
        data["_id"], data["_raw"] = paper_id, raw
        self.extractions[paper_id] = data
        return data

    def facts(self, paper_id: str, force: bool = False) -> Dict:
        if paper_id in self.facts_cache and not force:
            return self.facts_cache[paper_id]
        paper = self.library.papers[paper_id]
        raw = self.facts_chain.invoke({"title": paper.title, "context": self.library.paper_context(paper_id)})
        data = parse_facts(raw)
        data["_raw"] = raw
        self.facts_cache[paper_id] = data
        return data

    # ----- multi paper -----
    def compare(self, summaries: List[Dict[str, str]]) -> str:
        return self.compare_chain.invoke({
            "n": len(summaries), "summaries": summaries_to_text(summaries), "language_rule": self.language_rule,
        })

    def differences(self, summaries: List[Dict[str, str]]) -> str:
        return self.differences_chain.invoke({
            "n": len(summaries), "summaries": summaries_to_text(summaries), "language_rule": self.language_rule,
        })

    def run_full_comparison(self, paper_ids: List[str]) -> Tuple[List[Dict], str, str]:
        """Sequential chain: extract every paper -> compare -> differences."""
        summaries = [self.extract(pid) for pid in paper_ids]
        return summaries, self.compare(summaries), self.differences(summaries)

    def related_work(self, paper_ids: List[str]) -> Dict[str, str]:
        summaries = [self.extract(pid) for pid in paper_ids]
        text = self.related_chain.invoke({
            "summaries": summaries_to_text(summaries, numbered=True), "language_rule": self.language_rule,
        })
        return {"text": text.strip(),
                "references": build_references(self.library, paper_ids),
                "bibtex": build_bibtex(self.library, paper_ids)}

    # ----- Q&A + hallucination check -----
    def ask(self, question: str, papers: Optional[List[str]] = None, k_per_paper: int = 3):
        docs = self.library.search_each_paper(question, k_per_paper=k_per_paper, papers=papers)
        answer = self.qa_chain.invoke({
            "question": question, "context": format_docs(docs), "language_rule": self.language_rule,
        })
        return answer.strip(), docs

    def verify(self, answer: str, docs) -> Dict:
        claims = split_claims(answer)
        if not claims:
            return {"claims": [], "score": None}
        raw = self.verify_chain.invoke({
            "context": format_docs(docs),
            "claims": "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1)),
        })
        verdicts = parse_json_any(raw)
        by_id = {}
        if isinstance(verdicts, list):
            for v in verdicts:
                if isinstance(v, dict):
                    try:
                        by_id[int(v.get("id"))] = v
                    except (TypeError, ValueError):
                        continue
        out = []
        for i, c in enumerate(claims, start=1):
            v = by_id.get(i, {})
            verdict = str(v.get("verdict", "unknown")).lower()
            if verdict not in ("supported", "partial", "unsupported"):
                verdict = "unknown"
            out.append({"claim": c, "verdict": verdict, "source": str(v.get("source", ""))})
        weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
        scored = [weights[o["verdict"]] for o in out if o["verdict"] in weights]
        return {"claims": out, "score": (sum(scored) / len(scored)) if scored else None}

    # ----- agent router -----
    def route(self, request: str) -> Dict[str, str]:
        papers = "; ".join(p.title for p in self.library.papers.values()) or "none"
        raw = self.router_chain.invoke({"request": request, "papers": papers})
        data = parse_json_any(raw)
        data = data if isinstance(data, dict) else {}
        tool = str(data.get("tool", "qa")).strip().lower()
        if tool not in TOOLS:
            tool = "qa"
        if not self.library.papers and tool != "arxiv":
            tool = "arxiv"
        return {"tool": tool, "query": str(data.get("query", "")).strip() or request,
                "reason": str(data.get("reason", "")).strip()}
