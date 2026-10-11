"""AI Research Assistant
RAG + Hybrid Multi-Document Retrieval + Chains + Agent + Hallucination Check

Run:  streamlit run app.py
"""
import base64
import io
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from arxiv_client import download_pdf, search_arxiv
from chains import FIELD_NAMES, ResearchChains
from graph import LEGEND, build_graph
from llm_client import MockLLM, RemoteLLM
from rag import Library

EMBEDDING_MODELS = {
    "Multilingual MiniLM (fast, Arabic + English)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bge-m3 (multilingual, best, ~2.2GB)": "BAAI/bge-m3",
    "MiniLM (fastest, English only)": "sentence-transformers/all-MiniLM-L6-v2",
}
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # multilingual cross-encoder
LABELS = {"paper": "Paper", "methodology": "Methodology", "dataset": "Dataset",
          "results": "Results", "advantages": "Advantages", "limitations": "Limitations"}
VERDICT_ICON = {"supported": "✅", "partial": "🟡", "unsupported": "⚠️", "unknown": "❔"}

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")


# ---------------- cached models ----------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embeddings(name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})


@st.cache_resource(show_spinner="Loading reranker (first time only)...")
def get_reranker(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


# ---------------- state ----------------
ss = st.session_state
for key, default in {"library": None, "extractions": {}, "facts": {}, "chat": [], "comparison": None,
                     "arxiv_results": [], "related": None}.items():
    ss.setdefault(key, default)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Model server")
    demo = st.toggle("Demo mode (no GPU)", value=False, help="Fake answers, for testing the interface.")
    url = st.text_input("ngrok URL", value=os.getenv("LLM_URL", ""), disabled=demo,
                        placeholder="https://xxxx.ngrok-free.app")
    api_key = st.text_input("API key", value=os.getenv("LLM_API_KEY", "secret123"), type="password", disabled=demo)
    language = st.radio("Output language", ["English", "Arabic"], horizontal=True)

    llm = MockLLM() if demo else (RemoteLLM(url=url, api_key=api_key) if url else None)
    if not demo and st.button("Test connection"):
        if not llm:
            st.error("Add the ngrok URL first.")
        else:
            try:
                st.success(f"Connected: {llm.health().get('model')}")
            except Exception as e:
                st.error(f"Cannot reach server: {e}")

    st.divider()
    st.header("🧭 Retrieval")
    emb_label = st.selectbox("Embedding model", list(EMBEDDING_MODELS), disabled=ss.library is not None,
                             help="Fixed once papers are indexed. Clear all to change it.")
    emb_name = EMBEDDING_MODELS[emb_label]
    use_reranker = st.toggle("Cross-encoder reranker", value=True,
                             help="Re-scores the hybrid (BM25 + FAISS) results for higher precision.")

    st.divider()
    st.header("📚 Papers")
    files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)
    index_clicked = st.button("Index papers", type="primary", disabled=not files)


def ensure_library() -> Library:
    if ss.library is None:
        ss.library = Library(get_embeddings(emb_name), embedding_name=emb_name)
    return ss.library


if index_clicked:
    lib = ensure_library()
    for f in files:
        with st.sidebar, st.spinner(f"Indexing {f.name}..."):
            try:
                lib.add_pdf(io.BytesIO(f.getvalue()), f.name)
            except Exception as e:
                st.error(f"{f.name}: {e}")

if ss.library is not None:
    ss.library.reranker = get_reranker(RERANKER_MODEL) if use_reranker else None

with st.sidebar:
    if ss.library and ss.library.papers:
        for p in ss.library.papers.values():
            icon = "🌐" if p.source == "arxiv" else "📄"
            st.caption(f"{icon} **{p.short_title}**  \n{p.n_pages} pages · {p.n_chunks} chunks")
        if st.button("Clear all"):
            for k in ("library", "comparison", "related"):
                ss[k] = None
            ss.extractions, ss.facts, ss.chat, ss.arxiv_results = {}, {}, [], []
            st.rerun()

# ---------------- helpers ----------------
lib = ss.library
paper_ids = list(lib.papers) if lib else []
chains = ResearchChains(llm, lib, language, ss.extractions, ss.facts) if (llm and lib) else None


def title_of(pid):
    return lib.papers[pid].title


def ready(need_llm: bool = True, min_papers: int = 1) -> bool:
    if len(paper_ids) < min_papers:
        st.info(f"Add at least {min_papers} paper(s) first: upload PDFs in the sidebar or use the 🔎 Discover tab.")
        return False
    if need_llm and llm is None:
        st.warning("Add the ngrok URL in the sidebar or turn on Demo mode.")
        return False
    return True


def summary_table(ids):
    rows = [{LABELS[f]: ss.extractions[pid][f] for f in FIELD_NAMES} for pid in ids if pid in ss.extractions]
    return pd.DataFrame(rows)


def run_for_papers(ids, fn, label):
    """Run fn(pid) for every paper with a progress bar. Returns True if all succeeded."""
    bar, ok = st.progress(0.0, text=label), True
    for i, pid in enumerate(ids, start=1):
        try:
            fn(pid)
        except Exception as e:
            ok = False
            st.error(f"{title_of(pid)}: {e}")
        bar.progress(i / len(ids), text=f"{label} ({i}/{len(ids)})")
    bar.empty()
    return ok


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def results_df(ids) -> pd.DataFrame:
    rows = []
    for pid in ids:
        for r in ss.facts.get(pid, {}).get("results", []):
            rows.append({"paper": lib.papers[pid].short_title, "model": r["model"] or lib.papers[pid].short_title,
                         "dataset": r["dataset"], "metric": r["metric"], "value": r["value"], "unit": r["unit"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ds_key"], df["m_key"] = df["dataset"].map(_norm), df["metric"].map(_norm)
    return df


def chart_options(df):
    groups = (df.groupby(["ds_key", "m_key"])
                .agg(n=("paper", "nunique"), dataset=("dataset", "first"), metric=("metric", "first"))
                .reset_index().sort_values(["n"], ascending=False))
    return [(r.ds_key, r.m_key, f"{r.metric} on {r.dataset}  ({r.n} paper{'s' if r.n > 1 else ''})") for r in groups.itertuples()]


def make_chart(df, ds, m):
    sub = df[(df.ds_key == ds) & (df.m_key == m)]
    unit = sub["unit"].iloc[0] if len(sub) else ""
    fig = px.bar(sub, x="model", y="value", color="paper", barmode="group", text="value",
                 title=f"{sub['metric'].iloc[0]} on {sub['dataset'].iloc[0]}",
                 labels={"value": f"{sub['metric'].iloc[0]} {('(' + unit + ')') if unit else ''}", "model": ""})
    fig.update_traces(texttemplate="%{text:.4g}", textposition="outside")
    fig.update_layout(legend_title_text="Paper", margin=dict(t=60, b=20))
    return fig


def best_chart(ids):
    df = results_df(ids)
    if df.empty:
        return None, df
    ds, m, _ = chart_options(df)[0]
    return make_chart(df, ds, m), df


def graph_html(ids, only_shared=False):
    return build_graph({pid: (lib.papers[pid].title, ss.facts[pid]) for pid in ids if pid in ss.facts},
                       only_shared=only_shared)


def show_html(html: str, height: int):
    """Embed raw HTML. Works on old Streamlit (components.html) and new Streamlit (st.iframe)."""
    if hasattr(components, "html"):
        components.html(html, height=height)
    else:
        st.iframe("data:text/html;base64," + base64.b64encode(html.encode()).decode(), height=height)


def render_verification(v):
    if not v or v.get("score") is None:
        return
    score = v["score"]
    color = "green" if score >= 0.8 else ("orange" if score >= 0.5 else "red")
    st.markdown(f"**Faithfulness score:** :{color}[{score:.0%}] of the answer is supported by the papers")
    with st.expander("Claim-by-claim check"):
        for c in v["claims"]:
            src = f"  \n<small>{c['source']}</small>" if c["source"] else ""
            st.markdown(f"{VERDICT_ICON[c['verdict']]} {c['claim']}{src}", unsafe_allow_html=True)


def render_sources(docs):
    with st.expander(f"Sources ({len(docs)} chunks)"):
        for d in docs:
            st.caption(f"**{d.metadata['title'][:70]}** · {d.metadata['section']} · p. {d.metadata['page']}")
            st.text(d.page_content[:400] + "...")


def render_arxiv_list(results):
    for r in results:
        st.markdown(f"**[{r['title']}]({r['url']})**  \n"
                    f"<small>{', '.join(r['authors'][:4])}{' et al.' if len(r['authors']) > 4 else ''} · {r['year']}</small>",
                    unsafe_allow_html=True)


# ---------------- main ----------------
st.title("🔬 AI Research Assistant")
st.caption("Discover papers, extract their key information, compare results, map connections, "
           "and ask fact-checked questions across all of them.")

tabs = st.tabs(["🔎 Discover", "📄 Extract", "⚖️ Compare", "📊 Results", "🕸️ Graph", "✍️ Related Work", "💬 Ask"])

# ----- Discover (arXiv) -----
with tabs[0]:
    st.subheader("Find papers on arXiv")
    c1, c2 = st.columns([4, 1])
    query = c1.text_input("Topic", placeholder="e.g. hypertrophic cardiomyopathy deep learning MRI")
    n_results = c2.number_input("Results", 3, 20, 8)
    if st.button("Search arXiv", type="primary", disabled=not query):
        with st.spinner("Searching arXiv..."):
            try:
                ss.arxiv_results = search_arxiv(query, int(n_results))
            except Exception as e:
                st.error(f"arXiv search failed: {e}")
    if ss.arxiv_results:
        selected = []
        for r in ss.arxiv_results:
            in_lib = lib is not None and f"arXiv:{r['id']}" in lib.papers
            col_a, col_b = st.columns([0.06, 0.94])
            if col_a.checkbox("add", key=f"ax_{r['id']}", disabled=in_lib, label_visibility="collapsed"):
                selected.append(r)
            with col_b:
                st.markdown(f"**{r['title']}** {'✅ in library' if in_lib else ''}  \n"
                            f"<small>{', '.join(r['authors'][:4])}{' et al.' if len(r['authors']) > 4 else ''} · "
                            f"{r['year']} · <a href='{r['url']}'>arXiv:{r['id']}</a></small>", unsafe_allow_html=True)
                with st.expander("Abstract"):
                    st.write(r["summary"])
        if st.button(f"Download & index {len(selected)} selected", disabled=not selected):
            library = ensure_library()
            for r in selected:
                with st.spinner(f"Downloading {r['title'][:60]}..."):
                    try:
                        library.add_pdf(io.BytesIO(download_pdf(r["pdf_url"])), f"arXiv:{r['id']}", meta=r)
                    except Exception as e:
                        st.error(f"{r['title'][:60]}: {e}")
            st.rerun()

# ----- Extract -----
with tabs[1]:
    if ready():
        chosen = st.multiselect("Papers", paper_ids, default=paper_ids, format_func=title_of, key="ex_sel")
        if st.button("Extract information", type="primary", disabled=not chosen):
            run_for_papers(chosen, lambda pid: chains.extract(pid, force=True), "Reading papers")
        done = [pid for pid in chosen if pid in ss.extractions]
        if done:
            df = summary_table(done)
            st.dataframe(df, hide_index=True)
            st.download_button("Download table (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               "papers_summary.csv", "text/csv")
            for pid in done:
                data = ss.extractions[pid]
                with st.expander(data["paper"]):
                    if not data["_parsed_ok"]:
                        st.warning("The model did not return valid JSON. Raw output below.")
                        st.text(data["_raw"])
                    for f in FIELD_NAMES[1:]:
                        st.markdown(f"**{LABELS[f]}**")
                        st.markdown(data[f])

# ----- Compare -----
with tabs[2]:
    if ready(min_papers=2):
        chosen = st.multiselect("Choose at least 2 papers", paper_ids, default=paper_ids[:3],
                                format_func=title_of, key="cmp_sel")
        if st.button("Compare papers", type="primary", disabled=len(chosen) < 2):
            run_for_papers([p for p in chosen if p not in ss.extractions], chains.extract, "Extracting")
            with st.spinner("Comparing and finding differences..."):
                try:
                    _, comparison, diffs = chains.run_full_comparison(chosen)
                    ss.comparison = {"ids": chosen, "comparison": comparison, "differences": diffs}
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
        if ss.comparison:
            st.subheader("Side by side")
            st.dataframe(summary_table(ss.comparison["ids"]), hide_index=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Comparison")
                st.markdown(ss.comparison["comparison"])
            with col2:
                st.subheader("Key differences")
                st.markdown(ss.comparison["differences"])
            report = ("# Papers comparison\n\n" + summary_table(ss.comparison["ids"]).to_markdown(index=False)
                      + "\n\n" + ss.comparison["comparison"] + "\n\n# Key differences\n\n" + ss.comparison["differences"])
            st.download_button("Download report (Markdown)", report, "comparison_report.md")

# ----- Results chart -----
with tabs[3]:
    if ready():
        st.subheader("Compare the numbers")
        chosen = st.multiselect("Papers", paper_ids, default=paper_ids, format_func=title_of, key="res_sel")
        if st.button("Extract numeric results", type="primary", disabled=not chosen):
            run_for_papers(chosen, lambda pid: chains.facts(pid, force=True), "Extracting results")
        df = results_df(chosen)
        if not df.empty:
            options = chart_options(df)
            pick = st.selectbox("Metric", range(len(options)), format_func=lambda i: options[i][2])
            st.plotly_chart(make_chart(df, options[pick][0], options[pick][1]))
            if options[pick][2].endswith("(1 paper)"):
                st.caption("Only one paper reports this metric. Pick a metric shared by several papers to compare.")
            show = df.drop(columns=["ds_key", "m_key"])
            st.dataframe(show, hide_index=True)
            st.download_button("Download results (CSV)", show.to_csv(index=False).encode("utf-8-sig"), "results.csv")
        elif any(pid in ss.facts for pid in chosen):
            st.info("No numeric results were found in the selected papers.")

# ----- Knowledge graph -----
with tabs[4]:
    if ready():
        st.subheader("How the papers connect")
        chosen = st.multiselect("Papers", paper_ids, default=paper_ids, format_func=title_of, key="graph_sel")
        only_shared = st.toggle("Show only shared methods and datasets", value=False)
        if st.button("Build graph", type="primary", disabled=not chosen):
            run_for_papers([p for p in chosen if p not in ss.facts], chains.facts, "Extracting entities")
        if any(pid in ss.facts for pid in chosen):
            html, shared = graph_html(chosen, only_shared)
            st.markdown(LEGEND, unsafe_allow_html=True)
            show_html(html, 620)
            if shared:
                st.markdown("**Shared across papers:** " + ", ".join(
                    f"{s['entity']} ({s['type']}, {s['papers']} papers)" for s in shared))
            else:
                st.caption("No shared methods or datasets found between the selected papers.")

# ----- Related work -----
with tabs[5]:
    if ready():
        st.subheader("Generate a Related Work section")
        chosen = st.multiselect("Papers to cite (in this order)", paper_ids, default=paper_ids,
                                format_func=title_of, key="rw_sel")
        if st.button("Write related work", type="primary", disabled=not chosen):
            run_for_papers([p for p in chosen if p not in ss.extractions], chains.extract, "Extracting")
            with st.spinner("Writing..."):
                try:
                    ss.related = chains.related_work(chosen)
                except Exception as e:
                    st.error(f"Failed: {e}")
        if ss.related:
            st.markdown(ss.related["text"])
            st.markdown("**References**")
            st.markdown(ss.related["references"])
            with st.expander("BibTeX"):
                st.code(ss.related["bibtex"], language="bibtex")
            full = "## Related Work\n\n" + ss.related["text"] + "\n\n## References\n\n" + ss.related["references"]
            c1, c2 = st.columns(2)
            c1.download_button("Download (Markdown)", full, "related_work.md")
            c2.download_button("Download BibTeX", ss.related["bibtex"], "references.bib")

# ----- Ask (agent + hallucination check) -----
with tabs[6]:
    c1, c2 = st.columns(2)
    agent_mode = c1.toggle("🤖 Agent mode", value=True,
                           help="The agent picks the right tool: Q&A, compare, chart, graph, related work or arXiv search.")
    verify = c2.toggle("✅ Fact-check answers", value=True,
                       help="A second chain checks every sentence of the answer against the retrieved sources.")
    scope = st.multiselect("Search in", paper_ids, default=paper_ids,
                           format_func=title_of, key="ask_sel") if paper_ids else []

    for msg in ss.chat:
        with st.chat_message(msg["role"]):
            if msg.get("agent"):
                st.caption(f"🤖 Agent used **{msg['agent']['tool']}**: {msg['agent']['reason']}")
            if msg.get("content"):
                st.markdown(msg["content"])
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], key=f"fig_{id(msg)}")
            if msg.get("graph"):
                show_html(msg["graph"], 520)
            if msg.get("arxiv"):
                render_arxiv_list(msg["arxiv"])
                st.caption("Open the 🔎 Discover tab to add any of these papers to your library.")
            render_verification(msg.get("verification"))
            if msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask anything, e.g. Which paper gets the best accuracy on ImageNet? / شو الداتا المستخدمة؟")
    if question:
        ss.chat.append({"role": "user", "content": question})
        reply = {"role": "assistant"}
        try:
            if llm is None:
                raise RuntimeError("Add the ngrok URL in the sidebar or turn on Demo mode.")
            if agent_mode:
                if chains is None:  # no papers yet: the only useful tool is arXiv search
                    route = {"tool": "arxiv", "query": question, "reason": "No papers loaded yet, searching arXiv."}
                else:
                    with st.spinner("🤖 Agent is choosing a tool..."):
                        route = chains.route(question)
                reply["agent"] = route
                tool = route["tool"]
            else:
                if chains is None:
                    raise RuntimeError("Add papers first, or turn on Agent mode to search arXiv.")
                tool = "qa"

            ids = scope or paper_ids
            with st.spinner(f"Working ({tool})..."):
                if tool == "arxiv":
                    reply["arxiv"] = search_arxiv(route["query"], 6)
                    ss.arxiv_results = reply["arxiv"]
                    reply["content"] = f"Here are papers I found on arXiv for **{route['query']}**:"
                elif tool == "compare":
                    if len(ids) < 2:
                        raise RuntimeError("Comparison needs at least 2 papers.")
                    for pid in ids:
                        chains.extract(pid)
                    _, comparison, diffs = chains.run_full_comparison(ids)
                    ss.comparison = {"ids": ids, "comparison": comparison, "differences": diffs}
                    reply["content"] = comparison + "\n\n---\n\n" + diffs
                elif tool == "chart":
                    for pid in ids:
                        chains.facts(pid)
                    fig, df = best_chart(ids)
                    reply["fig"] = fig
                    reply["content"] = ("Here is the metric reported by the most papers. "
                                        "See the 📊 Results tab for all metrics." if fig is not None
                                        else "I could not find numeric results in these papers.")
                elif tool == "graph":
                    for pid in ids:
                        chains.facts(pid)
                    html, shared = graph_html(ids)
                    reply["graph"] = html
                    reply["content"] = ("**Shared:** " + ", ".join(s["entity"] for s in shared)) if shared \
                        else "The papers do not share methods or datasets."
                elif tool == "related_work":
                    ss.related = chains.related_work(ids)
                    reply["content"] = ss.related["text"] + "\n\n**References**\n\n" + ss.related["references"]
                else:  # qa
                    answer, docs = chains.ask(question, papers=ids)
                    reply["content"], reply["sources"] = answer, docs
                    if verify:
                        with st.spinner("Fact-checking the answer..."):
                            reply["verification"] = chains.verify(answer, docs)
        except Exception as e:
            reply["content"] = f"Error: {e}"
        ss.chat.append(reply)
        st.rerun()
