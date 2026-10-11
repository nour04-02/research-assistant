# 🔬 AI Research Assistant

>  **Note: This project is an MVP (Minimum Viable Product).**
> It is a working first version built to prove the idea end to end. Some operations are currently **slow**
> (loading the embedding and reranker models for the first time, and running several LLM calls per paper
> on a free Kaggle GPU through ngrok). Improving speed is the top priority for the next versions,
> and I will keep working on the project to make it faster, more accurate and easier to use.
> See the [Roadmap](#-roadmap) section for what is planned.

An AI assistant for researchers that helps you **find, read, compare and write about scientific papers**.
It combines **Retrieval-Augmented Generation (RAG)**, **hybrid multi-document retrieval**, **LLM chains**,
a **router agent** and an automatic **hallucination check**, all in a simple **Streamlit** interface.

---

## Table of Contents
- [The Problem](#-the-problem)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation and Running](#-installation-and-running)
- [How to Use](#-how-to-use)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)

---

##  The Problem
Reviewing literature is slow. A researcher has to search for papers, read each one, pull out the method,
dataset and results, compare them side by side, and finally write a related work section with correct citations.
This assistant automates most of that workflow, while showing the sources behind every answer so the
researcher can verify it.

---

##  Features

| Tab | What it does |
|---|---|
|  **Discover** | Search arXiv by topic, select papers, and download + index them automatically with clean metadata (title, authors, year). |
|  **Extract** | Extracts 6 structured fields from every paper: **Paper, Methodology, Dataset, Results, Advantages, Limitations**. Export as CSV. |
|  **Compare** | A sequential chain: *Extract → Compare → Differences*. Produces an overview, common ground, methodology comparison, key differences and **research gaps**. Export as Markdown. |
|  **Results** | Extracts numeric results (model, dataset, metric, value) as JSON and draws an interactive bar chart to compare papers on the same metric. |
|  **Graph** | An interactive knowledge graph showing how papers connect through shared **methods, datasets and tasks**. |
|  **Related Work** | Writes an academic *Related Work* draft with numbered citations, plus a references list and a ready **BibTeX** file. |
|  **Ask** | Chat with all papers at once. Answers include page-level citations and the retrieved source chunks. |

**Smart features inside Ask:**
-  **Agent mode:** a router chain reads your request and picks the right tool automatically
  (Q&A, compare, chart, graph, related work, or arXiv search).
-  **Fact-check:** a second chain checks every sentence of the answer against the retrieved sources and labels it
   supported, 🟡 partial or ⚠️ unsupported, with an overall **faithfulness score**.
-  **Arabic support:** ask in Arabic about English papers (multilingual embeddings + multilingual reranker),
  and get answers in English or Arabic.

---

## ⚙️ How It Works

```
                 ┌──────────────────────────── Local machine (Streamlit) ────────────────────────────┐
  PDFs / arXiv → │ PDF parsing → section detection → chunking → FAISS (dense) + BM25 (sparse)         │
                 │                                                       ↓                            │
   User query  → │ Hybrid search → Reciprocal Rank Fusion → Cross-encoder reranker → top chunks       │
                 │                                                       ↓                            │
                 │ LangChain chains (Extract / Facts / Compare / Differences / Q&A / Verify / Router) │
                 └───────────────────────────────────────────────┬───────────────────────────────────┘
                                                                 │ HTTPS (ngrok)
                                                  ┌──────────────▼──────────────┐
                                                  │ Kaggle GPU: FastAPI server  │
                                                  │ Mistral-Nemo-Instruct 4-bit │
                                                  └─────────────────────────────┘
```

1. **Ingestion:** each PDF is parsed page by page. Section headings (Abstract, Methods, Dataset, Results,
   Limitations, ...) are detected, and everything after *References* is dropped. Every chunk keeps its
   paper, section and page number as metadata.
2. **Hybrid retrieval:** every query runs through **FAISS** (semantic meaning) and **BM25** (exact keywords like
   `ResNet-50`). The two ranked lists are merged with **Reciprocal Rank Fusion**, then a **cross-encoder reranker**
   re-scores the best candidates. Retrieval is done **per paper**, so no paper is ignored.
3. **Chains:** prompt templates turn the retrieved context into structured outputs. JSON outputs are parsed
   with LangChain's `StructuredOutputParser` plus a robust fallback parser.
4. **LLM:** `Mistral-Nemo-Instruct-2407` quantized to 4-bit runs on a free Kaggle GPU and is exposed as an API
   with FastAPI + ngrok. A custom LangChain `LLM` class calls it from the app.

---

##  Tech Stack
- **LLM:** Mistral-Nemo-Instruct-2407 (4-bit, bitsandbytes), Transformers
- **Serving:** FastAPI, Uvicorn, ngrok (on Kaggle GPU)
- **RAG:** LangChain, FAISS, BM25 (`rank-bm25`), Sentence-Transformers (embeddings + cross-encoder reranker)
- **Data sources:** PDF upload (`pypdf`), arXiv API (`arxiv`)
- **UI and visualization:** Streamlit, Plotly, PyVis

---

##  Project Structure

```
research_assistant/
├── app.py                 # Streamlit interface (all tabs, agent dispatch, rendering)
├── rag.py                 # PDF parsing, section detection, chunking, hybrid search + reranking
├── chains.py              # All prompts and chains: extract, facts, compare, differences,
│                          # Q&A, verify (fact-check), related work, router agent
├── llm_client.py          # LangChain LLM that calls the Kaggle server + MockLLM for demo mode
├── arxiv_client.py        # arXiv search and PDF download
├── graph.py               # Knowledge graph builder (PyVis)
├── kaggle_server.ipynb    # Notebook that serves the LLM from a Kaggle GPU via FastAPI + ngrok
├── requirements.txt       # Python dependencies for the local app
└── .streamlit/config.toml # Streamlit settings
```

---

##  Installation and Running

### Part 1: Start the LLM server on Kaggle
1. Create a new notebook on [Kaggle](https://www.kaggle.com) and choose **File > Import Notebook** → `kaggle_server.ipynb`.
2. In **Settings**: set **Accelerator = GPU T4** and **Internet = On**.
3. Get a free auth token from [dashboard.ngrok.com](https://dashboard.ngrok.com) and paste it into `NGROK_TOKEN`.
4. Click **Run All**. The first run downloads the model and takes about 10 minutes.
5. Copy the URL printed in the last cell (`https://xxxx.ngrok-free.app`). Keep the notebook running.



### Part 2: Run the app locally
Requires **Python 3.10+**.

```bash
git clone https://github.com/nour04-02/research-assistant.git
cd research-assistant

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Part 3: Connect
Paste the ngrok URL in the sidebar and click **Test connection**.

> 💡 **Demo mode:** turn it on in the sidebar to try the whole interface with a fake model, no GPU needed.

---

##  How to Use
1. **Add papers:** upload PDFs from the sidebar, or search arXiv in the 🔎 **Discover** tab.
2. **Extract:** get the 6-field summary table for every paper.
3. **Compare:** choose 2 or more papers to get the comparison, differences and research gaps.
4. **Results:** extract the numbers and compare them on a chart.
5. **Graph:** see which methods and datasets the papers share.
6. **Related Work:** generate a draft section with citations and BibTeX.
7. **Ask:** ask anything (in English or Arabic). Keep **Agent mode** and **Fact-check** on.

Example questions:
- *Which paper achieves the best accuracy on ImageNet?*
- *What datasets do these papers use?*
- *شو الفرق بين الميثودولوجي بالورقتين؟*

---

##  Known Limitations
This is an MVP, so there are known issues I am aware of:

- **Speed:** the main limitation right now.
  - The first run downloads the embedding model and the reranker, which can take a while on slow connections.
  - Embeddings and reranking run on the CPU of the local machine.
  - Each paper needs more than one LLM call (extraction + facts), and every call goes over the network to Kaggle.
- **Kaggle sessions:** the free GPU session can stop after inactivity, and the ngrok URL changes on every run.
- **PDF parsing:** scanned PDFs (images) are not supported, and tables or two-column layouts are sometimes read imperfectly.
- **LLM accuracy**: The *Related Work* output is a draft, not final text.

---

##  Roadmap
Planned improvements for the next versions:

**Speed (top priority)**
- [ ] Stream LLM responses token by token instead of waiting for the full answer
- [ ] Run extraction for several papers in parallel, and merge the extraction + facts calls into one
- [ ] Cache embeddings, indexes and extractions on disk so papers are not reprocessed every session

**Quality**
- [ ] Better PDF parsing for tables, figures and two-column layouts, plus OCR for scanned papers
- [ ] Evaluation set to measure retrieval and answer quality (faithfulness and answer relevance)
- [ ] Smarter chunking by sections and paragraphs

**Product**
- [ ] Export full literature review reports (Word / PDF)
- [ ] More paper sources (Semantic Scholar, PubMed)

