# QUADRA-AI: what your role (Metadata Enhancement, Thematic Coding and Dataset Discovery) will probably involve

---

## 1. What the whole project is about

The UK Data Archive at Essex holds qualitative research data: interviews, focus groups, oral histories and fieldnotes. Researchers deposit it and other researchers reuse it through the UK Data Service. Today archive staff prepare this data by hand. They transcribe audio, anonymise it, check disclosure risk, write catalogue metadata and add subject keywords. This is slow and expensive.

QUADRA-AI (probably "QUAlitative DAta … AI") is building an AI-assisted curation pipeline where humans stay in the loop:

```
Audio → Transcript → Anonymised transcript + risk report → Themes/codes → Enriched metadata → Smarter catalogue search
        |_______________ Engineer A ________________|     |____________ YOU (Engineer B) ______________|
```

**What the final product should do:**

1. Transcribe interview audio automatically.
2. Detect identifiers and flag disclosive content for a reviewer.
3. Suggest themes and codes for the data.
4. Draft and improve catalogue metadata.
5. Power thematic and semantic search in the UKDS catalogue.

Experts approve every step. Nothing is fully automated.

## 2. How the work splits between the two engineers

| | Engineer A: Disclosure, Anonymisation, Transcription | **Engineer B (you): Metadata, Thematic Coding, Discovery** |
|---|---|---|
| Goal | Make the data **safe** to share | Make the data **findable and reusable** |
| Core tasks | Speech-to-text, speaker diarisation, NER for identifiers, disclosure risk scoring, pseudonymisation | Thematic coding, subject indexing, metadata generation, semantic search, recommendations |
| Key skills | ASR, NER, privacy-preserving AI | Topic modelling and LLMs, information retrieval, metadata standards |
| **Shared** | Human review interface, evaluation framework, audit logs, secure on-premises deployment, integration with archive systems | same |

The link between you: **A's anonymised transcripts are your input.**

## 3. The dataset (probably)

- **Study-level metadata** from the UKDS catalogue, in the **DDI** standard. This covers title, abstract, keywords, methodology, sample, geography, time period, and **ELSST subject terms** (the social science thesaurus the Archive maintains).
- **Qualitative documents**: interview transcripts (docx, rtf, pdf), topic guides, user guides and fieldnotes. These come from **QualiBank** (transcripts marked up in **QuDEx** XML) and **ReShare** (self-deposited data).
- **Historical human-assigned ELSST keywords.** This is the key asset, because it gives you free labelled training and test data.
- **Discover search logs** (queries and clicks), possibly, for evaluating search.
- **Constraint:** much of the data is safeguarded or controlled-access. You'll likely need **open-weight models (Llama, Qwen, Mistral, Gemma) on secure infrastructure**, with no external APIs like OpenAI.

## 4. Your tasks

### Task 0 – Landscape review and data audit (first month)

- **Problem:** "Assess existing solutions against UKDS requirements."
- **Input:** Existing tools, and a sample of the Archive's data.
- **Output:** A gap-analysis report and an evaluation benchmark (gold-standard test sets).
- **Tools to review:**
  - Qualitative analysis software with AI features: NVivo, ATLAS.ti and MAXQDA AI Assist.
  - **Annif**, an open-source subject-indexing tool from the National Library of Finland.
  - BERTopic and LLooM for finding themes.
  - OpenSearch and Elasticsearch with vector search.
- **Criteria:** Security, explainability, ELSST compatibility and cost.

### Task 1 – Automatic subject indexing (metadata enhancement)

- **Problem:** Multi-label classification over a **controlled vocabulary**. ELSST has thousands of hierarchical concepts.
- **Input:** Title, abstract and documentation, plus transcripts if needed.
- **Output:** Ranked ELSST terms with confidence scores and evidence, for a curator to approve. Possibly also CESSDA topic classes.
- **Current approaches:**
  - **Annif ensembles**: Omikuji/Parabel, MLLM lexical matching, fastText, with a neural-network ensemble on top.
  - **Embedding retrieval then reranking**: embed the text and all ELSST concepts, retrieve candidates, then rerank with a cross-encoder.
  - **Candidate retrieval plus an LLM ranker**: the LLM may only choose from retrieved vocabulary terms, so it can't invent terms. This is the state of the art from **SemEval-2025 Task 5 "LLMs4Subjects"**, which tackled exactly this problem for library subject tagging.
  - Fine-tuned encoders (DeBERTa, ModernBERT, SetFit) if there are enough labels.
- **Evaluation:** Precision, recall and F1 at k, plus nDCG, measured against historical human indexing. Also have curators rate the suggestions.

### Task 2 – Metadata generation and quality improvement

- **Problem:** Draft, complete or correct catalogue fields.
  - Abstracts and summaries.
  - Methodology: interview type, number of participants, sampling.
  - Geographic and time coverage.
  - Detecting missing or inconsistent fields.
- **Input:** Deposited documents plus the partial existing record.
- **Output:** Draft fields as schema-valid JSON or XML, each with its source evidence, for curator review.
- **Current approaches:**
  - **LLM structured extraction** with constrained decoding (vLLM guided decoding, Outlines, Instructor).
  - NER plus gazetteer linking for places (to ONS geographies).
  - Normalising dates and time expressions.
  - Local LLM summarisation with **faithfulness checks**: citing evidence, NLI-based checks such as AlignScore or SummaC.
- **Evaluation:** Field-level accuracy against curators, hallucination rate and human ratings. ROUGE is of little use here.

### Task 3 – AI-assisted thematic coding (the core qualitative task)

- **Problem:** Find themes in transcripts and code transcript segments. There are two modes:
  - **Inductive:** discover themes from scratch.
  - **Deductive:** apply an existing codebook.
- **Input:** Anonymised transcripts split by speaker turn, plus the research questions or codebook if available.
- **Output:** Themes with definitions, coded segments with supporting quotes, and document-level theme labels. These feed into metadata and search.
- **Current approaches:**
  - **BERTopic**: sentence embeddings, then UMAP, HDBSCAN and c-TF-IDF, with an LLM writing theme labels. Use LDA or STM as baselines (STM is popular in social science).
  - **LLM-based thematic analysis** following Braun & Clarke's phases:
    - initial codes per chunk,
    - then clustering and merging codes,
    - then themes (map-reduce over long transcripts).
    - Examples: **TopicGPT** (NAACL 2024), **LLooM** concept induction (CHI 2024), and LLM-in-the-loop thematic analysis.
  - **Deductive coding**: zero-shot or few-shot LLM classification against the codebook, or fine-tuned SetFit or ModernBERT once labels exist.
- **Evaluation:**
  - Agreement with human coders (Cohen's κ, Krippendorff's α).
  - Topic coherence (NPMI) and diversity.
  - Stability across runs.
  - Validation sessions with experts (the PDF requires this before the tools go into operational use).

### Task 4 – Thematic search and dataset discovery

- **Problem:** Given a natural-language query, rank datasets and possibly passages within them. An example query: "young people's experiences of unemployment in 1980s northern England".
- **Input:** The query and filters, plus an index of enriched metadata, themes and open-access transcript chunks.
- **Output:**
  - Ranked datasets with an **explanation** (matched themes, ELSST terms, snippets).
  - Facets.
  - "Similar datasets" recommendations.
- **Current approaches:**
  - **Hybrid search**: BM25 plus dense embeddings (bge-m3, e5, Qwen3-Embedding), combined with reciprocal rank fusion.
  - **Cross-encoder reranking**.
  - **Query expansion** using ELSST synonyms and broader/narrower terms, plus LLM query rewriting (HyDE).
  - Engines: OpenSearch or Elasticsearch k-NN, pgvector or Qdrant.
  - Optionally, **RAG conversational search** ("ask the catalogue") that cites datasets.
  - **"Use enhanced metadata to refine AI models"** likely means **fine-tuning the embedding model** on query–dataset pairs. The pairs can be synthetic (an LLM generates queries from the metadata, as in GPL and InPars) or come from click logs.
- **Evaluation:** Build a test collection of queries with curator relevance judgements, then measure nDCG@10, MRR and Recall@k. Follow up with user studies.

### Task 5 – Human-in-the-loop review, explainability and integration (shared with Engineer A)

- **Review UI:** Curators accept, edit or reject suggestions, using Streamlit or Gradio prototypes or Argilla or Label Studio. Their feedback is stored for active learning.
- **Audit trail:** Record the model and prompt version, inputs, outputs and the reviewer's decision (MLflow, model cards).
- **Deployment:** A FastAPI service in Docker on secure on-premises GPUs, exporting DDI XML into the UKDS catalogue.

## 5. Likely 6-month plan (October 2026 – March 2027, about 20 hours a week)

| Month | Focus |
|---|---|
| 1 | Onboarding, data audit, tool review, build the gold-standard test sets |
| 2 | ELSST indexing baseline (Annif and embeddings), metadata extraction prototype |
| 3 | Thematic coding prototype, first round of expert validation |
| 4 | Hybrid search, reranking, embedding fine-tuning on the enriched metadata |
| 5 | Review UI, integration with Engineer A's pipeline, pilot with curators |
| 6 | Evaluation report, documentation, handover and recommendations |

Sources:
- [UK Data Archive – Metadata standards](https://www.data-archive.ac.uk/managing-data/standards-and-procedures/metadata-standards/)
- [QuDEx](https://www.data-archive.ac.uk/managing-data/standards-and-procedures/metadata-standards/qudex/)
- [Searching QualiBank](https://ukdataservice.ac.uk/help/searching-data/searching-qualibank/)
- [UKDS – Metadata](https://ukdataservice.ac.uk/manage-data/document/metadata)
- [What's behind the UK QualiBank (Zenodo)](https://zenodo.org/records/3780770)
- [CAQDAS – AI Tools for QDA](https://www.surrey.ac.uk/computer-assisted-qualitative-data-analysis/qual-ai/ai-tools-qda)