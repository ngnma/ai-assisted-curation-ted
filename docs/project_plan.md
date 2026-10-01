# Project plan: TED-QUADRA, a practice version of the QUADRA-AI pipeline

**Goal:** Build a small version of your QUADRA-AI pipeline (Engineer B side) on the TED Talks dataset: automatic tagging, metadata generation, thematic coding, search and human review. Every component is evaluated properly and lives in a clean, reproducible GitHub repo. To practise under UKDA's "no external APIs" constraint, use only open-weight LLMs that run locally (Qwen2.5 or Llama 3.x through Ollama).

**Why TED works as a stand-in:** TED transcripts play the role of interview transcripts, and the talk details (title, description, speaker, event) play the role of catalogue metadata. The human-assigned `tags` act like UKDA's historical ELSST keywords, so you get labelled data for free. The `related_talks` field gives you ground truth for "similar datasets".

## Phase 0 — Setup, data audit and test sets (Days 1–2)
Before any modelling, set up the repo and get to know the data. The repo needs a src/ package layout, configs, tests, and the data kept out of git because of the TED licence.

TED tags are a flat list. UKDA uses ELSST, a **thesaurus**: a controlled list of allowed terms where each term also has broader and narrower terms. To imitate this, group the ~400 tags under about 20 broader concepts. The LLM suggests the groups and you approve them. Save the result in **SKOS**, the standard file format for thesauri.

Finally, split the data by publish date into train, validation and test sets, and freeze the test set. This frozen set is your **gold standard**: you only ever test on it.

**Output:** a working repo, a data card, a mini-thesaurus and frozen data splits.

## Phase 1 — Automatic subject indexing (Days 3–4)
**Subject indexing** means assigning terms from a controlled vocabulary to a document so people can find it. Here it means predicting each talk's tags from its title, description and transcript. Technically this is **multi-label classification**: each talk has several correct tags, and most tags are rare. Curators at UKDA do this by hand today, so it is the most direct time saving.

You build it in three steps:
1. **Baseline:** TF-IDF features with one logistic regression per tag.
2. **Annif:** the open-source indexing tool UKDA is reviewing, combining several of its algorithms.
3. **Retrieve-then-rank:** embeddings (vectors that represent meaning) retrieve candidate tags, then a local LLM picks the best ones *only from those candidates*, so it can't invent terms. This is the current state of the art from the SemEval-2025 LLMs4Subjects task.

Compare the three on the test set with Precision/Recall/F1@k and nDCG, which reward putting the correct tags near the top of the list.

**Output:** for each talk, a ranked list of tags with confidence scores and supporting evidence, plus a comparison table of the three approaches.

## Phase 2 — Metadata generation and quality checks (Days 5–6)
Curators also write catalogue records by hand. In this phase a local LLM drafts a record from each transcript: abstract, speaker role, places, time periods and main topics.

The key technique is **structured extraction**. You define the record as a fixed schema (Pydantic/JSON), and a tool such as Instructor or Outlines forces the LLM's output to be valid against it. That makes the output machine-usable, like the DDI XML UKDA uses.

The main risk is **hallucination**, meaning content the transcript doesn't support. Two safeguards address it. First, every field must quote its evidence. Second, an **NLI** model (a model that checks whether one text logically supports another) verifies each claim against the source. Alongside the LLM, a simple rule-based checker flags fields that are missing or inconsistent.

To evaluate, compare the extracted fields with fields TED already has (e.g. speaker occupation), measure the hallucination rate, and rate a small sample yourself.

**Output:** schema-valid draft records with evidence, a quality-issue report and an XML export.

## Phase 3 — AI-assisted thematic coding (Days 7–10)
This is the heart of your role. **Thematic analysis** is how qualitative researchers read transcripts. They attach short labels, called **codes**, to passages, then group the codes into broader **themes**.

Work on a focused subset of about 100 talks on one subject (e.g. education), split into short segments, and do it in two modes:
- **Inductive mode** (discover themes from scratch). Run **BERTopic**, which embeds segments, clusters them and has an LLM name each cluster, with LDA as the classic baseline. Then run an LLM pipeline that follows Braun & Clarke's method: code each chunk, merge similar codes, and form themes with definitions and quotes.
- **Deductive mode** (apply a known list of codes). You review the discovered themes and turn them into a **codebook**: the list of codes with definitions. You hand-code about 150 segments yourself, and an LLM (and a small SetFit model) codes them too.

Measure agreement between your coding and the AI's with **Cohen's κ / Krippendorff's α**, the standard measures researchers use to compare human coders. Also check topic coherence and whether results stay stable across runs.

**Output:** a codebook, coded segments with quotes, theme labels for each talk, and an agreement report.

## Phase 4 — Thematic search and dataset discovery (Days 11–12)
Here everything produced so far (tags, metadata, themes) feeds a search engine. A user types a plain-language query such as "how schools kill creativity" and gets ranked talks back, each with the reason it matched.

You build **hybrid search**:
- **BM25** (classic keyword matching) and **dense retrieval** (matching by embedding meaning) each produce a ranking.
- **Reciprocal Rank Fusion** merges the two rankings.
- A **cross-encoder** reranks the top results more accurately.
- **Query expansion** with your thesaurus adds broader and related terms.

A "similar talks" feature uses embedding similarity and is checked against TED's own related talks. To evaluate, write about 40 test queries with relevance judgements and measure nDCG@10, MRR and Recall. An ablation table shows how much each component (and the enriched metadata) improves the results.

**Output:** a working search module with explanations, and a results table.

## Phase 5 — Human-in-the-loop review and integration (Days 13–14)
At UKDA nothing is fully automated: a curator approves every suggestion. In this phase you build a small **Streamlit** app where a reviewer accepts, edits or rejects the AI's tags, metadata and codes.

Every decision is saved in an **audit log** together with the model and prompt version that produced it, so you can always trace who decided what and why. The corrections are fed back to retrain the Phase 1 model, a simple version of **active learning**. All components are exposed through a **FastAPI** service in Docker, with experiments tracked in MLflow.

**Output:** a review app, an audit log, a packaged API, and model cards documenting each model's purpose and limits.

## Phase 6 — Wrap-up (alongside Phase 5)
Write a README with an architecture diagram, results from every phase, an ethics and limitations section, and what you would change for real UKDA data. The ethics section should cover the licence, real speakers' names, TED's Western/English bias, and LLM hallucination.

**Output:** a portfolio-ready GitHub repo and a short evaluation report.
