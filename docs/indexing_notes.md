# Phase 1: Automatic subject indexing — notes

Task: suggest a ranked list of tags from the vocabulary (306 tags, `vocab/thesaurus.csv`) for each talk.
All methods use the same data, splits and evaluation (`src/indexing/`).

## Setup (shared by all methods)

- **Split:** time-based on `published_date` (`src/make_splits.py`).
  - train: 1,698 talks (2006 – Aug 2014)
  - validation: 364 (to Apr 2016)
  - test: 365 (Apr 2016 – Sep 2017)
- **Input text:** title + description + transcript.
- **Labels:** each talk's tags, keeping only vocabulary tags.
- **Main metric:** **nDCG@10** for ranking quality (used to choose a model). F1 is used to choose the cut-off.
- **Tagging changed over time:** test talks have ~12 tags each, train ~6 and validation ~7.
  - So on test, recall@k has a hard ceiling (e.g. 0.72 at k=8), precision is easier, and a cut-off tuned on validation is too small for test.
  - Always compare against the "perfect" row.
- **Process:** tune on validation only → retrain on train+val → evaluate on test **once**.

## Trivial baselines (test)

| k=8 | precision | recall | F1 | nDCG |
|---|---|---|---|---|
| perfect (gold tags) | 0.941 | 0.719 | 0.782 | 1.000 |
| most common train tags | 0.138 | 0.096 | 0.108 | 0.164 |
| random tags | 0.032 | 0.021 | 0.025 | 0.032 |

At k=10, the most common tags score nDCG@10 = 0.161. This is the bar every method must clearly beat.

## Method 1: TF-IDF + one-vs-rest logistic regression (`src/indexing/baseline_tfidf.py`)

**How it works:** TF-IDF features → one logistic regression per tag → tags ranked by probability.

### Tuning (validation, ranked by nDCG@10)

- **Fixed TF-IDF settings:** `sublinear_tf=True`, English stop words, `min_df=2`, `max_df=0.8`, `max_features=100k`.
- **Grid:** ngram_range {(1,1), (1,2)} × C {100, 300} × class_weight {None, balanced}. An earlier run covered C = 1–30.
- **Best:** unigrams, **C=100**, **class_weight=balanced**, nDCG@10 = **0.574**.
- **Findings:**
  - `balanced` beat `None` in every setting. It mainly lifts recall on rare tags.
  - Bigrams were worse and about 2× slower, probably because `max_features` is shared between single words and phrases.
  - The C curve flattens above about 30 (C=30: 0.569; C=100 and C=300: 0.574).

### Cut-off (validation, ranked by F1)

- **Fixed k:** best **k=8**, F1 0.402. But the curve is flat: any k from 6 to 10 gives 0.396–0.402.
- **Threshold:** best **0.1**, F1 0.397, with 6.6 tags per talk on average.
- **Chosen:** **fixed k=8**. The threshold wasn't better, and fixed k is simpler.

### Test results (trained on train+val)

| | precision@8 | recall@8 | F1@8 | nDCG@8 | nDCG@10 | recall@10 |
|---|---|---|---|---|---|---|
| TF-IDF + LR | **0.502** | **0.364** | **0.403** | **0.568** | **0.551** | 0.413 |

- About 3.5× the most-common baseline on every metric. On average, **4 of the 8 suggestions are correct**.
- It reaches about half of the possible score: F1 0.40 out of a maximum of 0.78, recall 0.36 out of a maximum of 0.72.
- **Validation → test:** F1 stayed the same (0.402 → 0.403), but precision went up and recall went down, because test talks have more gold tags.
- **Near-miss@8 = 0.63** (random gets 0.34): most wrong suggestions sit in the same broader group as a correct tag. Many are probably plausible tags TED didn't use (e.g. `creativity` on a Pixar talk tagged `art` and `film`). A human review would likely rate the model higher than F1 does.
- **Recall@8 by how often a tag appears in train+val:**
  - frequent (100+): 0.65
  - medium (30–99): 0.29
  - rare (<30): 0.14
  - unseen: 0.00

### Strengths and weaknesses

- **Strengths:**
  - fast (seconds to train)
  - easy to understand
  - evidence is available per tag: the words with the largest TF-IDF × coefficient values (not implemented yet)
- **Weaknesses:**
  - **Rare and unseen tags:** it can't predict a tag it never saw, and learns rare tags poorly. This is the main gap for the next methods (Annif, embeddings, LLM).
  - It relies on word overlap, so it misses synonyms and paraphrases.
  - Its probabilities are low overall (the best threshold is only 0.1), so they aren't calibrated confidence scores.
- **Limitations of this evaluation:**
  - One time-based split with 365 test talks and no confidence intervals. Treat differences under about 0.02 as noise.
  - TED tags are noisy gold labels.

### Files

- Tuning: `results/tfidf_lr_hparams_val.csv`, `results/tfidf_lr_cutoff_topk_val.csv`, `results/tfidf_lr_cutoff_threshold_val.csv`
- Test scores: `results/indexing_results.csv` (row `tfidf_logreg`)
- Test predictions for error analysis: `data/predictions/tfidf_logreg_test.parquet` (git-ignored)

## Not done (optional follow-ups)

- **Per-tag evidence words** (explainability) for the TF-IDF model.
- **Description-only vs full-text input:** is the abstract enough, or are transcripts needed?
