# Phases List
Phase 0: Setup, data audit and test sets: done
1. Project skeleton and tools (simplified later to requirements.txt and a lighter folder structure) ✓
2. Config and reproducibility ✓
3. Download the data ✓
4. Load and clean ✓
5. EDA and data notes (the data card was merged into docs/data_notes.md) ✓
6. Tag vocabulary and mini-thesaurus ✓

Phase 1: Automatic subject indexing: in progress
1. Split and labels ✓
2. Evaluation first ✓
3. TF-IDF + logistic regression baseline ✓
4. Annif
5. Embeddings
6. Retrieve, then let the LLM rank (state of the art)
7. Compare and analyse errors

Phase 2: Metadata generation and quality checks

Phase 3: AI-assisted thematic coding

Phase 4: Thematic search and dataset discovery

Phase 5: Human review and integration

Phase 6: Wrap-up

---
# Instructions
# > 1. Phase 1: Automatic subject indexing: in progress
## >> 1.3. Step 3: TF-IDF + logistic regression baseline

**Why this step:** this is your reference point. It's fast and transparent, and it's often surprisingly strong. Every later method (Annif, embeddings, LLM) has to beat it clearly to justify being slower and more complex. Expect it to be beaten, but not by as much as you might think.

**The pipeline**

1. **Load** the train, validation and test sets with `load_indexing_data()`.
2. **Labels → matrix:** turn each talk's tag list into a 0/1 matrix with one column per tag.
3. **Text → features:** turn each talk's text into TF-IDF vectors. A word's weight is high when it's frequent in this talk but rare across all talks.
4. **Train one logistic regression per tag**, each one answering "does this talk have tag X: yes or no?"
5. **Rank:** for each talk, sort the tags by predicted probability. The top 10 are your suggestions, and the probability is the confidence score.
6. **Tune** a few settings on the validation set, retrain the final model, and evaluate on the test set **once**.
7. **Save** the results and predictions.

**Where the code goes**

- `src/indexing/baseline_tfidf.py`: training, tuning, prediction and evaluation, as one runnable script.
- A small committed results folder (e.g. `results/indexing_results.csv`) with one row per method. Every later method adds its own row, which builds your comparison table.
- Saving test predictions (talk → ranked tags with scores) is useful for the error analysis in Step 7. They can go under `data/` (git-ignored).

Add `scikit-learn` to `requirements.txt`.

**Tools in scikit-learn**

- `MultiLabelBinarizer`: tag lists → 0/1 matrix. **Fit it on the training labels.** Tags that never appear in training can't be predicted. That's a known limitation (~2% of test tags), so don't hide it.
- `TfidfVectorizer`: **fit on the training texts only**, then only transform the validation and test texts. Fitting on all the data leaks information from the future.
- `OneVsRestClassifier(LogisticRegression(...))`: one classifier per tag.
- `predict_proba`: use it for **ranking**. **Don't use `predict()`**: it applies a 0.5 cut-off, and with rare tags almost nothing passes, so most talks would get zero tags. This is the most common mistake here.
- `numpy.argsort` on the probability rows gives the top-k tags. Map column positions back to names with `mlb.classes_`.

**The settings that matter**

| Setting | Try | Why |
|---|---|---|
| `TfidfVectorizer.sublinear_tf` | `True` | Long transcripts repeat words. Log-scaling stops repeated words from dominating. |
| `min_df` / `max_df` | ~2–5 / ~0.5–0.8 | Drops typos and very rare words, and words found in almost every talk. |
| `stop_words` | `"english"` | Removes "the", "and" and similar. |
| `ngram_range` | (1,1) vs (1,2) | Two-word phrases ("climate change") help, but make the model larger and slower. |
| `max_features` | ~50k–100k | Keeps memory and training time manageable. |
| `LogisticRegression C` | grid of e.g. 1, 4, 10, 30 | The main setting. TF-IDF data usually needs a fairly large C, meaning weaker regularisation. |
| `class_weight` | `None` vs `"balanced"` | `"balanced"` usually improves recall on rare tags but lowers precision. Measure it; don't assume. |
| solver | `liblinear` or `saga` | Both work for this size. Set `max_iter` high enough to avoid convergence warnings. |

Keep the grid small, around 8–12 combinations. You're building a baseline, not winning a competition.

**Design choices**

**1. How to tune.** Write a **simple loop** over the settings, scoring each one on the validation set with your own `evaluate()`. Don't use `GridSearchCV`: its default cross-validation shuffles talks randomly, which breaks your time-based split, and it doesn't know your ranking metrics. Choose one main metric before you start. I'd use **nDCG@10**, with F1@5 as a second check.

**2. Final model.** Once the best settings are chosen, retrain on **train + validation** and then evaluate on test. With a time-based split, the validation talks are the most recent ones you have, and they're the closest to the test talks. Test once and don't adjust anything after seeing test scores.

**3. Input text.** You currently join title, description and transcript into one text. The transcript is about 40 times longer, so it dominates. It's worth one quick experiment: **description only** vs **full text** on the validation set. This mirrors a real UKDA question: is the catalogue abstract enough, or do we need the full transcripts? Either result is a useful finding.

**4. Explainability (optional, cheap).** UKDA wants evidence for every suggestion. With a linear model, the evidence for a tag is the words that contributed most: the talk's TF-IDF values multiplied by that tag's coefficients. One small function gives you "suggested *education*: school, teachers, students". Later methods can't do this as easily, which is a point in the baseline's favour.

**What a good result looks like**

The test set has about 12 tags per talk, so compare at **k=10**:

- **The bar to beat:** the "most common tags" baseline scored nDCG@10 ≈ 0.16.
- **A reasonable baseline:** nDCG@10 around **0.35–0.50**, and clearly higher recall on frequent tags than on rare ones. This range is a rough expectation, not a guarantee.
- **Rare tags:** recall close to 0 is normal. This is exactly the weakness the later methods should address.
- **Warning signs:**
  - Below about 0.25 usually means a bug, such as tag columns mismatched with names, or using `predict()`.
  - Above about 0.8 usually means leakage, for example fitting TF-IDF on all the data, or tags slipping into the text.

## >> Step 4: Annif

**What Annif is (briefly)**

**Annif** is an open-source toolkit for automatic subject indexing, built by the National Library of Finland. UKDA is reviewing it for exactly your task. You give it three things:

- **a vocabulary:** your SKOS thesaurus
- **training documents:** text + correct subjects
- **"projects":** a project = one algorithm (a "backend") + its settings, defined in a config file

Annif then trains models and **suggests subjects with scores** for new text. You drive it from the command line: `annif load-vocab`, `annif train`, `annif suggest` / `annif index`, `annif eval`.

**The key concept it teaches** is that subject indexing algorithms come in two families:

- **Associative** (`tfidf`, `omikuji`, `fasttext`): learn from examples "texts like this got tag X". Your logistic regression is also in this family. They're strong on frequent tags and can't predict a tag they never saw.
- **Lexical** (`mllm`): look for the tag's **name or synonyms in the text** itself, plus a small learned model on top. They can find rare and even unseen tags, but they're blind to topics that aren't named in the words.

Annif's main recommendation is an **ensemble** that combines both families. That's exactly what should fix your baseline's weakness: rare tags at 0.14 and unseen tags at 0.

**How you'll use it:** train 3 backends and 1 ensemble on the same train/validation/test split, read Annif's suggestions back into Python, and score them with **your own `evaluate()`**. Then they're directly comparable with the logistic regression.

**Steps**

1. **Separate environment for Annif.**
2. **Prepare its inputs:** the vocabulary and the training/validation/test documents.
3. **Define the projects** in `projects.cfg`.
4. **Train each backend** on train and run a quick sanity check.
5. **Score each on validation with your `evaluate()`**, and tune a little.
6. **Build the ensemble and tune its weights.**
7. **Cut-off → retrain on train+val → test once → save** (same as Step 3), and add to the notes.

**1. Environment**

Annif doesn't support Python 3.14. When I checked, its package required **Python < 3.13**, and your venv is 3.14. Two options:

- **A second venv with Python 3.12** (recommended): install Python 3.12 with Homebrew and create a separate venv, e.g. `.venv-annif`, just for Annif. Your main venv stays as it is.
- **The official Annif Docker image.** It's cleaner, but it adds Docker to the learning load for now.

Install `annif` with the **`omikuji`** extra. Skip `fasttext` and `nn` (TensorFlow): they often fail to install on Macs, and you don't need them. Check that it works with `annif --version`. Pin the Annif version in a small `requirements-annif.txt`.

**2. Prepare the inputs**

**Folder layout:**

```
annif/
  projects.cfg     # project definitions (committed)
  corpus/          # generated documents (git-ignored)
  data/            # Annif's own models and vocab store (git-ignored)
src/indexing/annif_prep.py   # writes the corpus files from your splits
src/indexing/annif_eval.py   # reads Annif's suggestions, evaluates, saves results
```

Annif stores its models in a `data/` folder **under the directory you run it from**. Run it from inside `annif/` so it doesn't mix with your project's `data/`.

**Vocabulary:** load `vocab/ted_thesaurus.ttl` with `annif load-vocab`.

- **Trap:** your SKOS file also contains the **20 group concepts** ("health and medicine"…). Annif will treat them as subjects and may suggest them, but they're never in the gold tags. Keep them, because MLLM uses the hierarchy, but **drop group URIs from the suggestions before evaluating**.
- Annif identifies subjects by **URI**. Write a `uri → tag` mapping, using the same `tag/...` scheme as `export_skos.py`, so you can translate suggestions back to tag names.

**Documents** (in `annif_prep.py`, using `load_indexing_data()`):

- **Training data:** a TSV with one talk per line, `text<TAB><uri1> <uri2> …`. The text must be on **one line**, so replace newlines and tabs with spaces. Write `train.tsv`, and later `train_val.tsv`.
- **Documents to predict:** a **folder corpus** with one `talk_id.txt` file per talk, for validation and test. `annif index` writes a suggestions file next to each one, and the file name gives you the talk_id back.

**3. Projects (`projects.cfg`)**

One section per project. Each project has a vocab, a language (`en`), an analyzer (`snowball(english)`, which stems words), a backend, and a `limit` (e.g. 20 suggestions).

| Project | Backend | Role |
|---|---|---|
| `ted-tfidf` | `tfidf` | Simple associative baseline. It mainly tells you whether the setup works. |
| `ted-mllm` | `mllm` | Lexical: should help rare and unseen tags |
| `ted-omikuji` | `omikuji` | Strong associative model for large label sets (a tree of classifiers, Parabel/Bonsai style) |
| `ted-ensemble` | `ensemble` | Weighted average of the three |

Start every backend with its **default settings**. Only tune after you see results.

**4. Train and check**

`annif train <project> corpus/train.tsv` for each backend. Then run a quick `annif eval` on validation to catch setup errors, such as every score being near zero.

**Don't put Annif's own `eval` numbers into your comparison table.** Its metric definitions and averaging differ from yours. Use them only as a sanity check.

**5. Score on validation with your `evaluate()`**

- Run `annif index` on the validation folder with a high `limit` (20) and threshold 0, so you get the full ranked list.
- In `annif_eval.py`, read the suggestion files, map URIs to tags, drop group concepts, and build the same `{talk_id: [ranked tags]}` dictionary as before. Then call `evaluate()` with k=10, plus the bands and near-miss.
- **Compare each backend with the logistic regression on validation** (nDCG@10 = 0.574). Look closely at **recall on rare and unseen tags**, which is what you're hoping Annif improves.

**Light tuning only:**

- **MLLM:** start with the defaults. Its main limitation here is your thesaurus: tags have only a `prefLabel` and **no synonyms (`altLabel`)**. For example, "climate change" won't match "global warming". Adding a few synonyms for frequent tags is the most realistic improvement, and it's exactly what ELSST provides. Make it an optional experiment.
- **Omikuji:** try defaults, then one Bonsai-style variant: `cluster_balanced=False`, `cluster_k=100`, `max_depth=3`.
- **tfidf:** nothing to tune.

**6. Ensemble**

- Use the simple **`ensemble`** backend with weights per source, e.g. `ted-omikuji:2,ted-mllm:1,ted-tfidf:1`.
- Tune the weights on validation, either with a small manual grid scored by your `evaluate()`, or with `annif hyperopt`. You don't need `nn_ensemble`: it requires TensorFlow and **its own separate training data**, and you don't have enough data to spare for that.
- **Design point:** ensembles help most when their members make **different mistakes**. Check whether MLLM finds rare tags that Omikuji misses. If it does, the ensemble should beat both.

**7. Final steps (same pattern as Step 3)**

1. **Cut-off:** tune fixed k or a threshold on the validation suggestions, by F1. You can reuse your functions, since they only need ranked lists or scores. `annif optimize` does a similar search, which makes it a useful cross-check.
2. **Retrain** the chosen backends and ensemble on **train+val**, index test **once**, and evaluate.
3. **Save:**
   - one row per Annif method in `results/indexing_results.csv` (e.g. `annif_mllm`, `annif_omikuji`, `annif_ensemble`), with settings and the cut-off
   - predictions in `data/predictions/annif_ensemble_test.parquet`
4. **Notes:** add an Annif section to `docs/indexing_notes.md`.

**What to expect**

- **`tfidf`:** probably **below** your logistic regression. That's normal; it's a much simpler model.
- **`mllm`:** lower overall, but **clearly better recall on rare tags**, and the only method able to get **unseen** tags above 0.
- **`omikuji`:** around the logistic regression level. With only about 1,700 training talks it may not beat it. Associative methods need many examples per tag.
- **Ensemble:** the best Annif result, probably **similar to the logistic regression or a little better** on nDCG@10, and **better on rare tags**.
- **Warning signs:**
  - nDCG@10 near 0 for everything means the URIs don't line up, or you're evaluating group concepts.
  - MLLM scoring exactly 0 on rare tags means the labels aren't being matched (check the language/analyzer settings).




