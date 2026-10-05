# Data notes: TED Talks

Findings and decisions from `notebooks/01_eda.ipynb`.

## Source

- **Dataset:** TED Talks by Rounak Banik, Kaggle (2017), https://www.kaggle.com/datasets/rounakbanik/ted-talks
- **Files:**
  - `ted_main.csv`: talk metadata, 2,550 talks
  - `transcripts.csv`: English transcripts
- **Licence:** TED content is CC BY-NC-ND 4.0. The data is not redistributed in this repo; download it with `python -m src.download_data`.
- **Coverage:** talks published June 2006 – September 2017 (some filmed as early as 1984).
- **Personal data:** speakers are real, named public figures. We use names only as metadata and don't profile individuals.

## Role in this project

| TED field | Stands in for (QUADRA-AI) | Used in |
|---|---|---|
| `transcript` | Interview transcripts | All phases |
| `tags` | Human-assigned ELSST keywords (gold labels) | Phase 1 |
| `description` | Catalogue abstract | Phase 2 |
| `occupations` | A catalogue field to extract (speaker role) | Phase 2 |
| `related_talk_ids` | "Similar datasets" ground truth | Phase 4 |

## Processed dataset

`data/processed/talks.parquet`, built by `python -m src.prepare_data`: one row per talk, **2,427 talks**.

Cleaning steps:

- **Joining files:** stripped hidden newlines from URLs, then joined talks and transcripts on URL. Duplicate transcripts were dropped.
- **IDs:** `talk_id` is the last part of the URL.
- **Text fields:**
  - Lists stored as text (`tags`, `related_talks`) are parsed into real lists.
  - Tags are lowercased and de-duplicated.
- **Dates:** Unix timestamps are converted to dates.
- **Sound notes:** bracketed notes without a colon, such as (Laughter) and (Applause), are removed from transcripts. Bracketed speech such as "(MS: No idea.)" is kept.
- **Removed talks:** talks with no transcript, or under 300 words (mostly music and performances). 2,550 → 2,427.
- **Related talks:** converted from titles to `talk_id`s, keeping only talks still in the dataset.
- **Occupations:** `speaker_occupation` is split into an `occupations` list of roles.
- **Paragraphs:** the raw transcripts lost their paragraph breaks, so sentences were glued together ("play out.I have"). These are restored as `\n\n` (~18 paragraphs per talk).

## Key findings and decisions

1. **The tags have a long tail.**
   - 416 unique tags; each talk has 1–32 (median 6). 101 tags are used in fewer than 10 talks.
   - Keeping tags with **≥ 10 talks** leaves 315 tags, covering ~97% of all tag assignments, and no talk is left without tags.
   - 11 tags are TED event or programme labels, not subjects (`tedx`, `ted fellows`, `ted brain trust`, `ted prize`, `tedyouth`, `tedmed`, `tednyc`, `ted books`, `ted en español`, `ted residency`, `ted-ed`).
   - **Decision:** Phase 1 vocabulary = tags with ≥ 10 talks, minus the TED event tags.

2. **Time split is feasible.**
   - Roughly 200–270 talks are published per year from 2008 on.
   - With a 70/15/15 split by `published_date`, only ~2% of the tags used in the test set never appear in training. 16 tags first appear in 2015 or later, e.g. `blockchain`, `crispr`, `refugees`.
   - **Decision:** time-based 70/15/15 split. Report the unseen tags as a limitation.

3. **Transcripts fit the LLM but not embedding models.**
   - 310–9,044 words, median ~2,060 (≈ 2.7k tokens; the longest ≈ 12k tokens).
   - **Decision:** keep full transcripts.
     - Phases 1–2: use the whole transcript (LLM `num_ctx` ≈ 16k).
     - Phases 3–4: chunk into ~250-word overlapping segments, because embedding models accept only ~512 tokens and thematic codes attach to passages.

4. **Occupations are messy.**
   - 1,403 unique strings and 1,247 unique roles; ~21% of speakers list several roles (one speaker, not several; only 43 talks have more than one speaker). Synonyms are common ("writer" vs "author").
   - **Decision:** in Phase 2, a predicted role is correct if it matches any role in `occupations`, using lenient matching.

5. **Descriptions are short abstracts**, 11–130 words (median ~50).
   - **Decision:** use them as the reference when evaluating generated summaries in Phase 2.

6. **Phase 3 subset.**
   - **Decision:** use talks tagged `education` (146 talks). It's a good size and a typical social-science topic.

7. **Related talks exist for every talk** (3–6 each; 93% have 6).
   - **Decision:** use them as relevance labels for "similar talks" in Phase 4. They were chosen by TED, so they're a reasonable but imperfect ground truth.

8. **Sentences are glued together.**
   - The raw data lost its paragraph breaks ("play out.I have"), affecting ~40,000 places in 98% of talks.
   - **Decision:** restored as paragraph breaks in `prepare_data` (`restore_paragraphs`). That fixes the spacing and gives natural paragraphs for chunking in Phase 3.

## Biases and limits

- All transcripts are in English. Speakers and topics lean towards the US and Europe, and towards technology, science and business.
- Mostly main TED conferences; ~18% are TEDx talks.
- Data covers 2006–2017 only.
- Tags were assigned by TED staff for discovery on its website, not by trained indexers using a thesaurus, so they are noisier than ELSST indexing.
- Results won't transfer directly to UK social-science interview data. We report them as a learning benchmark.
