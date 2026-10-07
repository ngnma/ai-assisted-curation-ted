"""Evaluation for subject indexing. Every Phase 1 method is scored with this same code.

A prediction for one talk = a ranked list of tags, best first.
Metrics (averaged over talks):
- precision@k: of the top k suggestions, how many are correct
- recall@k:    of the talk's correct tags, how many are in the top k
- f1@k:        balance of the two
- ndcg@k:      like recall, but rewards putting correct tags near the top
- near_miss@k: of the wrong suggestions, how many share a broader concept with a correct tag
- recall@k by tag frequency band: does the method also find rare tags?

Note: test talks (2016-17) have ~12 tags each vs ~6 in train, so recall@5 can be at most ~0.5 on test.
That's why we report k=5 and k=10, and the "perfect" row shows the best possible score.

Run from the project root to see trivial baselines:  python -m src.indexing.evaluate
"""

import math
import random
from collections import Counter

import pandas as pd

from src.config import EVAL_KS, SEED
from src.indexing.data import load_indexing_data, load_vocabulary


def precision_at_k(pred: list[str], gold: set[str], k: int) -> float:
    return len(set(pred[:k]) & gold) / k


def recall_at_k(pred: list[str], gold: set[str], k: int) -> float:
    return len(set(pred[:k]) & gold) / len(gold)


def ndcg_at_k(pred: list[str], gold: set[str], k: int) -> float:
    """A correct tag at rank i earns 1/log2(i+1); divide by the best possible score."""
    dcg = sum(1 / math.log2(rank + 2) for rank, tag in enumerate(pred[:k]) if tag in gold)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return dcg / ideal


def frequency_bands(train_labels: pd.Series) -> dict[str, str]:
    """Group tags by how often they appear in the training talks."""
    counts = Counter(train_labels.explode())
    return {tag: "frequent (100+)" if n >= 100 else "medium (30-99)" if n >= 30 else "rare (<30)"
            for tag, n in counts.items()}


def evaluate(predictions: dict[str, list[str]], gold: dict[str, list[str]], k: int,
             broader_of: dict[str, str] | None = None, bands: dict[str, str] | None = None) -> dict:
    """Score ranked predictions against gold tags. Talks with no prediction count as empty."""
    rows, wrong, near = [], 0, 0
    band_hits, band_total = Counter(), Counter()

    for talk_id, gold_tags in gold.items():
        g = set(gold_tags)
        pred = list(dict.fromkeys(predictions.get(talk_id, [])))  # drop repeated tags, keep order
        p, r = precision_at_k(pred, g, k), recall_at_k(pred, g, k)
        rows.append({"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "ndcg": ndcg_at_k(pred, g, k)})

        if broader_of:  # wrong tags that are "close" (same broader concept as a gold tag)
            gold_groups = {broader_of.get(t) for t in g}
            for tag in pred[:k]:
                if tag not in g:
                    wrong += 1
                    near += broader_of.get(tag) in gold_groups

        if bands:  # recall split by how frequent the gold tag is
            for tag in g:
                band = bands.get(tag, "unseen in train")
                band_total[band] += 1
                band_hits[band] += tag in pred[:k]

    scores = {f"{m}@{k}": round(v, 3) for m, v in pd.DataFrame(rows).mean().items()}
    if broader_of:
        scores[f"near_miss@{k}"] = round(near / wrong, 3) if wrong else None
    for band in sorted(band_total):
        scores[f"recall@{k} {band}"] = round(band_hits[band] / band_total[band], 3)
    return scores


def main() -> None:
    """Sanity check with trivial baselines: every real method must beat these."""
    data = load_indexing_data()
    train, test = data["train"], data["test"]
    gold = dict(zip(test["talk_id"], test["labels"]))
    vocab = load_vocabulary()
    broader_of = dict(zip(vocab["tag"], vocab["broader"]))
    bands = frequency_bands(train["labels"])

    print({name: len(df) for name, df in data.items()}, "talks per split")

    k_max = max(EVAL_KS)
    most_common = [tag for tag, _ in Counter(train["labels"].explode()).most_common(k_max)]
    rng = random.Random(SEED)
    baselines = {
        "perfect (gold tags)": gold,  # upper bound: shows the best possible score
        "most common train tags": {t: most_common for t in gold},
        "random tags": {t: rng.sample(sorted(vocab["tag"]), k_max) for t in gold},
    }
    for k in EVAL_KS:
        results = {name: evaluate(preds, gold, k, broader_of, bands) for name, preds in baselines.items()}
        print(f"\n=== Test set, k={k} ===")
        print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
