"""Phase 1 baseline: TF-IDF + one-vs-rest logistic regression.
Step 1 here: tune settings on the validation set.
Run from the project root:  python -m src.indexing.baseline_tfidf
"""

import itertools
import time
import ast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import ROOT, SEED
from src.indexing.data import load_indexing_data, load_vocabulary
from src.indexing.evaluate import evaluate, frequency_bands

from joblib import parallel_config



# TF-IDF settings
TFIDF_FIXED = dict(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.8, max_features=100_000)
NGRAM_RANGES = [(1, 1), (1, 2)]

# Logistic regression settings
C_VALUES = [100, 300]
CLASS_WEIGHTS = [None, "balanced"]

# # TEMP -> DELETE
# NGRAM_RANGES = [(1, 1)]
# C_VALUES = [100, 300]
# CLASS_WEIGHTS = ["balanced"]


# Top-k settings -> for tuning cut-off threshold
FIXED_K = list(range(1, 15))  # method 1: fixed k
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.5, 0.8]  # method 2: thresholding
MIN_TAGS, MAX_TAGS = 1, 20  # thresholding: every talk gets at least 1 and at most 20 tags


def rank_tags(proba: np.ndarray, classes: np.ndarray, k: int) -> list[list[str]]:
    """Top-k tag names per talk, highest probability first."""
    top = np.argsort(-proba, axis=1)[:, :k]
    return [list(classes[row]) for row in top]

def select_by_threshold(proba: np.ndarray, classes: np.ndarray, threshold: float) -> list[list[str]]:
    """Tags with probability >= threshold per talk, highest first (at least MIN_TAGS, at most MAX_TAGS)."""
    order = np.argsort(-proba, axis=1)
    selected = []
    for row_proba, row_order in zip(proba, order):
        n = int((row_proba >= threshold).sum())  # how many tags pass the threshold
        n = min(max(n, MIN_TAGS), MAX_TAGS)
        selected.append(list(classes[row_order[:n]]))
    return selected

def predict(ngram_range, C, class_weight, train_df, val_df) -> tuple[list[list[str]], MultiLabelBinarizer]:
    """Train TF-IDF + logistic regression on the training set and predict on the validation set."""

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df["labels"])

    tfidf = TfidfVectorizer(ngram_range=ngram_range, **TFIDF_FIXED)
    X_train = tfidf.fit_transform(train_df["text"])  # fit on train only
    X_val = tfidf.transform(val_df["text"])
    X_train.sort_indices()
    X_val.sort_indices()

    clf = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=1000, random_state=SEED)
    with parallel_config(max_nbytes=None):
        model = OneVsRestClassifier(clf, n_jobs=-1).fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    return proba, mlb.classes_

def tune(train_df, val_df, broader_of, bands) -> pd.DataFrame:
    """Tune hyperparameters of TF-IDF + logistic regression on the validation set."""

    gold_val = dict(zip(val_df["talk_id"], val_df["labels"]))
    results = []
    k = 10 # Top k tags to predict for each talk in the validation set

    for ngram_range in NGRAM_RANGES:
        for C, class_weight in itertools.product(C_VALUES, CLASS_WEIGHTS):
            start = time.time()
            proba, classes = predict(ngram_range, C, class_weight, train_df, val_df)
            ranked = rank_tags(proba, classes, k)
            scores = evaluate(dict(zip(val_df["talk_id"], ranked)), gold_val, k, broader_of, bands)

            row = {"k": k, "ngram_range": ngram_range, "C": C, "class_weight": class_weight}
            row.update(scores)
            row["seconds"] = round(time.time() - start, 1)
            results.append(row)

            print(f"ngram_range={ngram_range} C={C} class_weight={class_weight} -> ndcg@k={scores["ndcg@k"]}")
            
    return pd.DataFrame(results).sort_values("ndcg@k", ascending=False)


def tune_cutoff_fixed_k(train_df, val_df, broader_of, bands, best_row) -> pd.DataFrame:
    """Tune cut-off threshold for top-k predictions on the validation set."""

    gold_val = dict(zip(val_df["talk_id"], val_df["labels"]))
    results = []
    class_weight = None if pd.isna(best_row.class_weight) else best_row.class_weight
    proba, classes = predict(ast.literal_eval(best_row.ngram_range), best_row.C, class_weight, train_df, val_df)
    ranked = rank_tags(proba, classes, k=max(FIXED_K))

    for k in FIXED_K:
        scores = evaluate(dict(zip(val_df["talk_id"], ranked)), gold_val, k, broader_of, bands)
        row = {"k": k}
        row.update(scores)
        results.append(row)
        print(f"k={k} -> f1@k={scores["f1@k"]}")

    return pd.DataFrame(results).sort_values("f1@k", ascending=False)

def tune_cutoff_thresholding(train_df, val_df, broader_of, bands, best_row) -> pd.DataFrame:
    """Tune cut-off threshold for top-k predictions on the validation set (method 2: thresholding)."""

    gold_val = dict(zip(val_df["talk_id"], val_df["labels"]))
    results = []
    class_weight = None if pd.isna(best_row.class_weight) else best_row.class_weight
    proba, classes = predict(ast.literal_eval(best_row.ngram_range), best_row.C, class_weight, train_df, val_df)

    for threshold in THRESHOLDS:
        selected = select_by_threshold(proba, classes, threshold)
        scores = evaluate(dict(zip(val_df["talk_id"], selected)), gold_val, None, broader_of, bands)  # k=None: score each talk's whole set
        row = {"cutoff_threshold": round(float(threshold), 2),
               "avg_suggested_tags": round(float(np.mean([len(tags) for tags in selected])), 1)}
        row.update(scores)
        results.append(row)
        print(f"threshold={row['cutoff_threshold']} -> f1@k={scores['f1@k']}, avg tags={row['avg_suggested_tags']}")

    return pd.DataFrame(results).sort_values("f1@k", ascending=False)

def tune_k(train_df, val_df, broader_of, bands, best_row, fixed_k_file, threshold_file) -> dict:
    """Tune the cut-off on the validation set with both methods, compare them and return the best one.

    Returns e.g. {"method": "fixed_k", "value": 8, "f1": 0.402, "avg_suggested_tags": 8}.
    """
    # method 1: fixed k
    if not fixed_k_file.exists():
        print("Tuning cut-off threshold for top-k predictions on the validation set (method 1: fixed K)...")
        cutoff_results = tune_cutoff_fixed_k(train_df, val_df, broader_of, bands, best_row)
        cutoff_results.to_csv(fixed_k_file, index=False)

    # method 2: thresholding
    if not threshold_file.exists():
        print("Tuning cut-off threshold for top-k predictions on the validation set (method 2: thresholding)...")
        threshold_results = tune_cutoff_thresholding(train_df, val_df, broader_of, bands, best_row)
        threshold_results.to_csv(threshold_file, index=False)

    # compare: best row of each method by F1 on validation
    best_k = pd.read_csv(fixed_k_file).sort_values("f1@k", ascending=False).iloc[0]
    best_t = pd.read_csv(threshold_file).sort_values("f1@k", ascending=False).iloc[0]

    print("\n=== Cut-off comparison (validation) ===")
    print(f"Method 1, fixed k:   k={int(best_k['k'])} -> f1@k={best_k['f1@k']}, "
          f"precision@k={best_k['precision@k']}, recall@k={best_k['recall@k']}")
    print(f"Method 2, threshold: threshold={best_t['cutoff_threshold']} -> f1@k={best_t['f1@k']}, "
          f"precision@k={best_t['precision@k']}, recall@k={best_t['recall@k']}, "
          f"avg tags={best_t['avg_suggested_tags']}")

    # threshold must be strictly better; on a tie keep fixed k (simpler)
    if best_t["f1@k"] > best_k["f1@k"]:
        best = {"method": "threshold", "value": float(best_t["cutoff_threshold"]), "f1": float(best_t["f1@k"]),
                "avg_suggested_tags": float(best_t["avg_suggested_tags"])}
    else:
        best = {"method": "fixed_k", "value": int(best_k["k"]), "f1": float(best_k["f1@k"]),
                "avg_suggested_tags": int(best_k["k"])}
    print(f"Best cut-off: {best['method']} = {best['value']} (f1@k={best['f1']})")
    return best


def main() -> None:
    print("Loading data...")
    data = load_indexing_data()
    train_df, val_df = data["train"], data["val"]
    vocab = load_vocabulary()
    broader_of = dict(zip(vocab["tag"], vocab["broader"]))
    bands = frequency_bands(train_df["labels"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tune_file = out_dir / "tfidf_lr_hparams_val.csv"
    fixed_k_file = out_dir / "tfidf_lr_cutoff_topk_val.csv"
    threshold_file = out_dir / "tfidf_lr_cutoff_threshold_val.csv"

    # tune hyperparameters of TF-IDF + logistic regression on the validation set
    if not tune_file.exists(): 
        print(f"Tuning TF-IDF + logistic regression on the validation set ...")
        tune_results = tune(train_df, val_df, broader_of, bands)
        tune_results.to_csv(out_dir / "tfidf_lr_hparams_val.csv", index=False)
    
    # best TF-IDF + logistic regression settings (first row = highest ndcg@k)
    best_row = pd.read_csv(tune_file).iloc[0]

    # tune the cut-off (how many tags to suggest) on the validation set
    best_cutoff = tune_k(train_df, val_df, broader_of, bands, best_row, fixed_k_file, threshold_file)

if __name__ == "__main__":
    main()




