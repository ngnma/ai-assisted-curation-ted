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



# Fixed TF-IDF settings
TFIDF_FIXED = dict(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.8, max_features=100_000)
NGRAM_RANGES = [(1, 1), (1, 2)]

# Fixed logistic regression settings
C_VALUES = [100, 300]
CLASS_WEIGHTS = [None, "balanced"]

# # TEMP -> DELETE
# NGRAM_RANGES = [(1, 1)]
# C_VALUES = [100, 300]
# CLASS_WEIGHTS = ["balanced"]


# Fixed Top-k settings
FIXED_K = list(range(1, 15))  # for tuning cut-off threshold


def rank_tags(proba: np.ndarray, classes: np.ndarray, k: int) -> list[list[str]]:
    """Top-k tag names per talk, highest probability first."""
    top = np.argsort(-proba, axis=1)[:, :k]
    return [list(classes[row]) for row in top]

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

def main() -> None:
    print("Loading data...")
    data = load_indexing_data()
    train_df, val_df = data["train"], data["val"]
    vocab = load_vocabulary()
    broader_of = dict(zip(vocab["tag"], vocab["broader"]))
    bands = frequency_bands(train_df["labels"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "tfidf_lr_tuning_val.csv"

    # tune hyperparameters of TF-IDF + logistic regression on the validation set
    if not out_file.exists(): 
        print(f"Tuning TF-IDF + logistic regression on the validation set (output: {out_file})...")
        tune_results = tune(train_df, val_df, broader_of, bands)
        tune_results.to_csv(out_dir / "tfidf_lr_tuning_val.csv", index=False)
    
    # tune cut-off threshold for top-k predictions on the validation set
    # method 1: fixed k
    print("Tuning cut-off threshold for top-k predictions on the validation set...")
    tune_results = pd.read_csv(out_file)
    best_row = tune_results.iloc[0]
    cutoff_results = tune_cutoff_fixed_k(train_df, val_df, broader_of, bands, best_row)
    cutoff_results.to_csv(out_dir / "tfidf_lr_cutoff_val.csv", index=False)

    # method 2: thresholding
    print("Tuning cut-off threshold for top-k predictions on the validation set (method 2: thresholding)...")

if __name__ == "__main__":
    main()



