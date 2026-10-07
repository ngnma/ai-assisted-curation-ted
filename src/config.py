"""Project settings: paths, dataset name and shared constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data locations (data/ is git-ignored)
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TALKS_FILE = PROCESSED_DIR / "talks.parquet"

# Source dataset on Kaggle
KAGGLE_DATASET = "rounakbanik/ted-talks"
RAW_FILES = ["ted_main.csv", "transcripts.csv"]

# Shared settings
SEED = 42
LLM_MODEL = "qwen2.5:7b"  # local open-weight model served by Ollama

# EDA settings
MIN_TRANSCRIPT_WORDS = 300  # shorter "transcripts" are mostly music/performances

# Vocabulary settings (from EDA)
VOCAB_DIR = ROOT / "vocab"
MIN_TAG_FREQ = 10
EXCLUDED_TAGS = [  # TED event/programme labels, not subjects
    "tedx", "ted fellows", "ted brain trust", "ted prize", "tedyouth", "tedmed",
    "tednyc", "ted books", "ted en español", "ted residency", "ted-ed",
]
# Phase 1: subject indexing
SPLIT_FILE = PROCESSED_DIR / "splits.csv"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}  # time-based: oldest -> newest
EVAL_KS = [5, 10]  # score the top 5 and top 10 suggestions (test talks have ~12 tags, train ~6)
