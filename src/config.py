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

# data preprocessing settings
MIN_TRANSCRIPT_WORDS = 300  # shorter "transcripts" are mostly music/performances