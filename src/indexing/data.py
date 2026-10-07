"""Load the data for Phase 1 (subject indexing): input text, gold tags and split for each talk."""

import pandas as pd

from src.config import SPLIT_FILE, TALKS_FILE, VOCAB_DIR


def load_vocabulary() -> pd.DataFrame:
    """The reviewed thesaurus: one row per allowed tag with its broader concept."""
    vocab = pd.read_csv(VOCAB_DIR / "thesaurus.csv")
    vocab["broader"] = vocab["broader"].str.strip().str.lower()  # e.g. two spellings of the same group
    return vocab


def load_indexing_data() -> dict[str, pd.DataFrame]:
    """Return {"train": df, "val": df, "test": df}. Each df has: talk_id, title, text, labels.

    text   = what the models read (title + description + transcript)
    labels = the talk's gold tags, keeping only tags in our vocabulary
    """
    vocab = set(load_vocabulary()["tag"])
    talks = pd.read_parquet(TALKS_FILE).merge(pd.read_csv(SPLIT_FILE), on="talk_id", validate="one_to_one")

    talks["text"] = talks["title"] + "\n\n" + talks["description"] + "\n\n" + talks["transcript"]
    talks["labels"] = talks["tags"].map(lambda tags: [t for t in tags if t in vocab])
    talks = talks[talks["labels"].map(len) > 0]  # a talk with no vocabulary tags can't be scored

    cols = ["talk_id", "title", "text", "labels"]
    return {name: group[cols].reset_index(drop=True) for name, group in talks.groupby("split")}
