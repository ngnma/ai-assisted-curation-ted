"""Load the data for Phase 1 (subject indexing): input text, gold tags and split for each talk."""

import pandas as pd

from src.config import SPLIT_FILE, TALKS_FILE, VOCAB_DIR

from rdflib import Graph
from rdflib.namespace import SKOS


def load_vocabulary() -> pd.DataFrame:
    """The reviewed thesaurus: one row per allowed tag with its broader concept."""
    vocab = pd.read_csv(VOCAB_DIR / "thesaurus.csv")
    vocab["broader"] = vocab["broader"].str.strip().str.lower()  # e.g. two spellings of the same group
    return vocab

def load_vocab_uris() -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Read the SKOS thesaurus and map between tag names and URIs (Annif works with URIs).

    Returns:
        tag_to_uri: tag name -> URI, e.g. "activism" -> ".../tag/activism"
        uri_to_tag: URI -> tag name (to translate Annif's suggestions back)
        group_uris: URIs of the broader groups (not real tags, drop them from suggestions)
    """
    graph = Graph().parse(VOCAB_DIR / "ted_thesaurus.ttl", format="turtle")

    tag_to_uri, uri_to_tag, group_uris = {}, {}, set()
    for uri, label in graph.subject_objects(SKOS.prefLabel):
        uri, label = str(uri), str(label)
        if "/tag/" in uri:
            tag_to_uri[label] = uri
            uri_to_tag[uri] = label
        elif "/group/" in uri:
            group_uris.add(uri)

    # every tag in the reviewed CSV must exist in the SKOS file
    missing = set(load_vocabulary()["tag"]) - set(tag_to_uri)
    assert not missing, f"Tags missing from ted_thesaurus.ttl (re-run export_skos): {sorted(missing)}"
    return tag_to_uri, uri_to_tag, group_uris

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
