"""Clean and merge the raw TED files into one table (one row per talk).

Output: data/processed/talks.parquet
Run from the project root:  python -m src.prepare_data
"""

import ast
import re

import pandas as pd
from src.config import MIN_TRANSCRIPT_WORDS, PROCESSED_DIR, RAW_DIR, TALKS_FILE

COLUMNS = [
    "talk_id", "title", "description", "main_speaker", "speaker_occupation", "occupations",
    "num_speaker", "event", "film_date", "published_date", "duration",
    "languages", "views", "tags", "related_talk_ids", "url",
    "transcript",
]


def parse_list(text):
    """Lists are stored as text in the CSV, e.g. "['a', 'b']". Turn them into real lists."""
    return ast.literal_eval(text) if isinstance(text, str) else []


def clean_tags(tags):
    """Lowercase, trim and de-duplicate tags, keeping their order."""
    return list(dict.fromkeys(t.strip().lower() for t in tags if t.strip()))


def load_talks():
    talks = pd.read_csv(RAW_DIR / "ted_main.csv")
    talks["url"] = talks["url"].str.strip()  # URLs end with a hidden newline
    talks["talk_id"] = talks["url"].str.rstrip("/").str.split("/").str[-1]
    talks["tags"] = talks["tags"].map(parse_list).map(clean_tags)
    for col in ["film_date", "published_date"]:
        talks[col] = pd.to_datetime(talks[col], unit="s")  # Unix timestamps -> dates

    # related_talks points to other talks by title; convert those to our talk_ids
    title_to_id = dict(zip(talks["title"], talks["talk_id"]))
    talks["related_talk_ids"] = talks["related_talks"].map(
        lambda text: [title_to_id[r["title"]] for r in parse_list(text) if r.get("title") in title_to_id]
    )
    # Occupations are messy and inconsistent; split them into a list of roles for each speaker
    talks["occupations"] = talks["speaker_occupation"].map(split_occupations)
    return talks


def load_transcripts():
    transcripts = pd.read_csv(RAW_DIR / "transcripts.csv")
    transcripts["url"] = transcripts["url"].str.strip()
    transcripts["transcript"] = transcripts["transcript"].map(remove_sound_notes).map(restore_paragraphs)
    return transcripts.drop_duplicates(subset="url")  # a few transcripts appear twice

def split_occupations(text):
    """'Biologist, Nobel laureate' -> ['biologist', 'nobel laureate']. One speaker, several roles."""
    if not isinstance(text, str):
        return []
    parts = re.split(r",|;|\band\b", text.lower())
    return [p.strip() for p in parts if p.strip()]

def restore_paragraphs(text):
    """The raw data lost its paragraph breaks, so sentences are glued: 'play out.I have'.
    Put a paragraph break back wherever . ? ! (optionally followed by a closing quote)
    is directly followed by a capital letter."""
    return re.sub(r'(?<=[a-z0-9][.?!])(?=[A-Z“])|(?<=[.?!]["”])(?=[A-Z])', "\n\n", text)


def remove_sound_notes(text):
    """Remove audience/sound notes like (Laughter), (Applause), (Music).
    Keeps brackets with a colon, e.g. '(MS: No idea.)', because those are real speech."""
    text = re.sub(r"\([^():]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    talks = load_talks()
    transcripts = load_transcripts()

    # validate="one_to_one" raises an error if a URL is duplicated on either side
    df = talks.merge(transcripts, on="url", how="left", validate="one_to_one")

    # Keep only talks with a usable transcript (drops missing ones and short performances)
    n_words = df["transcript"].str.split().str.len()
    df = df[n_words >= MIN_TRANSCRIPT_WORDS].reset_index(drop=True)

    # Related talks must point only to talks that are still in the dataset
    kept_ids = set(df["talk_id"])
    df["related_talk_ids"] = df["related_talk_ids"].map(lambda ids: [i for i in ids if i in kept_ids])

    df = df[COLUMNS]

    # Basic sanity checks: stop early if the data is broken
    assert df["talk_id"].is_unique, "talk_id is not unique"
    assert df["tags"].map(len).gt(0).all(), "some talks have no tags"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TALKS_FILE, index=False)
    print(f"Saved {len(df)} talks to {TALKS_FILE}")
    print(f"  unique tags: {df['tags'].explode().nunique()}")


if __name__ == "__main__":
    main()
