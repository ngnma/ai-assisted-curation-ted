"""Clean and merge the raw TED files into one table (one row per talk).

Output: data/processed/talks.parquet
Run from the project root:  python -m src.prepare_data
"""

import ast

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, TALKS_FILE

COLUMNS = [
    "talk_id", "title", "description", "main_speaker", "speaker_occupation",
    "num_speaker", "event", "film_date", "published_date", "duration",
    "languages", "views", "tags", "related_talk_ids", "url",
    "transcript", "has_transcript",
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
    return talks


def load_transcripts():
    transcripts = pd.read_csv(RAW_DIR / "transcripts.csv")
    transcripts["url"] = transcripts["url"].str.strip()
    transcripts["transcript"] = transcripts["transcript"].str.strip()
    return transcripts.drop_duplicates(subset="url")  # a few transcripts appear twice


def main() -> None:
    talks = load_talks()
    transcripts = load_transcripts()

    # validate="one_to_one" raises an error if a URL is duplicated on either side
    df = talks.merge(transcripts, on="url", how="left", validate="one_to_one")
    df["has_transcript"] = df["transcript"].notna()
    df = df[COLUMNS]

    # Basic sanity checks: stop early if the data is broken
    assert df["talk_id"].is_unique, "talk_id is not unique"
    assert df["tags"].map(len).gt(0).all(), "some talks have no tags"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TALKS_FILE, index=False)
    print(f"Saved {len(df)} talks to {TALKS_FILE}")
    print(f"  with transcript: {df['has_transcript'].sum()}")
    print(f"  unique tags:     {df['tags'].explode().nunique()}")


if __name__ == "__main__":
    main()
