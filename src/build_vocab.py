"""Build the tag vocabulary and an LLM-drafted hierarchy for human review.

Output: vocab/thesaurus_draft.csv  (columns: tag, broader)
Run from the project root:  python -m src.build_vocab
"""

import pandas as pd
import json

import ollama

from src.config import EXCLUDED_TAGS, LLM_MODEL, MIN_TAG_FREQ, SEED, TALKS_FILE, VOCAB_DIR

BATCH_SIZE = 40  # tags per LLM call when assigning; small batches = fewer mistakes for a 7B model

PROPOSE_PROMPT = """You are designing a small subject thesaurus for a catalogue of talks.
Read the tags below and propose about 20 broader subject concepts that together cover all of them.
Use short, general subject labels in lowercase (e.g. "health and medicine").

Tags:
{tags}"""

ASSIGN_PROMPT = """Assign each tag below to the single best broader concept from this list:
{concepts}

Tags:
{tags}"""


def build_vocabulary() -> list[str]:
    """Tags used in at least MIN_TAG_FREQ talks, minus TED event tags."""
    counts = pd.read_parquet(TALKS_FILE)["tags"].explode().value_counts()
    keep = counts[(counts >= MIN_TAG_FREQ) & ~counts.index.isin(EXCLUDED_TAGS)]
    return sorted(keep.index)


def ask_llm(prompt: str, schema: dict) -> dict:
    """Call the local LLM and force its answer to match the JSON schema (structured output)."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": 0, "seed": SEED, "num_ctx": 8192},
    )
    return json.loads(response["message"]["content"])


def propose_concepts(tags: list[str]) -> list[str]:
    schema = {
        "type": "object",
        "properties": {"concepts": {"type": "array", "items": {"type": "string"}}},
        "required": ["concepts"],
    }
    answer = ask_llm(PROPOSE_PROMPT.format(tags="\n".join(tags)), schema)
    return sorted({c.strip().lower() for c in answer["concepts"]})


def assign_tags(tags: list[str], concepts: list[str]) -> dict[str, str]:
    schema = {  # "enum" = the broader concept MUST be one from our list
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"tag": {"type": "string"}, "broader": {"type": "string", "enum": concepts}},
                    "required": ["tag", "broader"],
                },
            }
        },
        "required": ["assignments"],
    }
    result = {}
    for i in range(0, len(tags), BATCH_SIZE):
        batch = tags[i : i + BATCH_SIZE]
        prompt = ASSIGN_PROMPT.format(concepts="\n".join(concepts), tags="\n".join(batch))
        for a in ask_llm(prompt, schema)["assignments"]:
            if a["tag"] in batch and a["tag"] not in result:  # ignore invented or repeated tags
                result[a["tag"]] = a["broader"]
        print(f"Assigned {min(i + BATCH_SIZE, len(tags))}/{len(tags)} tags")
    return result


def main() -> None:
    vocab = build_vocabulary()
    print(f"Vocabulary: {len(vocab)} tags")

    concepts = propose_concepts(vocab)
    print(f"LLM proposed {len(concepts)} broader concepts:", concepts)

    assigned = assign_tags(vocab, concepts)
    draft = pd.DataFrame({"tag": vocab})
    draft["broader"] = draft["tag"].map(assigned).fillna("UNASSIGNED")  # tags the LLM skipped

    VOCAB_DIR.mkdir(exist_ok=True)
    draft.sort_values(["broader", "tag"]).to_csv(VOCAB_DIR / "thesaurus_draft.csv", index=False)
    print(draft["broader"].value_counts().to_string())
    print(f"UNASSIGNED: {(draft['broader'] == 'UNASSIGNED').sum()} - fix these during review")


if __name__ == "__main__":
    main()