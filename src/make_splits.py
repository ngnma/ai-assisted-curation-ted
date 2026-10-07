"""Split talks by publish date: oldest 70% train, next 15% validation, newest 15% test.

Why by date: the real system tags *future* deposits, so we train on the past and test on the future.
Output: data/processed/splits.csv  (talk_id, split)
Run from the project root:  python -m src.make_splits
"""

import pandas as pd

from src.config import SPLIT_FILE, SPLIT_RATIOS, TALKS_FILE


def main() -> None:
    df = pd.read_parquet(TALKS_FILE, columns=["talk_id", "published_date"])
    df = df.sort_values(["published_date", "talk_id"]).reset_index(drop=True)  # talk_id breaks ties

    n_train = int(len(df) * SPLIT_RATIOS["train"])
    n_val = int(len(df) * SPLIT_RATIOS["val"])
    df["split"] = "test"
    df.loc[: n_train - 1, "split"] = "train"
    df.loc[n_train : n_train + n_val - 1, "split"] = "val"

    df[["talk_id", "split"]].to_csv(SPLIT_FILE, index=False)
    print(f"Saved {SPLIT_FILE}")
    print(df.groupby("split", sort=False)["published_date"].agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
