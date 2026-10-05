"""Download the TED Talks dataset from Kaggle into data/raw/.

Run from the project root:  python -m src.download_data
"""

from src.config import KAGGLE_DATASET, RAW_DIR, RAW_FILES


def main() -> None:
    if all((RAW_DIR / name).exists() for name in RAW_FILES):
        print(f"Raw files already in {RAW_DIR}, nothing to download.")
        return

    # Imported here because importing kaggle immediately looks for your API token
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)
    print(f"Downloaded {KAGGLE_DATASET} to {RAW_DIR}")


if __name__ == "__main__":
    main()
