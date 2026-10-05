# metadata_archive_AI

A learning project that rebuilds a small version of an AI-assisted archive curation pipeline on TED Talks:
automatic subject indexing, metadata generation, thematic coding, thematic search, and human-in-the-loop review.

> Work in progress. Full documentation will be added at the end of the project.

## Project structure

```
data/        raw and processed data (git-ignored, downloaded by a script)
docs/        project plan and notes
notebooks/   exploration and experiments
src/         Python code for each pipeline step
```

## Quick start

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data    # needs a Kaggle API token in ~/.kaggle/access_token
python -m src.prepare_data     # builds data/processed/talks.parquet
```

## Data

TED Talks dataset by Rounak Banik, Kaggle (2017): https://www.kaggle.com/datasets/rounakbanik/ted-talks

The data is not included in this repository. TED content is licensed CC BY-NC-ND 4.0;
please check the Kaggle page for the dataset's terms.
