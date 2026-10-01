.PHONY: setup lint test data clean

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check . --fix
	uv run ruff format .

test:
	uv run pytest -q

data:
	uv run python scripts/download_data.py
	uv run python scripts/build_dataset.py
	uv run python scripts/make_splits.py

clean:
	rm -rf .pytest_cache .ruff_cache data/processed/*
