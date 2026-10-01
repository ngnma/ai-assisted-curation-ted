"""Load and validate the project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config, resolve paths from the repo root and validate it."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {name: PROJECT_ROOT / p for name, p in cfg["paths"].items()}
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    splits = cfg["splits"]
    total = splits["train"] + splits["val"] + splits["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
