"""Shared config loading + path resolution for EvoDesign-Agent."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path = REPO_ROOT / "configs" / "default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to this repo's root (EvoDesign Agent/)."""
    return REPO_ROOT.joinpath(*parts)


def external_path(config: dict, which: str, *parts: str) -> Path:
    """Resolve a path relative to one of the external repos (amr_moega_path / protein_rag_path)."""
    base = Path(config["external"][which])
    return base.joinpath(*parts)
