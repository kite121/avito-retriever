from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    candidates = [Path(start or Path.cwd()).resolve(), Path.cwd().resolve()]
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if (parent / "pyproject.toml").exists() and (parent / "src" / "avito_retriever").exists():
                return parent
    colab_default = Path("/content/avito-retriever")
    if colab_default.exists():
        return colab_default
    raise FileNotFoundError("Could not locate avito-retriever project root")


def resolve_data_dir(project_root: Path) -> Path:
    explicit = os.environ.get("AVITO_DATA_DIR")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        project_root / "data" / "candidate_data",
        Path("/content/data"),
        Path("/content/drive/MyDrive/avito-retriever/candidate_data"),
        Path("/Users/kite/Downloads/candidate_public/candidate_data"),
    ]
    for candidate in candidates:
        if candidate and all((candidate / name).exists() for name in ["articles.f", "calibration.f", "test.f"]):
            return candidate.resolve()
    raise FileNotFoundError(
        "Dataset not found. Set AVITO_DATA_DIR to a directory containing articles.f, "
        "calibration.f and test.f."
    )


def result_dir(project_root: Path, notebook_name: str) -> Path:
    destination = project_root / "artifacts" / "notebook_results" / notebook_name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_json(value: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        ),
        encoding="utf-8",
    )
    return destination


def save_yaml(value: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return destination


def leaderboard(
    rows: Iterable[dict[str, Any]],
    objective: str = "map@10",
    ascending: bool = False,
) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    return frame.sort_values(objective, ascending=ascending).reset_index(drop=True)


def highlight_best(frame: pd.DataFrame, metric_columns: list[str]):
    formatters = {column: "{:.4f}" for column in metric_columns if column in frame.columns}
    styler = frame.style.format(formatters)
    for column in metric_columns:
        if column in frame.columns:
            styler = styler.highlight_max(subset=[column], color="#c6efce")
    return styler


def minmax_by_query(frame: pd.DataFrame, score_column: str = "score") -> pd.Series:
    def normalize(group: pd.Series) -> pd.Series:
        low, high = float(group.min()), float(group.max())
        if high <= low:
            return pd.Series(np.zeros(len(group)), index=group.index)
        return (group - low) / (high - low)

    return frame.groupby("query_id", sort=False)[score_column].transform(normalize)
