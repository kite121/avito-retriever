from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_articles(data_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_feather(Path(data_dir) / "articles.f")
    expected = ["article_id", "title", "body"]
    if list(frame.columns) != expected:
        raise ValueError(f"articles.f columns must be {expected}, got {list(frame.columns)}")
    if frame["article_id"].duplicated().any():
        raise ValueError("articles.f contains duplicated article_id")
    return frame


def load_calibration(data_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_feather(Path(data_dir) / "calibration.f")
    expected = ["query_id", "query_text", "ground_truth"]
    if list(frame.columns) != expected:
        raise ValueError(f"calibration.f columns must be {expected}, got {list(frame.columns)}")
    return frame


def load_test(data_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_feather(Path(data_dir) / "test.f")
    expected = ["query_id", "query_text"]
    if list(frame.columns) != expected:
        raise ValueError(f"test.f columns must be {expected}, got {list(frame.columns)}")
    return frame


def parse_ground_truth(value: str) -> list[int]:
    return [int(article_id) for article_id in str(value).split()]

