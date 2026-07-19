from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import pandas as pd


RANKING_COLUMNS = ["query_id", "article_id", "score", "rank", "source"]


@dataclass(slots=True)
class TextBlock:
    order: int
    kind: str
    text: str
    heading: str | None = None
    tab: str | None = None
    spoiler: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    order: int
    src: str
    alt: str = ""
    title: str = ""
    heading: str | None = None
    tab: str | None = None
    spoiler: str | None = None
    local_path: str | None = None
    sha256: str | None = None
    ocr_text: str = ""
    ocr_status: str = "pending"


@dataclass(slots=True)
class ParsedArticle:
    article_id: int
    title: str
    blocks: list[TextBlock]
    images: list[ImageRecord]
    fields: dict[str, str]


class Retriever(Protocol):
    name: str

    def fit(self, articles: Sequence[ParsedArticle], calibration: pd.DataFrame | None = None) -> None:
        ...

    def retrieve(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Return columns from ``RANKING_COLUMNS``."""
        ...


class Reranker(Protocol):
    name: str

    def rerank(self, queries: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return columns from ``RANKING_COLUMNS``."""
        ...


def validate_ranking_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in RANKING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Ranking frame is missing columns: {missing}")
    if frame[["query_id", "article_id", "rank"]].isna().any().any():
        raise ValueError("Ranking identifiers and ranks cannot be null")
    duplicates = frame.duplicated(["query_id", "article_id"])
    if duplicates.any():
        raise ValueError("Each article_id must occur at most once per query")

