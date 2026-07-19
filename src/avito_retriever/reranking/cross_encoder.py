from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from avito_retriever.contracts import RANKING_COLUMNS


def rerank_with_cross_encoder(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    text_by_pair: Mapping[tuple[int, int], str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
    source: str = "reranker",
) -> pd.DataFrame:
    query_text = {
        int(row.query_id): str(row.query_text) for row in queries.itertuples(index=False)
    }
    ordered = candidates.sort_values(["query_id", "rank"]).reset_index(drop=True)
    pairs = [
        (query_text[int(row.query_id)], text_by_pair[(int(row.query_id), int(row.article_id))])
        for row in ordered.itertuples(index=False)
    ]
    model = CrossEncoder(model_name, max_length=max_length, device=device)
    scores = np.asarray(
        model.predict(pairs, batch_size=batch_size, show_progress_bar=True), dtype=float
    ).reshape(-1)
    ordered["score"] = scores
    ordered["source"] = source
    ordered["rank"] = (
        ordered.groupby("query_id")["score"].rank(method="first", ascending=False).astype(int)
    )
    return ordered[RANKING_COLUMNS].sort_values(["query_id", "rank"]).reset_index(drop=True)

