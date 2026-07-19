from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from avito_retriever.contracts import RANKING_COLUMNS


def rerank_with_bi_encoder(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    text_by_pair: Mapping[tuple[int, int], str],
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    source: str = "bi_encoder_reranker",
) -> pd.DataFrame:
    """Rerank a fixed candidate set with cosine similarity from a bi-encoder."""

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    query_by_id = dict(zip(queries.query_id.astype(int), queries.query_text.astype(str)))
    ordered = candidates.sort_values(["query_id", "rank"]).reset_index(drop=True).copy()
    unique_query_ids = ordered["query_id"].astype(int).drop_duplicates().tolist()
    query_texts = [query_by_id[query_id] for query_id in unique_query_ids]
    document_texts = [
        text_by_pair[(int(row.query_id), int(row.article_id))]
        for row in ordered.itertuples(index=False)
    ]
    if "e5" in model_name.lower():
        query_texts = [f"query: {text}" for text in query_texts]
        document_texts = [f"passage: {text}" for text in document_texts]
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    document_embeddings = model.encode(
        document_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_position = {query_id: position for position, query_id in enumerate(unique_query_ids)}
    ordered["score"] = np.asarray(
        [
            float(document_embeddings[position] @ query_embeddings[query_position[int(query_id)]])
            for position, query_id in enumerate(ordered["query_id"])
        ]
    )
    ordered["source"] = source
    ordered["rank"] = (
        ordered.groupby("query_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ordered[RANKING_COLUMNS].sort_values(["query_id", "rank"]).reset_index(drop=True)
