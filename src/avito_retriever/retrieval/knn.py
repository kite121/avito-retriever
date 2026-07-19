from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from avito_retriever.contracts import RANKING_COLUMNS
from avito_retriever.data.io import parse_ground_truth
from avito_retriever.retrieval.bm25 import BM25Index


NEIGHBOUR_COLUMNS = ["query_id", "neighbour_id", "score", "rank", "source"]


def lexical_neighbours_oof(
    calibration: pd.DataFrame,
    folds: pd.Series,
    tokenizer: Callable[[str], list[str]],
    k1: float,
    b: float,
    depth: int,
) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for fold in sorted(folds.unique()):
        train = calibration.loc[folds != fold]
        valid = calibration.loc[folds == fold]
        index = BM25Index(tokenizer, k1=k1, b=b).fit(
            [(int(row.query_id), str(row.query_text)) for row in train.itertuples(index=False)]
        )
        for query in valid.itertuples(index=False):
            for rank, (neighbour_id, score) in enumerate(index.search(query.query_text, depth), start=1):
                rows.append(
                    {
                        "query_id": int(query.query_id),
                        "neighbour_id": int(neighbour_id),
                        "score": float(score),
                        "rank": rank,
                        "source": "knn_lexical",
                    }
                )
    return pd.DataFrame(rows, columns=NEIGHBOUR_COLUMNS)


def dense_neighbours_oof(
    calibration: pd.DataFrame,
    folds: pd.Series,
    embeddings: np.ndarray,
    depth: int,
) -> pd.DataFrame:
    query_ids = calibration["query_id"].astype(int).to_numpy()
    fold_values = folds.to_numpy()
    similarities = embeddings @ embeddings.T
    rows: list[dict[str, int | float | str]] = []
    for index, query_id in enumerate(query_ids):
        allowed = np.flatnonzero(fold_values != fold_values[index])
        scores = similarities[index, allowed]
        k = min(depth, len(allowed))
        selected = np.argpartition(-scores, k - 1)[:k]
        selected = selected[np.argsort(-scores[selected])]
        for rank, local_index in enumerate(selected, start=1):
            neighbour_index = allowed[local_index]
            rows.append(
                {
                    "query_id": int(query_id),
                    "neighbour_id": int(query_ids[neighbour_index]),
                    "score": float(similarities[index, neighbour_index]),
                    "rank": rank,
                    "source": "knn_dense",
                }
            )
    return pd.DataFrame(rows, columns=NEIGHBOUR_COLUMNS)


def fuse_neighbours_rrf(
    frames: dict[str, pd.DataFrame], weights: dict[str, float], rrf_k: int = 20
) -> pd.DataFrame:
    scores: dict[tuple[int, int], float] = defaultdict(float)
    source_count: dict[tuple[int, int], int] = defaultdict(int)
    for name, frame in frames.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        for row in frame.itertuples(index=False):
            key = (int(row.query_id), int(row.neighbour_id))
            scores[key] += weight / (rrf_k + int(row.rank))
            source_count[key] += 1

    rows: list[dict[str, int | float]] = []
    by_query: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (query_id, neighbour_id), score in scores.items():
        by_query[query_id].append((neighbour_id, score))
    for query_id, values in by_query.items():
        for rank, (neighbour_id, score) in enumerate(
            sorted(values, key=lambda item: (-item[1], item[0])), start=1
        ):
            rows.append(
                {
                    "query_id": query_id,
                    "neighbour_id": neighbour_id,
                    "score": score,
                    "rank": rank,
                    "agreement": source_count[(query_id, neighbour_id)],
                }
            )
    return pd.DataFrame(rows)


def neighbours_to_article_rankings(
    neighbours: pd.DataFrame,
    calibration: pd.DataFrame,
    top_k_neighbours: int,
    top_k_articles: int = 100,
    normalize_by_label_count: bool = False,
) -> pd.DataFrame:
    labels = {
        int(row.query_id): parse_ground_truth(row.ground_truth)
        for row in calibration.itertuples(index=False)
    }
    rows: list[dict[str, int | float | str]] = []
    for query_id, group in neighbours.groupby("query_id", sort=False):
        votes: dict[int, float] = defaultdict(float)
        selected = group.sort_values(["rank", "neighbour_id"]).head(top_k_neighbours)
        for neighbour in selected.itertuples(index=False):
            article_ids = labels[int(neighbour.neighbour_id)]
            denominator = len(article_ids) if normalize_by_label_count else 1
            for article_id in article_ids:
                votes[int(article_id)] += float(neighbour.score) / denominator
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[:top_k_articles]
        for rank, (article_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "query_id": int(query_id),
                    "article_id": int(article_id),
                    "score": float(score),
                    "rank": rank,
                    "source": "knn",
                }
            )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)

