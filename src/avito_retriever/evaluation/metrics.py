from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from avito_retriever.contracts import validate_ranking_frame
from avito_retriever.data.io import parse_ground_truth


def average_precision_at_k(
    predicted: Sequence[int],
    relevant: Iterable[int],
    k: int = 10,
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in seen:
            continue
        seen.add(article_id)
        if article_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant_set), k)


def candidate_recall_at_k(
    predicted: Sequence[int],
    relevant: Iterable[int],
    k: int,
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(predicted[:k]) & relevant_set) / len(relevant_set)


def _predictions_from_frame(rankings: pd.DataFrame) -> dict[int, list[int]]:
    validate_ranking_frame(rankings)
    ordered = rankings.sort_values(["query_id", "rank", "article_id"])
    return {
        int(query_id): group["article_id"].astype(int).tolist()
        for query_id, group in ordered.groupby("query_id", sort=False)
    }


def evaluate_predictions(
    predictions: Mapping[int, Sequence[int]],
    calibration: pd.DataFrame,
    k: int = 10,
    candidate_depths: Sequence[int] = (20, 50, 100),
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for record in calibration.itertuples(index=False):
        query_id = int(record.query_id)
        relevant = parse_ground_truth(record.ground_truth)
        predicted = list(predictions.get(query_id, []))
        row: dict[str, float | int] = {
            "query_id": query_id,
            f"ap@{k}": average_precision_at_k(predicted, relevant, k=k),
        }
        for depth in candidate_depths:
            row[f"recall@{depth}"] = candidate_recall_at_k(predicted, relevant, depth)
        rows.append(row)

    per_query = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    metrics = {
        column: float(per_query[column].mean())
        for column in per_query.columns
        if column != "query_id"
    }
    metrics[f"map@{k}"] = metrics.pop(f"ap@{k}")
    return metrics, per_query


def evaluate_rankings(
    rankings: pd.DataFrame,
    calibration: pd.DataFrame,
    k: int = 10,
    candidate_depths: Sequence[int] = (20, 50, 100),
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = _predictions_from_frame(rankings)
    return evaluate_predictions(predictions, calibration, k, candidate_depths)


def mean_reciprocal_rank_at_k(
    predicted: Sequence[int], relevant: Iterable[int], k: int = 10
) -> float:
    relevant_set = set(relevant)
    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in relevant_set:
            return 1.0 / rank
    return 0.0


def binary_ndcg_at_k(predicted: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    gains = np.array([1.0 if article_id in relevant_set else 0.0 for article_id in predicted[:k]])
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains / discounts).sum())
    ideal_length = min(len(relevant_set), k)
    idcg = float((np.ones(ideal_length) / np.log2(np.arange(2, ideal_length + 2))).sum())
    return dcg / idcg if idcg else 0.0

