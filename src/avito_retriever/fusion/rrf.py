from __future__ import annotations

from collections import defaultdict

import pandas as pd

from avito_retriever.contracts import RANKING_COLUMNS


def weighted_rrf(
    rankings: dict[str, pd.DataFrame],
    weights: dict[str, float],
    rrf_k: int = 40,
    top_k: int = 100,
) -> pd.DataFrame:
    scores: dict[tuple[int, int], float] = defaultdict(float)
    for name, frame in rankings.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        for row in frame.itertuples(index=False):
            scores[(int(row.query_id), int(row.article_id))] += weight / (rrf_k + int(row.rank))

    by_query: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (query_id, article_id), score in scores.items():
        by_query[query_id].append((article_id, score))

    rows: list[dict[str, int | float | str]] = []
    for query_id, values in by_query.items():
        ranked = sorted(values, key=lambda item: (-item[1], item[0]))[:top_k]
        for rank, (article_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "article_id": article_id,
                    "score": score,
                    "rank": rank,
                    "source": "rrf",
                }
            )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)

