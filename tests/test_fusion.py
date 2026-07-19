import pandas as pd

from avito_retriever.fusion.rrf import weighted_rrf


def test_rrf_rewards_articles_supported_by_multiple_systems() -> None:
    left = pd.DataFrame(
        [(1, 10, 1.0, 1, "a"), (1, 20, 0.5, 2, "a")],
        columns=["query_id", "article_id", "score", "rank", "source"],
    )
    right = pd.DataFrame(
        [(1, 20, 1.0, 1, "b"), (1, 30, 0.5, 2, "b")],
        columns=left.columns,
    )
    fused = weighted_rrf({"a": left, "b": right}, {"a": 1.0, "b": 1.0}, rrf_k=10)
    assert int(fused.iloc[0]["article_id"]) == 20

