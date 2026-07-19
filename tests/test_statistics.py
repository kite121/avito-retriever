import pandas as pd

from avito_retriever.evaluation.statistics import compare_paired_runs


def test_paired_comparison_aligns_queries() -> None:
    baseline = pd.DataFrame({"query_id": [1, 2, 3], "ap@10": [0.0, 0.5, 1.0]})
    candidate = pd.DataFrame({"query_id": [3, 1, 2], "ap@10": [1.0, 0.5, 1.0]})
    result = compare_paired_runs(
        baseline,
        candidate,
        bootstrap_samples=100,
        permutation_samples=100,
        seed=1,
    )
    assert result["n_queries"] == 3
    assert result["mean_difference"] > 0
    assert result["wins"] == 2

