import pandas as pd
import pytest

from avito_retriever.evaluation.metrics import average_precision_at_k, evaluate_predictions


def test_average_precision_at_10_uses_task_denominator() -> None:
    predicted = [10, 99, 20, 30]
    relevant = [10, 20, 30]
    expected = (1.0 + 2 / 3 + 3 / 4) / 3
    assert average_precision_at_k(predicted, relevant, k=10) == pytest.approx(expected)


def test_average_precision_ignores_duplicate_predictions() -> None:
    assert average_precision_at_k([10, 10, 20], [10, 20], k=10) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )


def test_evaluation_includes_queries_without_predictions() -> None:
    calibration = pd.DataFrame(
        {
            "query_id": [1, 2],
            "query_text": ["a", "b"],
            "ground_truth": ["10", "20"],
        }
    )
    metrics, per_query = evaluate_predictions({1: [10]}, calibration)
    assert metrics["map@10"] == pytest.approx(0.5)
    assert len(per_query) == 2

