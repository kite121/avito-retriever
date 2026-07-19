import pandas as pd

from avito_retriever.evaluation.multiple import holm_bonferroni
from avito_retriever.evaluation.selection import make_tune_confirm_split


def test_duplicate_queries_stay_in_same_split() -> None:
    calibration = pd.DataFrame(
        {
            "query_id": [1, 2, 3, 4, 5, 6],
            "query_text": ["Один", "один", "два", "три", "четыре", "пять"],
        }
    )
    split = make_tune_confirm_split(calibration, n_folds=3, seed=1)
    assert split.loc[split["query_id"].isin([1, 2]), "split"].nunique() == 1


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[1]

