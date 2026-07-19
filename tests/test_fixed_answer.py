import pandas as pd

from avito_retriever.pipeline.fixed import _rankings_to_answer


def test_fixed_rankings_build_valid_submission() -> None:
    rankings = pd.DataFrame(
        {
            "query_id": [10, 10, 20, 20],
            "article_id": [2, 1, 1, 3],
            "score": [0.9, 0.8, 0.7, 0.6],
            "rank": [1, 2, 1, 2],
            "source": ["x"] * 4,
        }
    )
    test = pd.DataFrame({"query_id": [10, 20], "query_text": ["a", "b"]})
    articles = pd.DataFrame({"article_id": [1, 2, 3]})
    answer = _rankings_to_answer(rankings, test, articles, top_k=2)
    assert answer.to_dict("records") == [
        {"query_id": 10, "answer": "2 1"},
        {"query_id": 20, "answer": "1 3"},
    ]
