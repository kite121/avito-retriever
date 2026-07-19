import pandas as pd

from avito_retriever.retrieval.bm25f import BM25FIndex


def test_title_weight_can_move_document_above_body_match() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [1, 2],
            "title": ["доставка", "другое"],
            "body": ["", "доставка доставка доставка"],
        }
    )
    index = BM25FIndex(
        {"title": {"weight": 8.0, "b": 0.0}, "body": {"weight": 1.0, "b": 0.0}},
        lambda text: text.split(),
    ).fit(articles)
    assert index.search("доставка", top_k=2)[0][0] == 1

