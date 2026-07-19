import pandas as pd
import pytest

from avito_retriever.cli.validate_submission import validate_submission_frame


def test_valid_submission() -> None:
    test = pd.DataFrame({"query_id": [1, 2], "query_text": ["a", "b"]})
    answer = pd.DataFrame({"query_id": [1, 2], "answer": ["10 20", "20 30"]})
    validate_submission_frame(answer, test, {10, 20, 30})


def test_submission_rejects_duplicate_article_ids() -> None:
    test = pd.DataFrame({"query_id": [1], "query_text": ["a"]})
    answer = pd.DataFrame({"query_id": [1], "answer": ["10 10"]})
    with pytest.raises(ValueError, match="duplicated article_id"):
        validate_submission_frame(answer, test, {10})

