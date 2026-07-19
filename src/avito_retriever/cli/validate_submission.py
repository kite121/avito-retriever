from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from avito_retriever.data.io import load_articles, load_test

app = typer.Typer(add_completion=False)


def validate_submission_frame(
    answer: pd.DataFrame, test: pd.DataFrame, valid_article_ids: set[int], max_k: int = 10
) -> None:
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Submission columns must be exactly ['query_id', 'answer']")
    if answer["query_id"].duplicated().any():
        raise ValueError("Submission contains duplicated query_id")
    if set(answer["query_id"]) != set(test["query_id"]):
        raise ValueError("Submission query_id set does not match test.f")

    for record in answer.itertuples(index=False):
        ids = [int(value) for value in str(record.answer).split()]
        if not ids:
            raise ValueError(f"query_id={record.query_id} has an empty answer")
        if len(ids) > max_k:
            raise ValueError(f"query_id={record.query_id} has more than {max_k} articles")
        if len(ids) != len(set(ids)):
            raise ValueError(f"query_id={record.query_id} contains duplicated article_id")
        unknown = set(ids) - valid_article_ids
        if unknown:
            raise ValueError(f"query_id={record.query_id} contains unknown article_id: {unknown}")


@app.command()
def main(
    answer_csv: Path = typer.Argument(..., exists=True, readable=True),
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    answer = pd.read_csv(answer_csv, dtype={"answer": "string"})
    articles = load_articles(data_dir)
    test = load_test(data_dir)
    validate_submission_frame(answer, test, set(articles["article_id"].astype(int)))
    typer.echo(f"Valid submission: {len(answer)} queries")


if __name__ == "__main__":
    app()

