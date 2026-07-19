from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from avito_retriever.data.io import load_calibration
from avito_retriever.evaluation.metrics import evaluate_rankings

app = typer.Typer(add_completion=False)


@app.command()
def main(
    rankings: Path = typer.Argument(..., exists=True, readable=True),
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path | None = typer.Option(None),
    k: int = typer.Option(10),
) -> None:
    """Evaluate a standardized rankings parquet independently of the pipeline."""

    ranking_frame = pd.read_parquet(rankings)
    calibration = load_calibration(data_dir)
    metrics, per_query = evaluate_rankings(ranking_frame, calibration, k=k)
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_query.to_parquet(output_dir / "per_query_metrics.parquet", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    app()

