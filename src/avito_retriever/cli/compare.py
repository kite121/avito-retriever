from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from avito_retriever.evaluation.statistics import compare_paired_runs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    baseline: Path = typer.Argument(..., exists=True, readable=True),
    candidate: Path = typer.Argument(..., exists=True, readable=True),
    metric: str = typer.Option("ap@10"),
    bootstrap_samples: int = typer.Option(10_000),
    permutation_samples: int = typer.Option(10_000),
    seed: int = typer.Option(42),
    output: Path | None = typer.Option(None),
) -> None:
    """Run paired query-level statistical tests for two saved experiments."""

    result = compare_paired_runs(
        pd.read_parquet(baseline),
        pd.read_parquet(candidate),
        metric=metric,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    typer.echo(rendered)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    app()

