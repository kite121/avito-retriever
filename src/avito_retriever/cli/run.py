from __future__ import annotations

import json
from pathlib import Path

import typer

from avito_retriever.config import load_config
from avito_retriever.pipeline.baseline import run_bm25f_baseline

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    override: list[str] | None = typer.Option(None, "--override", "-o"),
) -> None:
    """Run one configured experiment and save a self-contained run directory."""

    project_root = Path(__file__).resolve().parents[3]
    config = load_config(config_path, override or [])
    dense_enabled = bool(config["retrieval"]["dense"]["enabled"])
    knn_enabled = bool(config["retrieval"]["knn"]["enabled"])
    reranker_enabled = bool(config["reranker"]["enabled"])
    if dense_enabled or knn_enabled or reranker_enabled:
        raise NotImplementedError(
            "This runner currently supports the BM25F baseline. Hybrid stages are implemented next."
        )

    store = run_bm25f_baseline(config, project_root)
    metrics = json.loads((store.path / "metrics.json").read_text(encoding="utf-8"))
    typer.echo(json.dumps({"run_id": store.run_id, **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()

