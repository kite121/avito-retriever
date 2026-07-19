from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from avito_retriever.data.io import load_articles, load_calibration
from avito_retriever.evaluation.metrics import evaluate_rankings
from avito_retriever.preprocessing.html import FIELD_COLUMNS, parse_articles
from avito_retriever.preprocessing.normalize import normalize_lexical
from avito_retriever.retrieval.bm25f import BM25FIndex
from avito_retriever.tokenization.sentencepiece import train_or_load
from avito_retriever.tracking.runs import RunStore


def run_bm25f_baseline(config: dict[str, Any], project_root: str | Path) -> RunStore:
    timings: dict[str, float] = {}
    paths = config["paths"]
    data_dir = paths["data_dir"]

    started = time.perf_counter()
    articles = load_articles(data_dir)
    calibration = load_calibration(data_dir)
    timings["load_data"] = time.perf_counter() - started

    parsed_path = Path(project_root) / paths["parsed_articles"]
    started = time.perf_counter()
    if parsed_path.exists():
        parsed = pd.read_parquet(parsed_path)
    else:
        parsed = parse_articles(articles)
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.to_parquet(parsed_path, index=False)
    timings["parse_html"] = time.perf_counter() - started

    normalization = config["preprocessing"]["normalization"]
    lexical = parsed.copy()
    for field in FIELD_COLUMNS:
        lexical[field] = lexical[field].fillna("").map(lambda value: normalize_lexical(value, normalization))
    normalized_queries = calibration[["query_id", "query_text", "ground_truth"]].copy()
    normalized_queries["query_text"] = normalized_queries["query_text"].map(
        lambda value: normalize_lexical(value, normalization)
    )

    training_texts: list[str] = []
    for field in FIELD_COLUMNS:
        training_texts.extend(lexical[field].tolist())
    if config["sentencepiece"].get("train_on_calibration_queries", True):
        training_texts.extend(normalized_queries["query_text"].tolist())

    started = time.perf_counter()
    tokenizer = train_or_load(
        training_texts,
        config["sentencepiece"],
        Path(project_root) / paths["index_dir"] / "sentencepiece",
    )
    timings["sentencepiece"] = time.perf_counter() - started

    started = time.perf_counter()
    bm25f_config = config["retrieval"]["bm25f"]
    index = BM25FIndex(bm25f_config["fields"], tokenizer.encode, k1=bm25f_config["k1"])
    index.fit(lexical)
    rankings = index.retrieve(normalized_queries, top_k=int(bm25f_config["top_k"]))
    timings["bm25f"] = time.perf_counter() - started

    metrics, per_query = evaluate_rankings(
        rankings,
        calibration,
        k=int(config["evaluation"]["k"]),
        candidate_depths=config["evaluation"]["candidate_recall_depths"],
    )

    store = RunStore(Path(project_root) / paths["run_dir"], config, project_root)
    store.write_rankings(rankings)
    store.write_per_query(per_query)
    store.write_json("metrics.json", metrics)
    store.write_json("timings.json", timings)
    store.write_json(
        "artifacts.json",
        {
            "parsed_articles": str(parsed_path),
            "sentencepiece_model": str(tokenizer.model_path),
        },
    )
    return store

