from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from avito_retriever.cli.validate_submission import validate_submission_frame
from avito_retriever.config import load_config
from avito_retriever.data.io import load_articles, load_calibration, load_test
from avito_retriever.experiments.notebook import save_json
from avito_retriever.fusion.rrf import weighted_rrf
from avito_retriever.pipeline.final import _bm25_neighbours, _dense_neighbours
from avito_retriever.preprocessing.html import FIELD_COLUMNS, parse_articles
from avito_retriever.preprocessing.images import download_images, extract_image_manifest
from avito_retriever.preprocessing.normalize import normalize_lexical
from avito_retriever.preprocessing.ocr import aggregate_ocr_by_article, paddle_ocr_manifest
from avito_retriever.reranking.cross_encoder import rerank_with_cross_encoder
from avito_retriever.retrieval.bm25f import BM25FIndex
from avito_retriever.retrieval.dense import DenseIndex, build_chunks
from avito_retriever.retrieval.knn import (
    fuse_neighbours_rrf,
    neighbours_to_article_rankings,
)
from avito_retriever.tokenization.sentencepiece import train_or_load


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_fingerprint(config: dict[str, Any], data_dir: Path, use_ocr: bool) -> str:
    payload = {
        "config": config,
        "use_ocr": use_ocr,
        "data": {
            name: _sha256(data_dir / name)
            for name in ("articles.f", "calibration.f", "test.f")
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _prepare_articles(
    articles: pd.DataFrame,
    project_root: Path,
    use_ocr: bool,
) -> pd.DataFrame:
    parsed_path = project_root / "artifacts" / "cache" / "parsed_articles.parquet"
    if parsed_path.exists():
        parsed = pd.read_parquet(parsed_path)
    else:
        parsed = parse_articles(articles)
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.to_parquet(parsed_path, index=False)
    if not use_ocr:
        return parsed

    parsed_ocr_path = project_root / "artifacts" / "cache" / "parsed_articles_with_ocr.parquet"
    if parsed_ocr_path.exists():
        return pd.read_parquet(parsed_ocr_path)

    ocr_dir = project_root / "artifacts" / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    download_manifest_path = ocr_dir / "download_manifest.parquet"
    if download_manifest_path.exists():
        downloaded = pd.read_parquet(download_manifest_path)
    else:
        manifest = extract_image_manifest(articles)
        downloaded = download_images(manifest, project_root / "artifacts" / "images")
        downloaded.to_parquet(download_manifest_path, index=False)

    ocr_manifest_path = ocr_dir / "ocr_manifest.parquet"
    if ocr_manifest_path.exists():
        ocr_manifest = pd.read_parquet(ocr_manifest_path)
    else:
        ocr_manifest = paddle_ocr_manifest(
            downloaded, language="ru", confidence_threshold=0.50
        )
        ocr_manifest.to_parquet(ocr_manifest_path, index=False)

    ocr_by_article = aggregate_ocr_by_article(ocr_manifest)
    parsed = parsed.drop(columns=["image_ocr"]).merge(ocr_by_article, on="article_id", how="left")
    parsed["image_ocr"] = parsed["image_ocr"].fillna("")
    parsed.to_parquet(parsed_ocr_path, index=False)
    return parsed


def _normalise_articles_and_queries(
    parsed: pd.DataFrame,
    queries: pd.DataFrame,
    normalization: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lexical = parsed.copy()
    for field in FIELD_COLUMNS:
        lexical[field] = lexical[field].fillna("").map(
            lambda value: normalize_lexical(value, normalization)
        )
    normalized_queries = queries.copy()
    normalized_queries["query_text"] = normalized_queries["query_text"].map(
        lambda value: normalize_lexical(value, normalization)
    )
    return lexical, normalized_queries


def _rankings_to_answer(
    rankings: pd.DataFrame,
    test: pd.DataFrame,
    articles: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    selected = rankings[rankings["rank"] <= top_k].sort_values(["query_id", "rank"])
    answers = (
        selected.groupby("query_id")["article_id"]
        .apply(lambda values: " ".join(map(str, list(dict.fromkeys(values))[:top_k])))
        .rename("answer")
        .reset_index()
    )
    answers = test[["query_id"]].merge(answers, on="query_id", how="left")
    validate_submission_frame(
        answers,
        test,
        set(articles["article_id"].astype(int)),
        max_k=top_k,
    )
    return answers


def run_fixed_submission(
    project_root: str | Path,
    data_dir: str | Path,
    output_csv: str | Path,
    config_path: str | Path | None = None,
    use_ocr: bool = False,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run one fixed BM25F+dense+kNN+RRF+cross-encoder architecture."""

    root = Path(project_root).resolve()
    data = Path(data_dir).resolve()
    config_file = Path(config_path or root / "configs/experiments/fixed_submission.yaml")
    config = load_config(config_file)
    fingerprint = _run_fingerprint(config, data, use_ocr)
    run_dir = root / "artifacts" / "fixed_solution" / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    final_rankings_path = run_dir / "final_rankings.parquet"
    output_path = Path(output_csv).resolve()

    articles = load_articles(data)
    calibration = load_calibration(data)
    test = load_test(data)
    top_k = int(config["final_ranking"]["top_k"])
    if final_rankings_path.exists() and not force:
        rankings = pd.read_parquet(final_rankings_path)
        answers = _rankings_to_answer(rankings, test, articles, top_k)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        answers.to_csv(output_path, index=False)
        manifest = {
            "fingerprint": fingerprint,
            "cache_hit": True,
            "answer_csv": str(output_path),
            "rows": len(answers),
        }
        save_json(manifest, run_dir / "manifest.json")
        return output_path, manifest

    timings: dict[str, float] = {}
    started = time.perf_counter()
    parsed = _prepare_articles(articles, root, use_ocr)
    timings["html_and_ocr"] = time.perf_counter() - started

    normalization = config["preprocessing"]["normalization"]
    lexical, test_lexical = _normalise_articles_and_queries(parsed, test, normalization)
    _, calibration_lexical = _normalise_articles_and_queries(
        parsed, calibration, normalization
    )

    training_texts = [
        text
        for field in FIELD_COLUMNS
        for text in lexical[field].astype(str).tolist()
    ]
    training_texts.extend(calibration_lexical["query_text"].astype(str).tolist())
    started = time.perf_counter()
    sentencepiece = train_or_load(
        training_texts,
        config["sentencepiece"],
        root / "artifacts" / "indexes" / "sentencepiece_fixed",
    )
    timings["sentencepiece"] = time.perf_counter() - started

    bm25_config = config["retrieval"]["bm25f"]
    started = time.perf_counter()
    bm25 = BM25FIndex(
        bm25_config["fields"], sentencepiece.encode, k1=float(bm25_config["k1"])
    ).fit(lexical)
    bm25_ranking = bm25.retrieve(
        test_lexical, top_k=int(bm25_config["top_k"]), source="bm25f"
    )
    bm25_ranking.to_parquet(run_dir / "bm25f_rankings.parquet", index=False)
    timings["bm25f"] = time.perf_counter() - started

    chunk_config = config["chunking"]
    dense_config = config["retrieval"]["dense"]
    chunks = build_chunks(
        parsed,
        size_words=int(chunk_config["size_tokens"]),
        overlap_words=int(chunk_config["overlap_tokens"]),
    )
    calibration_dense = calibration[["query_id", "query_text"]].copy()
    test_dense = test[["query_id", "query_text"]].copy()
    if "e5" in str(dense_config["model_name"]).lower():
        chunks["text"] = "passage: " + chunks["text"]
        calibration_dense["query_text"] = "query: " + calibration_dense["query_text"]
        test_dense["query_text"] = "query: " + test_dense["query_text"]

    started = time.perf_counter()
    dense = DenseIndex(dense_config, root / "artifacts" / "embeddings" / fingerprint)
    dense.fit_chunks(chunks)
    dense_ranking = dense.retrieve(test_dense, top_k_articles=100)
    dense_ranking.to_parquet(run_dir / "dense_rankings.parquet", index=False)
    timings["dense"] = time.perf_counter() - started

    knn_config = config["retrieval"]["knn"]
    started = time.perf_counter()
    lexical_neighbours = _bm25_neighbours(
        calibration_lexical,
        test_lexical,
        sentencepiece.encode,
        k1=float(knn_config["lexical"]["k1"]),
        b=float(knn_config["lexical"]["b"]),
        depth=int(knn_config["lexical"]["depth"]),
    )
    calibration_embeddings = dense.encode(
        calibration_dense["query_text"].astype(str).tolist(), "calibration_queries"
    )
    test_embeddings = dense.encode(
        test_dense["query_text"].astype(str).tolist(), "test_queries"
    )
    dense_neighbours = _dense_neighbours(
        calibration,
        test,
        calibration_embeddings,
        test_embeddings,
        depth=int(knn_config["dense"]["depth"]),
    )
    neighbour_fusion = knn_config["neighbour_fusion"]
    neighbours = fuse_neighbours_rrf(
        {"lexical": lexical_neighbours, "dense": dense_neighbours},
        {
            "lexical": float(neighbour_fusion["lexical_weight"]),
            "dense": float(neighbour_fusion["dense_weight"]),
        },
        rrf_k=int(neighbour_fusion["rrf_k"]),
    )
    knn_ranking = neighbours_to_article_rankings(
        neighbours,
        calibration,
        top_k_neighbours=int(knn_config["top_k"]),
        top_k_articles=100,
    )
    knn_ranking.to_parquet(run_dir / "knn_rankings.parquet", index=False)
    timings["knn"] = time.perf_counter() - started

    fusion_config = config["fusion"]
    hybrid = weighted_rrf(
        {"bm25f": bm25_ranking, "dense": dense_ranking, "knn": knn_ranking},
        fusion_config["inputs"],
        rrf_k=int(fusion_config["rrf_k"]),
        top_k=100,
    )
    hybrid.to_parquet(run_dir / "hybrid_rankings.parquet", index=False)

    reranker_config = config["reranker"]
    candidates = hybrid[hybrid["rank"] <= int(reranker_config["candidate_depth"])].copy()
    contexts = dense.best_chunk_texts(
        test_dense,
        candidates,
        n_chunks=int(reranker_config["chunks_per_article"]),
    )
    started = time.perf_counter()
    rankings = rerank_with_cross_encoder(
        test,
        candidates,
        contexts,
        model_name=str(reranker_config["model_name"]),
        batch_size=int(reranker_config["batch_size"]),
        max_length=int(reranker_config["max_length"]),
        device=dense.device,
        source="fixed_cross_encoder",
    )
    timings["reranker"] = time.perf_counter() - started
    rankings.to_parquet(final_rankings_path, index=False)

    answers = _rankings_to_answer(rankings, test, articles, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    answers.to_csv(output_path, index=False)
    manifest = {
        "fingerprint": fingerprint,
        "cache_hit": False,
        "architecture": "HTML + SentencePiece BM25F + dense + calibration kNN + RRF + cross-encoder",
        "use_ocr": use_ocr,
        "dense_model": dense_config["model_name"],
        "reranker_model": reranker_config["model_name"],
        "answer_csv": str(output_path),
        "rows": len(answers),
        "top_k": top_k,
        "timings_seconds": timings,
    }
    save_json(manifest, run_dir / "manifest.json")
    return output_path, manifest
