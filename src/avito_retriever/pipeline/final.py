from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from avito_retriever.config import load_config
from avito_retriever.contracts import RANKING_COLUMNS
from avito_retriever.data.io import load_calibration, load_test
from avito_retriever.fusion.rrf import weighted_rrf
from avito_retriever.preprocessing.html import FIELD_COLUMNS
from avito_retriever.preprocessing.normalize import normalize_lexical
from avito_retriever.reranking.cross_encoder import rerank_with_cross_encoder
from avito_retriever.reranking.bi_encoder import rerank_with_bi_encoder
from avito_retriever.retrieval.bm25 import BM25Index
from avito_retriever.retrieval.bm25f import BM25FIndex
from avito_retriever.retrieval.dense import DenseIndex, build_chunks
from avito_retriever.retrieval.knn import (
    NEIGHBOUR_COLUMNS,
    fuse_neighbours_rrf,
    neighbours_to_article_rankings,
)
from avito_retriever.tokenization.sentencepiece import SentencePieceTokenizer


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required selected-parameter artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_frame(
    parsed: pd.DataFrame, queries: pd.DataFrame, normalization: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lexical = parsed.copy()
    for field in FIELD_COLUMNS:
        lexical[field] = lexical[field].fillna("").map(
            lambda value: normalize_lexical(value, normalization)
        )
    normalised_queries = queries.copy()
    normalised_queries["query_text"] = normalised_queries["query_text"].map(
        lambda value: normalize_lexical(value, normalization)
    )
    return lexical, normalised_queries


def _bm25_neighbours(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    tokenizer,
    k1: float,
    b: float,
    depth: int,
) -> pd.DataFrame:
    index = BM25Index(tokenizer, k1=k1, b=b).fit(
        [(int(row.query_id), str(row.query_text)) for row in calibration.itertuples(index=False)]
    )
    rows: list[dict[str, Any]] = []
    for query in test.itertuples(index=False):
        for rank, (neighbour_id, score) in enumerate(
            index.search(str(query.query_text), depth), start=1
        ):
            rows.append(
                {
                    "query_id": int(query.query_id),
                    "neighbour_id": int(neighbour_id),
                    "score": float(score),
                    "rank": rank,
                    "source": "knn_lexical",
                }
            )
    return pd.DataFrame(rows, columns=NEIGHBOUR_COLUMNS)


def _dense_neighbours(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    calibration_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    depth: int,
) -> pd.DataFrame:
    calibration_ids = calibration["query_id"].astype(int).to_numpy()
    similarities = test_embeddings @ calibration_embeddings.T
    rows: list[dict[str, Any]] = []
    for position, query_id in enumerate(test["query_id"].astype(int)):
        count = min(depth, len(calibration_ids))
        selected = np.argpartition(-similarities[position], count - 1)[:count]
        selected = selected[np.argsort(-similarities[position, selected])]
        for rank, neighbour_position in enumerate(selected, start=1):
            rows.append(
                {
                    "query_id": int(query_id),
                    "neighbour_id": int(calibration_ids[neighbour_position]),
                    "score": float(similarities[position, neighbour_position]),
                    "rank": rank,
                    "source": "knn_dense",
                }
            )
    return pd.DataFrame(rows, columns=NEIGHBOUR_COLUMNS)


def _qwen_rerank(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    contexts: dict[tuple[int, int], str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The selected Qwen reranker requires a CUDA GPU for final fitting")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    no_id = tokenizer.convert_tokens_to_ids("no")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    system = (
        "Judge whether the Document meets the requirements based on the Query. "
        'Answer only "yes" or "no".'
    )
    query_by_id = dict(zip(queries.query_id.astype(int), queries.query_text.astype(str)))
    ordered = candidates.sort_values(["query_id", "rank"]).reset_index(drop=True)
    pairs = [
        (query_by_id[int(row.query_id)], contexts[(int(row.query_id), int(row.article_id))])
        for row in ordered.itertuples(index=False)
    ]
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        prompts = [
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n"
            for query, document in pairs[start : start + batch_size]
        ]
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, [no_id, yes_id]]
        scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    ordered["score"] = scores
    ordered["source"] = model_name
    ordered["rank"] = (
        ordered.groupby("query_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ordered[RANKING_COLUMNS].sort_values(["query_id", "rank"]).reset_index(drop=True)


def fit_predict_selected(
    project_root: str | Path,
    data_dir: str | Path,
    decision: dict[str, Any],
    top_k: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit the frozen notebook-selected architecture and rank test queries."""

    root = Path(project_root)
    results = root / "artifacts" / "notebook_results"
    lexical_dir = results / "01_sentencepiece_bm25f_search"
    hybrid_dir = results / "02_dense_knn_fusion_search"
    reranker_dir = results / "03_reranker_comparison"
    ocr_dir = results / "04_ocr_ablation"

    calibration = load_calibration(data_dir)
    test = load_test(data_dir)
    parsed_path = root / "artifacts" / "cache" / "parsed_articles.parquet"
    architecture = str(decision["architecture"])
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed article artifact is missing: {parsed_path}")
    parsed = pd.read_parquet(parsed_path)

    base = load_config(root / "configs" / "experiments" / "hybrid_knn.yaml")
    normalization = base["preprocessing"]["normalization"]
    lexical, test_lexical = _normalise_frame(parsed, test, normalization)
    calibration_lexical = calibration.copy()
    calibration_lexical["query_text"] = calibration_lexical["query_text"].map(
        lambda value: normalize_lexical(value, normalization)
    )

    bm25_config = yaml.safe_load(
        (lexical_dir / "best_bm25f_config.yaml").read_text(encoding="utf-8")
    )
    sentencepiece_path = _read_json(lexical_dir / "best_artifacts.json")[
        "sentencepiece_model"
    ]
    sentencepiece = SentencePieceTokenizer(sentencepiece_path)
    bm25 = BM25FIndex(
        bm25_config["bm25f"]["fields"],
        sentencepiece.encode,
        k1=float(bm25_config["bm25f"]["k1"]),
    ).fit(lexical)
    bm25_ranking = bm25.retrieve(test_lexical, top_k=100, source="bm25f_final")
    manifest: dict[str, Any] = {
        "architecture": architecture,
        "sentencepiece_model": str(sentencepiece_path),
        "bm25f": bm25_config["bm25f"],
        "calibration_queries": len(calibration),
        "test_queries": len(test),
    }
    if architecture == "BM25F":
        return bm25_ranking[bm25_ranking["rank"] <= top_k].copy(), manifest

    hybrid_config = _read_json(hybrid_dir / "best_hybrid_config.json")
    dense_choice = hybrid_config["dense"]
    dense_model = str(dense_choice["model_name"])
    chunks = build_chunks(
        parsed,
        size_words=int(dense_choice["chunk_size"]),
        overlap_words=int(dense_choice["overlap"]),
    )
    calibration_dense = calibration[["query_id", "query_text"]].copy()
    test_dense = test[["query_id", "query_text"]].copy()
    if "e5" in dense_model.lower():
        chunks["text"] = "passage: " + chunks["text"]
        calibration_dense["query_text"] = "query: " + calibration_dense["query_text"]
        test_dense["query_text"] = "query: " + test_dense["query_text"]
    dense_config = {
        **base["retrieval"]["dense"],
        "model_name": dense_model,
        "batch_size": int(base["retrieval"]["dense"].get("batch_size", 32)),
    }
    dense = DenseIndex(dense_config, root / "artifacts" / "embeddings" / "final")
    dense.fit_chunks(chunks)
    dense_ranking = dense.retrieve(test_dense, top_k_articles=100)
    manifest["dense"] = dense_choice
    if architecture == "Dense":
        return dense_ranking[dense_ranking["rank"] <= top_k].copy(), manifest

    knn_choice = hybrid_config["knn"]
    lexical_settings = base["retrieval"]["knn"]["lexical"]
    lexical_neighbours = _bm25_neighbours(
        calibration_lexical,
        test_lexical,
        sentencepiece.encode,
        k1=float(lexical_settings["k1"]),
        b=float(lexical_settings["b"]),
        depth=int(lexical_settings.get("depth", 30)),
    )
    calibration_embeddings = dense.encode(
        calibration_dense["query_text"].astype(str).tolist(), "final_calibration_queries"
    )
    test_embeddings = dense.encode(
        test_dense["query_text"].astype(str).tolist(), "final_test_queries"
    )
    dense_neighbours = _dense_neighbours(
        calibration,
        test,
        calibration_embeddings,
        test_embeddings,
        depth=int(base["retrieval"]["knn"]["dense"].get("depth", 30)),
    )
    fused_neighbours = fuse_neighbours_rrf(
        {"lexical": lexical_neighbours, "dense": dense_neighbours},
        {
            "lexical": float(knn_choice["lexical_weight"]),
            "dense": float(knn_choice["dense_weight"]),
        },
        rrf_k=int(base["retrieval"]["knn"]["neighbour_fusion"].get("rrf_k", 20)),
    )
    knn_ranking = neighbours_to_article_rankings(
        fused_neighbours,
        calibration,
        top_k_neighbours=int(knn_choice["k"]),
        top_k_articles=100,
    )
    manifest["knn"] = knn_choice
    if architecture == "kNN":
        return knn_ranking[knn_ranking["rank"] <= top_k].copy(), manifest

    component_rankings = {
        "bm25f": bm25_ranking,
        "dense": dense_ranking,
        "knn": knn_ranking,
    }
    architecture_to_subset = {
        "BM25F + Dense": "bm25f_dense",
        "BM25F + kNN": "bm25f_knn",
        "Dense + kNN": "dense_knn",
        "BM25F + Dense + kNN": "bm25f_dense_knn",
        "Hybrid": "bm25f_dense_knn",
    }
    variant_choices = hybrid_config.get("architecture_variants", {})
    if not variant_choices:
        legacy = hybrid_config["fusion"]
        variant_choices = {
            "bm25f_dense_knn": {
                "bm25f_weight": 1.0,
                "dense_weight": float(legacy["dense_weight"]),
                "knn_weight": float(legacy["knn_weight"]),
                "rrf_k": int(legacy["rrf_k"]),
            }
        }

    def build_variant(subset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        choice = variant_choices[subset]
        components = subset.split("_")
        weights = {name: float(choice[f"{name}_weight"]) for name in components}
        ranking = weighted_rrf(
            {name: component_rankings[name] for name in components},
            weights,
            rrf_k=int(choice["rrf_k"]),
            top_k=100,
        )
        return ranking, choice

    hybrid, fusion_choice = build_variant("bm25f_dense_knn")
    manifest["fusion"] = fusion_choice

    if architecture in architecture_to_subset:
        selected_subset = architecture_to_subset[architecture]
        selected_ranking, selected_choice = build_variant(selected_subset)
        manifest["selected_fusion"] = selected_choice
        return selected_ranking[selected_ranking["rank"] <= top_k].copy(), manifest

    if architecture == "OCR fusion":
        ocr_choice = _read_json(ocr_dir / "best_ocr_config.json")
        ocr_weight = float(ocr_choice["ocr_weight"])
        ocr_parsed_path = ocr_dir / "parsed_articles_with_ocr.parquet"
        if not ocr_parsed_path.exists():
            raise FileNotFoundError(f"OCR article artifact is missing: {ocr_parsed_path}")
        ocr_lexical, _ = _normalise_frame(
            pd.read_parquet(ocr_parsed_path), test, normalization
        )
        ocr_bm25 = BM25FIndex(
            bm25_config["bm25f"]["fields"],
            sentencepiece.encode,
            k1=float(bm25_config["bm25f"]["k1"]),
        ).fit(ocr_lexical)
        ocr_ranking = ocr_bm25.retrieve(test_lexical, top_k=100, source="bm25f_ocr_final")
        final = (
            hybrid
            if ocr_weight == 0
            else weighted_rrf(
                {"hybrid": hybrid, "ocr": ocr_ranking},
                {"hybrid": 1.0, "ocr": ocr_weight},
                rrf_k=40,
                top_k=100,
            )
        )
        manifest["ocr"] = ocr_choice
        return final[final["rank"] <= top_k].copy(), manifest

    if architecture != "Reranked":
        raise ValueError(f"Unsupported selected architecture: {architecture}")
    reranker_choice = _read_json(reranker_dir / "best_reranker_config.json")
    candidates = hybrid[hybrid["rank"] <= 50].copy()
    contexts = dense.best_chunk_texts(test_dense, candidates, n_chunks=2)
    reranker_model = str(reranker_choice["model"])
    if str(reranker_choice.get("kind")) == "bi_encoder":
        final = rerank_with_bi_encoder(
            test,
            candidates,
            contexts,
            reranker_model,
            batch_size=64 if dense.device == "cuda" else 16,
            device=dense.device,
            source=reranker_model,
        )
    elif reranker_model == "Qwen/Qwen3-Reranker-0.6B":
        final = _qwen_rerank(
            test,
            candidates,
            contexts,
            reranker_model,
            batch_size=8,
            max_length=2048,
        )
    else:
        final = rerank_with_cross_encoder(
            test,
            candidates,
            contexts,
            reranker_model,
            batch_size=32 if dense.device == "cuda" else 8,
            max_length=512,
            device=dense.device,
            source=reranker_model,
        )
    manifest["reranker"] = reranker_choice
    return final[final["rank"] <= top_k].copy(), manifest
