from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from avito_retriever.contracts import RANKING_COLUMNS
from avito_retriever.preprocessing.normalize import normalize_natural


def build_chunks(parsed: pd.DataFrame, size_words: int = 256, overlap_words: int = 48) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    step = max(1, size_words - overlap_words)
    for article in parsed.itertuples(index=False):
        prefix = normalize_natural(
            "\n".join(
                value
                for value in [article.title, article.headings, article.ui_labels]
                if str(value).strip()
            )
        )
        content = normalize_natural(
            "\n".join(
                value
                for value in [article.body, article.tables, article.image_alt, article.image_ocr]
                if str(value).strip()
            )
        )
        words = content.split()
        if not words:
            words = prefix.split()
            prefix = ""
        starts = range(0, max(len(words), 1), step)
        for chunk_number, start in enumerate(starts):
            piece = " ".join(words[start : start + size_words])
            if not piece and chunk_number > 0:
                continue
            text = normalize_natural(f"{prefix}\n{piece}")
            rows.append(
                {
                    "chunk_id": len(rows),
                    "article_id": int(article.article_id),
                    "chunk_number": chunk_number,
                    "text": text,
                }
            )
            if start + size_words >= len(words):
                break
    return pd.DataFrame(rows)


def _texts_hash(texts: list[str], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class DenseIndex:
    def __init__(self, config: dict[str, Any], cache_dir: str | Path):
        self.config = config
        self.model_name = str(config["model_name"])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        requested_device = str(config.get("device", "auto"))
        if requested_device == "auto":
            try:
                import torch

                requested_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested_device = "cpu"
        self.device = requested_device
        self.model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )
        self.chunks: pd.DataFrame | None = None
        self.chunk_embeddings: np.ndarray | None = None

    def encode(self, texts: list[str], cache_name: str | None = None) -> np.ndarray:
        cache_path: Path | None = None
        if cache_name:
            cache_path = self.cache_dir / f"{cache_name}-{_texts_hash(texts, self.model_name)}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        embeddings = self.model.encode(
            texts,
            batch_size=int(self.config.get("batch_size", 32)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        if cache_path:
            np.save(cache_path, embeddings)
        return embeddings

    def fit_chunks(self, chunks: pd.DataFrame) -> None:
        self.chunks = chunks.reset_index(drop=True)
        self.chunk_embeddings = self.encode(self.chunks["text"].tolist(), cache_name="chunks")

    def retrieve(self, queries: pd.DataFrame, top_k_articles: int = 100) -> pd.DataFrame:
        if self.chunks is None or self.chunk_embeddings is None:
            raise RuntimeError("DenseIndex.fit_chunks must be called first")
        query_embeddings = self.encode(queries["query_text"].astype(str).tolist(), cache_name="queries")
        article_ids = self.chunks["article_id"].astype(int).to_numpy()
        rows: list[dict[str, int | float | str]] = []
        for query, vector in zip(queries.itertuples(index=False), query_embeddings):
            chunk_scores = self.chunk_embeddings @ vector
            article_scores: dict[int, float] = defaultdict(lambda: -np.inf)
            for article_id, score in zip(article_ids, chunk_scores):
                if score > article_scores[int(article_id)]:
                    article_scores[int(article_id)] = float(score)
            ranked = sorted(article_scores.items(), key=lambda item: (-item[1], item[0]))[
                :top_k_articles
            ]
            for rank, (article_id, score) in enumerate(ranked, start=1):
                rows.append(
                    {
                        "query_id": int(query.query_id),
                        "article_id": article_id,
                        "score": score,
                        "rank": rank,
                        "source": "dense",
                    }
                )
        return pd.DataFrame(rows, columns=RANKING_COLUMNS)

    def query_embeddings(self, queries: pd.DataFrame) -> np.ndarray:
        return self.encode(queries["query_text"].astype(str).tolist(), cache_name="queries")

    def best_chunk_texts(self, queries: pd.DataFrame, candidates: pd.DataFrame, n_chunks: int = 2) -> dict[tuple[int, int], str]:
        if self.chunks is None or self.chunk_embeddings is None:
            raise RuntimeError("DenseIndex.fit_chunks must be called first")
        query_embeddings = self.query_embeddings(queries)
        query_position = {int(query_id): index for index, query_id in enumerate(queries["query_id"])}
        article_indices: dict[int, np.ndarray] = {
            int(article_id): group.index.to_numpy()
            for article_id, group in self.chunks.groupby("article_id", sort=False)
        }
        result: dict[tuple[int, int], str] = {}
        for row in candidates.itertuples(index=False):
            query_id, article_id = int(row.query_id), int(row.article_id)
            indices = article_indices[article_id]
            scores = self.chunk_embeddings[indices] @ query_embeddings[query_position[query_id]]
            count = min(n_chunks, len(indices))
            selected = np.argpartition(-scores, count - 1)[:count]
            selected = selected[np.argsort(-scores[selected])]
            texts = self.chunks.iloc[indices[selected]]["text"].tolist()
            result[(query_id, article_id)] = "\n\n".join(texts)
        return result
