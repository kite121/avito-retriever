from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping

import pandas as pd

from avito_retriever.contracts import RANKING_COLUMNS


class BM25FIndex:
    def __init__(
        self,
        field_config: Mapping[str, Mapping[str, float]],
        tokenizer: Callable[[str], list[str]],
        k1: float = 1.5,
    ):
        self.field_config = {field: dict(values) for field, values in field_config.items()}
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.article_ids: list[int] = []
        self.postings: dict[str, dict[int, dict[str, int]]] = {}
        self.doc_freq: dict[str, int] = {}
        self.field_lengths: dict[str, dict[int, int]] = {}
        self.avg_field_lengths: dict[str, float] = {}
        self.n_docs = 0

    def fit(self, articles: pd.DataFrame) -> "BM25FIndex":
        self.article_ids = articles["article_id"].astype(int).tolist()
        self.n_docs = len(self.article_ids)
        mutable: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        doc_terms: dict[int, set[str]] = defaultdict(set)
        self.field_lengths = {field: {} for field in self.field_config}

        for record in articles.itertuples(index=False):
            article_id = int(record.article_id)
            for field in self.field_config:
                tokens = self.tokenizer(str(getattr(record, field, "") or ""))
                self.field_lengths[field][article_id] = len(tokens)
                for term, frequency in Counter(tokens).items():
                    mutable[term][article_id][field] = int(frequency)
                    doc_terms[article_id].add(term)

        self.postings = {
            term: {article_id: dict(fields) for article_id, fields in documents.items()}
            for term, documents in mutable.items()
        }
        self.doc_freq = Counter(term for terms in doc_terms.values() for term in terms)
        self.avg_field_lengths = {
            field: sum(lengths.values()) / max(len(lengths), 1)
            for field, lengths in self.field_lengths.items()
        }
        return self

    def score(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(self.tokenizer(query)):
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for article_id, field_tfs in self.postings[term].items():
                weighted_tf = 0.0
                for field, tf in field_tfs.items():
                    settings = self.field_config[field]
                    weight = float(settings.get("weight", 1.0))
                    b = float(settings.get("b", 0.75))
                    length = self.field_lengths[field][article_id]
                    average = max(self.avg_field_lengths[field], 1e-12)
                    norm = (1.0 - b) + b * length / average
                    weighted_tf += weight * tf / max(norm, 1e-12)
                saturation = (self.k1 + 1.0) * weighted_tf / (self.k1 + weighted_tf)
                scores[article_id] += idf * saturation
        return dict(scores)

    def search(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        scores = self.score(query)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100, source: str = "bm25f") -> pd.DataFrame:
        rows: list[dict[str, int | float | str]] = []
        for record in queries.itertuples(index=False):
            for rank, (article_id, score) in enumerate(self.search(record.query_text, top_k), start=1):
                rows.append(
                    {
                        "query_id": int(record.query_id),
                        "article_id": int(article_id),
                        "score": float(score),
                        "rank": rank,
                        "source": source,
                    }
                )
        return pd.DataFrame(rows, columns=RANKING_COLUMNS)

