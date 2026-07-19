from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence


class BM25Index:
    def __init__(self, tokenizer: Callable[[str], list[str]], k1: float = 1.2, b: float = 0.3):
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.n_docs = 0
        self.avg_length = 0.0

    def fit(self, documents: Sequence[tuple[int, str]]) -> "BM25Index":
        mutable: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths = {}
        for document_id, text in documents:
            tokens = self.tokenizer(text)
            self.doc_lengths[int(document_id)] = len(tokens)
            for term, frequency in Counter(tokens).items():
                mutable[term].append((int(document_id), int(frequency)))
        self.postings = dict(mutable)
        self.n_docs = len(self.doc_lengths)
        self.avg_length = sum(self.doc_lengths.values()) / max(self.n_docs, 1)
        return self

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(self.tokenizer(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for document_id, tf in postings:
                length_norm = (1.0 - self.b) + self.b * self.doc_lengths[document_id] / max(
                    self.avg_length, 1e-12
                )
                saturation = tf * (self.k1 + 1.0) / (tf + self.k1 * length_norm)
                scores[document_id] += idf * saturation
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

