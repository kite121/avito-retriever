from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import sentencepiece as spm


def _fingerprint(texts: Iterable[str], config: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    def encode(self, text: str) -> list[str]:
        return list(self.processor.encode(str(text), out_type=str))


def train_or_load(
    texts: list[str],
    config: dict[str, Any],
    output_dir: str | Path,
) -> SentencePieceTokenizer:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(texts, config)
    prefix = output / f"sentencepiece-{fingerprint}"
    model_path = prefix.with_suffix(".model")
    if model_path.exists():
        return SentencePieceTokenizer(model_path)

    corpus_path = output / f"sentencepiece-{fingerprint}.txt"
    corpus_path.write_text("\n".join(text.replace("\n", " ") for text in texts if text.strip()), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        model_type=str(config.get("model_type", "unigram")),
        vocab_size=int(config.get("vocab_size", 8000)),
        character_coverage=float(config.get("character_coverage", 1.0)),
        byte_fallback=bool(config.get("byte_fallback", True)),
        normalization_rule_name="identity",
        split_digits=True,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
    )
    return SentencePieceTokenizer(model_path)

