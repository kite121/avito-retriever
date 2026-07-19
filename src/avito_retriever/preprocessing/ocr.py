from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _paddle_payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, dict):
        return value.get("res", value)
    if isinstance(result, dict):
        return result.get("res", result)
    return {}


def paddle_ocr_manifest(
    manifest: pd.DataFrame,
    language: str = "ru",
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        lang=language,
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    cache: dict[str, tuple[str, float, str]] = {}
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        path = str(record.local_path or "")
        if path and path not in cache:
            try:
                texts: list[str] = []
                scores: list[float] = []
                for result in engine.predict(path):
                    payload = _paddle_payload(result)
                    rec_texts = payload.get("rec_texts", [])
                    rec_scores = payload.get("rec_scores", [])
                    for text, score in zip(rec_texts, rec_scores):
                        if float(score) >= confidence_threshold and str(text).strip():
                            texts.append(str(text).strip())
                            scores.append(float(score))
                cache[path] = (
                    "\n".join(dict.fromkeys(texts)),
                    sum(scores) / len(scores) if scores else 0.0,
                    "success",
                )
            except Exception as error:
                cache[path] = ("", 0.0, f"error:{type(error).__name__}")
        text, confidence, status = cache.get(path, ("", 0.0, "missing"))
        row = record._asdict()
        row.update({"ocr_text": text, "ocr_confidence": confidence, "ocr_status": status})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_ocr_by_article(ocr_manifest: pd.DataFrame) -> pd.DataFrame:
    values = (
        ocr_manifest.assign(ocr_text=ocr_manifest["ocr_text"].fillna(""))
        .groupby("article_id", as_index=False)["ocr_text"]
        .agg(lambda items: "\n".join(dict.fromkeys(text for text in items if str(text).strip())))
    )
    return values

