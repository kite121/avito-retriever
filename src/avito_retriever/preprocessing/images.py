from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_image_manifest(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for article in articles.itertuples(index=False):
        soup = BeautifulSoup(str(article.body), "lxml")
        for order, image in enumerate(soup.find_all("img")):
            src = str(image.get("src", "")).strip()
            if not src:
                continue
            rows.append(
                {
                    "article_id": int(article.article_id),
                    "image_order": order,
                    "src": src,
                    "alt": str(image.get("alt", "")).strip(),
                    "title": str(image.get("title", "")).strip(),
                }
            )
    return pd.DataFrame(rows)


def download_images(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    timeout: int = 30,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "avito-retriever-research/0.1"
    records: list[dict[str, object]] = []
    url_cache: dict[str, tuple[str | None, str, str]] = {}

    for row in manifest.itertuples(index=False):
        if row.src not in url_cache:
            try:
                response = session.get(row.src, timeout=timeout)
                response.raise_for_status()
                content = response.content
                sha256 = hashlib.sha256(content).hexdigest()
                suffix = Path(urlparse(row.src).path).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    suffix = ".img"
                path = destination / f"{sha256}{suffix}"
                if not path.exists():
                    path.write_bytes(content)
                url_cache[row.src] = (str(path), sha256, "success")
            except Exception as error:  # network failures must remain visible in the manifest
                url_cache[row.src] = (None, "", f"error:{type(error).__name__}")
        local_path, sha256, status = url_cache[row.src]
        record = row._asdict()
        record.update({"local_path": local_path, "sha256": sha256, "download_status": status})
        records.append(record)
    return pd.DataFrame(records)

