from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import unquote, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from avito_retriever.preprocessing.normalize import normalize_natural


FIELD_COLUMNS = [
    "title",
    "headings",
    "ui_labels",
    "body",
    "tables",
    "image_alt",
    "image_ocr",
]

_NOISY_ALT = re.compile(
    r"^(?:\d+|плюс|карандаш|шестер[её]нка|иконка|icon[_ -]?.*|edit\.png|copy)$",
    re.IGNORECASE,
)
_FILELIKE_ALT = re.compile(r"\.(?:png|jpe?g|gif|webp)(?:\?.*)?$", re.IGNORECASE)


def _unique_text(values: Iterable[str]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_natural(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return "\n".join(result)


def _image_description(tag) -> str:
    candidates = [tag.get("alt", ""), tag.get("title", "")]
    src = str(tag.get("src", ""))
    if src:
        filename = unquote(urlparse(src).path.rsplit("/", 1)[-1])
        candidates.append(filename)

    useful: list[str] = []
    for value in candidates:
        cleaned = normalize_natural(value)
        if not cleaned or _NOISY_ALT.match(cleaned) or _FILELIKE_ALT.search(cleaned):
            continue
        useful.append(cleaned)
    return _unique_text(useful)


def parse_article_html(article_id: int, title: str, html: str) -> dict[str, str | int]:
    soup = BeautifulSoup(str(html), "lxml")
    for tag in soup.find_all(["input", "script", "style", "noscript", "svg"]):
        tag.decompose()

    heading_values: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        heading_values.append(tag.get_text(" ", strip=True))
    for tag in soup.find_all("headline"):
        heading_values.append(tag.get_text(" ", strip=True))
    for tag in soup.find_all("chunk"):
        heading_values.append(str(tag.get("title", "")))

    ui_values: list[str] = []
    for selector in ["label.tab-label", ".spoiler-text"]:
        ui_values.extend(tag.get_text(" ", strip=True) for tag in soup.select(selector))

    table_values: list[str] = []
    for row in soup.find_all("tr"):
        cells = [normalize_natural(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if any(cells):
            table_values.append(" | ".join(cells))

    image_values = [_image_description(tag) for tag in soup.find_all("img")]

    # Build body from a fresh tree so that structural fields are not counted twice.
    body_soup = BeautifulSoup(str(soup), "lxml")
    for tag in body_soup.find_all(["h1", "h2", "h3", "headline", "table", "img"]):
        tag.decompose()
    for selector in ["label.tab-label", ".spoiler-text"]:
        for tag in body_soup.select(selector):
            tag.decompose()
    for br in body_soup.find_all("br"):
        br.replace_with("\n")

    return {
        "article_id": int(article_id),
        "title": normalize_natural(title),
        "headings": _unique_text(heading_values),
        "ui_labels": _unique_text(ui_values),
        "body": normalize_natural(body_soup.get_text(" ", strip=True)),
        "tables": _unique_text(table_values),
        "image_alt": _unique_text(image_values),
        "image_ocr": "",
    }


def parse_articles(articles: pd.DataFrame) -> pd.DataFrame:
    rows = [
        parse_article_html(record.article_id, record.title, record.body)
        for record in articles.itertuples(index=False)
    ]
    parsed = pd.DataFrame(rows)
    if parsed["article_id"].duplicated().any():
        raise ValueError("Parsed articles contain duplicate article_id")
    if parsed[FIELD_COLUMNS].fillna("").apply(lambda row: not any(str(x).strip() for x in row), axis=1).any():
        raise ValueError("At least one parsed article has no searchable text")
    return parsed

