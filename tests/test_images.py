import pandas as pd

from avito_retriever.preprocessing.images import extract_image_manifest


def test_extract_image_manifest() -> None:
    articles = pd.DataFrame(
        {"article_id": [1], "title": ["x"], "body": ['<p>a</p><img src="https://x/a.png" alt="Экран">']}
    )
    manifest = extract_image_manifest(articles)
    assert len(manifest) == 1
    assert manifest.iloc[0]["alt"] == "Экран"

