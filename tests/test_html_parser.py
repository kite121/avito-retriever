from avito_retriever.preprocessing.html import parse_article_html


def test_html_parser_keeps_structural_fields_without_body_duplicates() -> None:
    html = """
    <h2>Отправка</h2>
    <label class="tab-label">Android</label>
    <div class="tab-panel"><p>Откройте заказ</p></div>
    <div class="spoiler"><div class="spoiler-text">Если не получается</div>
      <div class="spoiler-content"><p>Попробуйте снова</p></div></div>
    <table><tr><th>Служба</th><td>СДЭК</td></tr></table>
    <img alt="Ошибка оплаты" src="https://example.test/error.png" />
    """
    parsed = parse_article_html(1, "Как отправить", html)
    assert parsed["headings"] == "Отправка"
    assert "Android" in parsed["ui_labels"]
    assert "Если не получается" in parsed["ui_labels"]
    assert "Откройте заказ" in parsed["body"]
    assert "Android" not in parsed["body"]
    assert "Служба | СДЭК" in parsed["tables"]
    assert parsed["image_alt"] == "Ошибка оплаты"

