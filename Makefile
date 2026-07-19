PYTHON ?= python3

.PHONY: install test notebooks notebooks-no-ocr answer answer-ocr bundle

install:
	$(PYTHON) -m pip install -e ".[lexical,neural,dev]"

test:
	$(PYTHON) -m pytest -q

notebooks:
	$(PYTHON) tools/run_notebooks.py

notebooks-no-ocr:
	$(PYTHON) tools/run_notebooks.py --skip-ocr

answer:
	$(PYTHON) tools/make_answer.py

answer-ocr:
	$(PYTHON) tools/make_answer.py --ocr

bundle:
	$(PYTHON) tools/collect_results.py
