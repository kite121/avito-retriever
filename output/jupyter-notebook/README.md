# Research notebook runbook

Run the notebooks in numeric order. Each stage saves machine-readable results under
`artifacts/notebook_results/<notebook-name>/`, so the next notebook can reuse the
selected artifacts without copying values by hand.

## One-time setup

The dataset is already included under `data/candidate_data/`. From the project
directory, install the research dependencies:

```bash
python -m pip install -e ".[lexical,neural,dev]"
```

In Colab, place the complete project at `/content/avito-retriever`. The notebooks
install the standard research extras automatically and find the included Feather
files. Select a GPU runtime before notebooks 02, 03 and 06.

For notebook 04, install PaddlePaddle first and then OCR support. A common Colab
CUDA 11.8 setup is:

```bash
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install "paddleocr[all]"
```

If the Colab runtime exposes a different CUDA version, use the matching official
PaddlePaddle package index rather than forcing the command above.

1. `00_data_html_ocr_audit.ipynb` — validates the three Feather files, audits HTML
   and image coverage, and writes the parsed corpus.
2. `01_sentencepiece_bm25f_search.ipynb` — searches SentencePiece and BM25F
   parameters, then freezes the strongest lexical configuration.
3. `02_dense_knn_fusion_search.ipynb` — compares dense models and chunk sizes,
   tests SentencePiece-BM25 plus dense kNN, and selects weighted RRF parameters
   separately for every non-empty BM25F/dense/kNN component subset.
4. `03_reranker_comparison.ipynb` — compares local cross-encoders and optionally
   Qwen3-Reranker-0.6B on the same top-50 pool.
5. `04_ocr_ablation.ipynb` — downloads article images, runs PaddleOCR and checks
   whether OCR gives a reproducible gain. This experiment is optional but must be
   run before including OCR in the final comparison.
6. `05_final_architecture_significance.ipynb` — compares frozen architectures on
   the reserved confirm split, shows the seven BM25F/dense/kNN ablations, and
   applies paired tests with Holm correction.
7. `06_final_fit_submission.ipynb` — refits the selected architecture using all
   calibration queries and writes the validated `artifacts/submissions/answer.csv`.

Use a Colab GPU for notebooks 02–04 and 06 if a dense or reranker architecture is
selected. Notebook 01 is CPU-friendly but its complete grid can take time. Every
long stage caches models, embeddings, rankings, per-query metrics and leaderboards.

After notebook 06, send the generated
`artifacts/analysis_bundles/avito-results-*.zip` for analysis. The same file can be
created at any time with `python tools/collect_results.py`.
