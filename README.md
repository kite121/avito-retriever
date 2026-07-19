# Avito Retriever

Config-driven project for retrieving Avito help-center articles and evaluating
ranked `article_id` lists with MAP@10.

## Clone and run

The three candidate Feather files are included in `data/candidate_data/`, so a
fresh clone does not require a separate data download.

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL> retriever-repository
cd retriever-repository/avito-retriever
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
make test
make notebooks
```

The full run includes OCR and neural models and is best executed on a GPU machine.
Use `make notebooks-no-ocr` for a first run without OCR. Individual notebooks can
also be opened and executed in numeric order from `output/jupyter-notebook/`.

After the run, create one compact file for analysis:

```bash
make bundle
```

Send the resulting `artifacts/analysis_bundles/avito-results-*.zip`. It contains
executed notebooks, leaderboards, selected per-query rankings, metrics,
configuration and `answer.csv`, while excluding raw data, model weights, images
and embedding caches.

### Google Colab

1. Clone or upload the complete repository to `/content/avito-retriever`.
2. Choose a GPU runtime.
3. Open notebooks 00–06 and run them in numeric order in the same runtime.

The notebooks detect Colab, install the standard research extras automatically,
and use the included dataset. Notebook 04 separately installs the matching
PaddlePaddle package and PaddleOCR. Set `AVITO_AUTO_INSTALL=0` only when the
environment has already been prepared manually.

Because the dataset is now stored in Git, keep the repository private unless its
redistribution terms explicitly allow public mirroring.

The project separates data preparation, retrieval, fusion, reranking, evaluation,
and statistical comparison. Every ranking stage writes the same long-form table:

| query_id | article_id | score | rank | source |
|---:|---:|---:|---:|---|
| 1 | 1909 | 12.43 | 1 | bm25f |

This makes partial experiments first-class: a BM25F run can be evaluated without
OCR or neural models, cached dense results can be fused with new lexical weights,
and rerankers can be compared on an identical candidate pool.

## Layout

```text
configs/                 Resolved experiment settings
  base.yaml              Shared defaults and all feature switches
  experiments/           Small overrides for individual experiments
src/avito_retriever/
  data/                   Feather loading, schemas, validation
  preprocessing/          HTML, normalization, images, OCR, chunking
  tokenization/           Shared SentencePiece training and encoding
  retrieval/              BM25F, dense and calibration-query kNN
  fusion/                 RRF and score fusion
  reranking/              Cross-encoder and Qwen rerankers
  pipeline/               Stage orchestration and prediction
  evaluation/             MAP@10, candidate recall, CV, statistics
  tracking/               Run manifests, artifacts and timings
  cli/                    Independent command-line entry points
tests/                    Unit and contract tests
artifacts/                Generated/cacheable outputs; not committed
```

## Artifact contract

Each run lives under `artifacts/runs/<run_id>/` and may contain:

```text
config.resolved.yaml
manifest.json
rankings.parquet
per_query_metrics.parquet
metrics.json
timings.json
```

Downloaded images, OCR results, parsed articles, SentencePiece models, embeddings,
and indexes are cached separately and keyed by input/config fingerprints.

## Experiment flow

1. Prepare parsed article fields and optional OCR.
2. Train one SentencePiece model.
3. Build article BM25F and article dense indexes.
4. Build SentencePiece-BM25 and dense indexes over calibration queries.
5. Save every component ranking independently.
6. Fuse cached rankings with weighted RRF.
7. Rerank a frozen candidate pool.
8. Evaluate per query, aggregate MAP@10, and run paired comparisons.
9. Fit the selected configuration on all calibration queries and predict test.

Initial development should use `configs/experiments/baseline_bm25f.yaml`. Neural,
kNN, OCR, and reranking stages remain disabled until their individual baselines
are verified.

## Research notebooks

The complete tune → confirm → significance → submission workflow is in
`output/jupyter-notebook/`. Start with its `README.md` and run notebooks 00–06 in
order. Intermediate leaderboards and per-query results are written to
`artifacts/notebook_results/`; the final validated file is
`artifacts/submissions/answer.csv`.
