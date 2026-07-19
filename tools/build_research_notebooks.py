from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "jupyter-notebook"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


SETUP = r'''
from pathlib import Path
import os, sys, json, time, copy, itertools, importlib.util, subprocess

def locate_project(start=Path.cwd()):
    for parent in [start.resolve(), *start.resolve().parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src/avito_retriever").exists():
            return parent
    candidate = Path("/content/avito-retriever")
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Clone/open avito-retriever or change PROJECT_ROOT in this cell")

PROJECT_ROOT = locate_project()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

IN_COLAB = module_available("google.colab")
if IN_COLAB and os.environ.get("AVITO_AUTO_INSTALL", "1") != "0":
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-e",
        f"{PROJECT_ROOT}[lexical,neural,dev]",
    ])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from avito_retriever.experiments.notebook import resolve_data_dir, result_dir, highlight_best, save_json, save_yaml

DATA_DIR = resolve_data_dir(PROJECT_ROOT)
SEED = 42
np.random.seed(SEED)
print("Project:", PROJECT_ROOT)
print("Data:", DATA_DIR)
'''


def write(name: str, cells: list[dict]) -> None:
    path = OUT / name
    template = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    template["cells"] = cells
    path.write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding="utf-8")


write(
    "00_data_html_ocr_audit.ipynb",
    [
        md('''# 00 — Data, HTML and OCR Audit

**Question.** What information is present in the corpus, which HTML structures must be preserved, and how much useful image context is available?

**Success criteria.** Produce a validated parsed corpus, compact quality tables, and reusable artifacts for all later notebooks.'''),
        md("## Setup and reproducibility"),
        code(SETUP),
        code('''from avito_retriever.data.io import load_articles, load_calibration, load_test
from avito_retriever.preprocessing.html import parse_articles, FIELD_COLUMNS
from avito_retriever.preprocessing.images import extract_image_manifest

OUT = result_dir(PROJECT_ROOT, "00_data_html_ocr_audit")
articles = load_articles(DATA_DIR)
calibration = load_calibration(DATA_DIR)
test = load_test(DATA_DIR)'''),
        md("## Dataset contract"),
        code('''summary = pd.DataFrame([
    {"dataset": "articles", "rows": len(articles), "columns": ", ".join(articles.columns), "missing": int(articles.isna().sum().sum())},
    {"dataset": "calibration", "rows": len(calibration), "columns": ", ".join(calibration.columns), "missing": int(calibration.isna().sum().sum())},
    {"dataset": "test", "rows": len(test), "columns": ", ".join(test.columns), "missing": int(test.isna().sum().sum())},
])
display(summary)
assert articles.article_id.is_unique and calibration.query_id.is_unique and test.query_id.is_unique'''),
        md("## HTML structure inventory"),
        code('''from bs4 import BeautifulSoup
from collections import Counter

tags, classes = Counter(), Counter()
for html in articles.body:
    soup = BeautifulSoup(str(html), "lxml")
    tags.update(tag.name for tag in soup.find_all(True))
    classes.update(cls for tag in soup.find_all(True) for cls in tag.get("class", []))

html_inventory = pd.DataFrame(tags.most_common(25), columns=["tag", "count"])
class_inventory = pd.DataFrame(classes.most_common(20), columns=["css_class", "count"])
display(html_inventory.head(15), class_inventory.head(15))'''),
        md("## Parse articles into retrieval fields"),
        code('''parsed_path = PROJECT_ROOT / "artifacts/cache/parsed_articles.parquet"
if parsed_path.exists():
    parsed = pd.read_parquet(parsed_path)
else:
    parsed = parse_articles(articles)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(parsed_path, index=False)

field_stats = pd.DataFrame([
    {"field": field, "non_empty_articles": int(parsed[field].str.len().gt(0).sum()),
     "median_chars": float(parsed[field].str.len().median()), "p95_chars": float(parsed[field].str.len().quantile(.95))}
    for field in FIELD_COLUMNS
])
display(field_stats)'''),
        md("## Image and OCR opportunity"),
        code('''image_manifest = extract_image_manifest(articles)
image_summary = pd.DataFrame([{
    "images": len(image_manifest),
    "articles_with_images": image_manifest.article_id.nunique(),
    "unique_urls": image_manifest.src.nunique(),
    "non_empty_alt": int(image_manifest.alt.str.len().gt(0).sum()),
    "unique_alt": image_manifest.alt.nunique(),
}])
display(image_summary)
display(image_manifest[["article_id", "alt", "src"]].head(12))'''),
        md("## Query and label profile"),
        code('''label_counts = calibration.ground_truth.str.split().str.len()
label_profile = pd.DataFrame([{
    "queries": len(calibration), "mean_labels": label_counts.mean(), "max_labels": label_counts.max(),
    "unique_relevant_articles": len({x for value in calibration.ground_truth for x in value.split()}),
    "exact_calibration_test_overlap": len(set(calibration.query_text.str.casefold()) & set(test.query_text.str.casefold())),
}])
display(label_profile)'''),
        md("## Persist audit artifacts"),
        code('''summary.to_csv(OUT / "dataset_summary.csv", index=False)
field_stats.to_csv(OUT / "field_statistics.csv", index=False)
html_inventory.to_csv(OUT / "html_tags.csv", index=False)
class_inventory.to_csv(OUT / "html_classes.csv", index=False)
image_manifest.to_parquet(OUT / "image_manifest.parquet", index=False)
audit = {"parsed_articles": str(parsed_path), **image_summary.iloc[0].to_dict(), **label_profile.iloc[0].to_dict()}
save_json(audit, OUT / "audit.json")
print(f"Saved audit artifacts to {OUT}")'''),
        md('''## Interpretation

Use `field_statistics.csv` to detect empty/noisy fields and `image_manifest.parquet` as the input to the OCR ablation. Later notebooks must read `parsed_articles.parquet`; they do not reparse HTML.'''),
    ],
)


write(
    "01_sentencepiece_bm25f_search.ipynb",
    [
        md('''# 01 — SentencePiece and BM25F Search

**Question.** Which SentencePiece representation and BM25F parameters provide the strongest lexical retrieval?

Model selection uses only the `tune` queries. The reserved `confirm` split is shown once for the selected variant.'''),
        md("## Setup"), code(SETUP),
        code('''from avito_retriever.config import load_config
from avito_retriever.data.io import load_calibration
from avito_retriever.preprocessing.html import FIELD_COLUMNS
from avito_retriever.preprocessing.normalize import normalize_lexical
from avito_retriever.tokenization.sentencepiece import train_or_load
from avito_retriever.retrieval.bm25f import BM25FIndex
from avito_retriever.evaluation.metrics import evaluate_rankings
from avito_retriever.evaluation.selection import make_tune_confirm_split, metric_on_split

OUT = result_dir(PROJECT_ROOT, "01_sentencepiece_bm25f_search")
parsed = pd.read_parquet(PROJECT_ROOT / "artifacts/cache/parsed_articles.parquet")
calibration = load_calibration(DATA_DIR)
split = make_tune_confirm_split(calibration, n_folds=5, confirm_fold=0, seed=SEED)
split.to_csv(OUT / "selection_split.csv", index=False)
base = load_config(PROJECT_ROOT / "configs/experiments/baseline_bm25f.yaml")'''),
        md("## Search space"),
        code('''MODEL_TYPES = ["unigram", "bpe"]
VOCAB_SIZES = [4000, 8000, 16000]
K1_VALUES = [0.8, 1.2, 1.5, 2.0]
TITLE_WEIGHTS = [3.0, 5.0, 8.0]

normalization = base["preprocessing"]["normalization"]
lexical = parsed.copy()
for field in FIELD_COLUMNS:
    lexical[field] = lexical[field].fillna("").map(lambda x: normalize_lexical(x, normalization))
queries = calibration.copy()
queries["query_text"] = queries.query_text.map(lambda x: normalize_lexical(x, normalization))
training_texts = sum((lexical[field].tolist() for field in FIELD_COLUMNS), []) + queries.query_text.tolist()
print(f"Trials: {len(MODEL_TYPES)*len(VOCAB_SIZES)*len(K1_VALUES)*len(TITLE_WEIGHTS)}")'''),
        md("## Run resumable grid search"),
        code('''leaderboard_path = OUT / "bm25f_trials.csv"
done = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
done_ids = set(done.trial_id) if not done.empty else set()
rows = done.to_dict("records") if not done.empty else []

for model_type, vocab_size in itertools.product(MODEL_TYPES, VOCAB_SIZES):
    sp_cfg = {**base["sentencepiece"], "model_type": model_type, "vocab_size": vocab_size}
    tokenizer = train_or_load(training_texts, sp_cfg, PROJECT_ROOT / "artifacts/indexes/sentencepiece")
    for k1, title_weight in itertools.product(K1_VALUES, TITLE_WEIGHTS):
        trial_id = f"{model_type}-v{vocab_size}-k{k1}-tw{title_weight}"
        if trial_id in done_ids: continue
        fields = copy.deepcopy(base["retrieval"]["bm25f"]["fields"])
        fields["title"]["weight"] = title_weight
        started = time.perf_counter()
        index = BM25FIndex(fields, tokenizer.encode, k1=k1).fit(lexical)
        ranking = index.retrieve(queries, top_k=100, source=trial_id)
        metrics, per_query = evaluate_rankings(ranking, calibration)
        per_query.to_parquet(OUT / f"per_query_{trial_id}.parquet", index=False)
        rows.append({"trial_id": trial_id, "model_type": model_type, "vocab_size": vocab_size,
                     "k1": k1, "title_weight": title_weight,
                     "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"),
                     "map@10": metrics["map@10"], "recall@50": metrics["recall@50"],
                     "seconds": time.perf_counter()-started, "model_path": str(tokenizer.model_path)})
        pd.DataFrame(rows).to_csv(leaderboard_path, index=False)

trials = pd.DataFrame(rows).sort_values("tune_map@10", ascending=False).reset_index(drop=True)
display(highlight_best(trials.head(15), ["tune_map@10", "map@10", "recall@50"]))'''),
        md("## Select on tune, confirm once"),
        code('''best = trials.iloc[0].to_dict()
best_per_query = pd.read_parquet(OUT / f"per_query_{best['trial_id']}.parquet")
confirm_map = metric_on_split(best_per_query, split, "ap@10", "confirm")
selection = pd.DataFrame([{"selected_trial": best["trial_id"], "tune_map@10": best["tune_map@10"],
                           "confirm_map@10": confirm_map, "overall_map@10": best["map@10"],
                           "recall@50": best["recall@50"]}])
display(selection)'''),
        md("## Rebuild and save selected lexical ranking"),
        code('''tokenizer = train_or_load(training_texts, {**base["sentencepiece"], "model_type": best["model_type"],
                           "vocab_size": int(best["vocab_size"])}, PROJECT_ROOT / "artifacts/indexes/sentencepiece")
fields = copy.deepcopy(base["retrieval"]["bm25f"]["fields"])
fields["title"]["weight"] = float(best["title_weight"])
best_index = BM25FIndex(fields, tokenizer.encode, k1=float(best["k1"])).fit(lexical)
best_ranking = best_index.retrieve(queries, top_k=100, source="bm25f_best")
best_ranking.to_parquet(OUT / "best_bm25f_rankings.parquet", index=False)
best_per_query.to_parquet(OUT / "best_bm25f_per_query.parquet", index=False)
best_config = {"sentencepiece": {"model_type": best["model_type"], "vocab_size": int(best["vocab_size"])},
               "bm25f": {"k1": float(best["k1"]), "fields": fields}}
save_yaml(best_config, OUT / "best_bm25f_config.yaml")
save_json({"sentencepiece_model": str(tokenizer.model_path)}, OUT / "best_artifacts.json")'''),
        md("## Visual sensitivity check"),
        code('''pivot = trials.groupby(["vocab_size", "k1"])["tune_map@10"].max().unstack()
display(pivot.style.format("{:.4f}").highlight_max(axis=None, color="#c6efce"))
pivot.plot(marker="o", figsize=(8,4), title="Best tune MAP@10 by vocabulary and k1")
plt.ylabel("MAP@10"); plt.tight_layout(); plt.show()'''),
        md('''## Decision rule

The selected BM25F configuration is frozen in `best_bm25f_config.yaml`. Notebook 02 uses that ranking unchanged while testing dense, kNN and fusion.'''),
    ],
)


write(
    "02_dense_knn_fusion_search.ipynb",
    [
        md('''# 02 — Dense, kNN and Fusion Search

**Question.** Which dense model, chunking, calibration-query kNN, and RRF weights improve the frozen BM25F baseline?

All selection remains tune-only; the chosen hybrid is evaluated once on confirm.'''),
        md("## Setup"), code(SETUP),
        code('''import yaml
from avito_retriever.config import load_config
from avito_retriever.data.io import load_calibration
from avito_retriever.retrieval.dense import DenseIndex, build_chunks
from avito_retriever.retrieval.knn import lexical_neighbours_oof, dense_neighbours_oof, fuse_neighbours_rrf, neighbours_to_article_rankings
from avito_retriever.evaluation.cv import make_grouped_query_folds
from avito_retriever.evaluation.metrics import evaluate_rankings
from avito_retriever.evaluation.selection import metric_on_split
from avito_retriever.fusion.rrf import weighted_rrf
from avito_retriever.tokenization.sentencepiece import SentencePieceTokenizer

OUT = result_dir(PROJECT_ROOT, "02_dense_knn_fusion_search")
PREV = result_dir(PROJECT_ROOT, "01_sentencepiece_bm25f_search")
parsed = pd.read_parquet(PROJECT_ROOT / "artifacts/cache/parsed_articles.parquet")
calibration = load_calibration(DATA_DIR)
split = pd.read_csv(PREV / "selection_split.csv")
folds = make_grouped_query_folds(calibration, n_folds=5, seed=SEED)
bm25f_ranking = pd.read_parquet(PREV / "best_bm25f_rankings.parquet")
sp_model = json.loads((PREV / "best_artifacts.json").read_text())["sentencepiece_model"]
sp = SentencePieceTokenizer(sp_model)
base = load_config(PROJECT_ROOT / "configs/experiments/hybrid_knn.yaml")'''),
        md("## Dense model and chunk search space"),
        code('''DENSE_MODELS = ["intfloat/multilingual-e5-small", "deepvk/USER2-base", "deepvk/USER-bge-m3"]
CHUNKS = [(128, 32), (256, 48), (384, 64)]
BATCH_SIZE = 64
print(f"Dense trials: {len(DENSE_MODELS)*len(CHUNKS)}; GPU recommended")'''),
        md("## Dense trials"),
        code('''progress_path = OUT / "dense_leaderboard.csv"
dense_rows = pd.read_csv(progress_path).to_dict("records") if progress_path.exists() else []
completed = {row["trial_id"] for row in dense_rows if pd.notna(row.get("tune_map@10"))}
dense_files = {}
for model_name, (size, overlap) in itertools.product(DENSE_MODELS, CHUNKS):
    trial = f"{model_name.split('/')[-1]}-{size}-{overlap}"
    path = OUT / f"dense_{trial}.parquet"; dense_files[trial] = str(path)
    if trial in completed and path.exists(): continue
    started = time.perf_counter()
    try:
        chunks = build_chunks(parsed, size_words=size, overlap_words=overlap)
        query_frame = calibration[["query_id", "query_text"]].copy()
        if "e5" in model_name.lower():
            chunks["text"] = "passage: " + chunks.text
            query_frame["query_text"] = "query: " + query_frame.query_text
        cfg = {**base["retrieval"]["dense"], "model_name": model_name, "batch_size": BATCH_SIZE}
        dense = DenseIndex(cfg, PROJECT_ROOT / "artifacts/embeddings" / trial)
        dense.fit_chunks(chunks)
        ranking = dense.retrieve(query_frame, top_k_articles=100)
        metrics, per_query = evaluate_rankings(ranking, calibration)
        ranking.to_parquet(path, index=False)
        per_query.to_parquet(OUT / f"dense_per_query_{trial}.parquet", index=False)
        dense_rows.append({"trial_id": trial, "model_name": model_name, "chunk_size": size,
                           "overlap": overlap, "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"),
                           "seconds": time.perf_counter()-started, "status": "ok", **metrics})
    except Exception as error:
        dense_rows.append({"trial_id": trial, "model_name": model_name, "chunk_size": size,
                           "overlap": overlap, "seconds": time.perf_counter()-started,
                           "status": f"error: {error}"})
    pd.DataFrame(dense_rows).drop_duplicates("trial_id", keep="last").to_csv(progress_path, index=False)
dense_results = pd.DataFrame(dense_rows).drop_duplicates("trial_id", keep="last")
dense_table = dense_results.dropna(subset=["tune_map@10"]).sort_values("tune_map@10", ascending=False).reset_index(drop=True)
assert not dense_table.empty, "Every dense trial failed; inspect dense_leaderboard.csv"
display(highlight_best(dense_table, ["tune_map@10", "map@10", "recall@50", "recall@100"]))'''),
        md("## SentencePiece-BM25 and dense kNN"),
        code('''best_dense = dense_table.iloc[0].to_dict()
best_dense_ranking = pd.read_parquet(dense_files[best_dense["trial_id"]])
_, best_dense_per_query = evaluate_rankings(best_dense_ranking, calibration)
best_dense_ranking.to_parquet(OUT / "best_dense_rankings.parquet", index=False)
best_dense_per_query.to_parquet(OUT / "best_dense_per_query.parquet", index=False)
lex_cfg = base["retrieval"]["knn"]["lexical"]
lex_neighbours = lexical_neighbours_oof(calibration, folds, sp.encode, lex_cfg["k1"], lex_cfg["b"], depth=30)

# Reuse query embeddings from the selected dense model.
chunks = build_chunks(parsed, int(best_dense["chunk_size"]), int(best_dense["overlap"]))
query_frame = calibration[["query_id", "query_text"]].copy()
if "e5" in best_dense["model_name"].lower():
    chunks["text"] = "passage: " + chunks.text; query_frame["query_text"] = "query: " + query_frame.query_text
dense = DenseIndex({**base["retrieval"]["dense"], "model_name": best_dense["model_name"], "batch_size": BATCH_SIZE},
                   PROJECT_ROOT / "artifacts/embeddings" / best_dense["trial_id"])
dense.fit_chunks(chunks)
query_embeddings = dense.query_embeddings(query_frame)
dense_neighbours = dense_neighbours_oof(calibration, folds, query_embeddings, depth=30)'''),
        code('''knn_rows, knn_rankings = [], {}
for k, lexical_weight, dense_weight in itertools.product([3,5,10,20], [0.5,1.0], [0.5,1.0]):
    trial = f"k{k}-l{lexical_weight}-d{dense_weight}"
    neighbours = fuse_neighbours_rrf({"lexical": lex_neighbours, "dense": dense_neighbours},
                                     {"lexical": lexical_weight, "dense": dense_weight}, rrf_k=20)
    ranking = neighbours_to_article_rankings(neighbours, calibration, top_k_neighbours=k, top_k_articles=100)
    metrics, per_query = evaluate_rankings(ranking, calibration)
    knn_rankings[trial] = ranking
    knn_rows.append({"trial_id": trial, "k": k, "lexical_weight": lexical_weight,
                     "dense_weight": dense_weight, "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"), **metrics})
knn_table = pd.DataFrame(knn_rows).sort_values("tune_map@10", ascending=False).reset_index(drop=True)
display(highlight_best(knn_table.head(15), ["tune_map@10", "map@10", "recall@50"]))'''),
        md("## Freeze the strongest kNN"),
        code('''best_knn = knn_table.iloc[0].to_dict()
best_knn_ranking = knn_rankings[best_knn["trial_id"]]
_, best_knn_per_query = evaluate_rankings(best_knn_ranking, calibration)
best_knn_ranking.to_parquet(OUT / "best_knn_rankings.parquet", index=False)
best_knn_per_query.to_parquet(OUT / "best_knn_per_query.parquet", index=False)'''),
        md('''## All BM25F / dense / kNN combinations

There are seven non-empty subsets of three retrievers. Single-component systems are already frozen. Each pair and the three-way hybrid receives its own tune-only RRF weight search.'''),
        code('''component_rankings = {
    "bm25f": bm25f_ranking,
    "dense": best_dense_ranking,
    "knn": best_knn_ranking,
}
WEIGHT_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
RRF_K_GRID = [20, 40, 60]
SUBSETS = [
    ("bm25f_dense", ["bm25f", "dense"]),
    ("bm25f_knn", ["bm25f", "knn"]),
    ("dense_knn", ["dense", "knn"]),
    ("bm25f_dense_knn", ["bm25f", "dense", "knn"]),
]

fusion_rows, fusion_rankings = [], {}
for subset_name, components in SUBSETS:
    # The first component is fixed at 1.0 because RRF weights are scale-invariant.
    free_components = components[1:]
    for free_weights in itertools.product(WEIGHT_GRID, repeat=len(free_components)):
        weights = {components[0]: 1.0, **dict(zip(free_components, free_weights))}
        for rrf_k in RRF_K_GRID:
            trial = subset_name + "-" + "-".join(f"{key}{value}" for key, value in weights.items()) + f"-r{rrf_k}"
            ranking = weighted_rrf({key: component_rankings[key] for key in components},
                                   weights, rrf_k=rrf_k, top_k=100)
            metrics, per_query = evaluate_rankings(ranking, calibration)
            fusion_rankings[trial] = ranking
            fusion_rows.append({"subset": subset_name, "trial_id": trial,
                                **{f"{key}_weight": weights.get(key, 0.0) for key in component_rankings},
                                "rrf_k": rrf_k,
                                "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"),
                                **metrics})

fusion_table = pd.DataFrame(fusion_rows).sort_values(["subset", "tune_map@10"], ascending=[True, False])
fusion_winners = fusion_table.groupby("subset", as_index=False, sort=False).head(1).reset_index(drop=True)
display(highlight_best(fusion_winners, ["tune_map@10", "map@10", "recall@50", "recall@100"]))'''),
        md("## Freeze every architecture variant and build reranker contexts"),
        code('''architecture_variants = {}
for row in fusion_winners.to_dict("records"):
    ranking = fusion_rankings[row["trial_id"]]
    _, per_query = evaluate_rankings(ranking, calibration)
    ranking.to_parquet(OUT / f"best_{row['subset']}_rankings.parquet", index=False)
    per_query.to_parquet(OUT / f"best_{row['subset']}_per_query.parquet", index=False)
    architecture_variants[row["subset"]] = {
        "trial_id": row["trial_id"],
        "bm25f_weight": float(row["bm25f_weight"]),
        "dense_weight": float(row["dense_weight"]),
        "knn_weight": float(row["knn_weight"]),
        "rrf_k": int(row["rrf_k"]),
        "tune_map@10": float(row["tune_map@10"]),
    }

best_fusion = architecture_variants["bm25f_dense_knn"]
hybrid = pd.read_parquet(OUT / "best_bm25f_dense_knn_rankings.parquet")
hybrid_per_query = pd.read_parquet(OUT / "best_bm25f_dense_knn_per_query.parquet")
hybrid.to_parquet(OUT / "best_hybrid_rankings.parquet", index=False)
hybrid_per_query.to_parquet(OUT / "best_hybrid_per_query.parquet", index=False)
candidates = hybrid[hybrid["rank"] <= 50].copy()
contexts = dense.best_chunk_texts(query_frame, candidates, n_chunks=2)
pd.DataFrame([{"query_id": q, "article_id": a, "text": text} for (q,a),text in contexts.items()]).to_parquet(
    OUT / "reranker_contexts.parquet", index=False)
selected = {"dense": best_dense, "knn": best_knn, "fusion": best_fusion,
            "architecture_variants": architecture_variants,
            "confirm_map@10": metric_on_split(hybrid_per_query, split, "ap@10", "confirm")}
save_json(selected, OUT / "best_hybrid_config.json")
dense_results.to_csv(OUT / "dense_leaderboard.csv", index=False)
knn_table.to_csv(OUT / "knn_leaderboard.csv", index=False); fusion_table.to_csv(OUT / "fusion_leaderboard.csv", index=False)
display(pd.DataFrame([selected["fusion"] | {"confirm_map@10": selected["confirm_map@10"]}]))'''),
        md("## Interpretation\nNotebook 03 receives one frozen top-50 candidate pool and identical contexts for every reranker. This makes reranker comparisons paired and fair."),
    ],
)


write(
    "03_reranker_comparison.ipynb",
    [
        md('''# 03 — Reranker Comparison

**Question.** Which local model below 1B parameters best reorders the same frozen hybrid candidate pool?

Cross-encoders and the 0.6B Qwen reranker are evaluated on identical query–article contexts.'''),
        md("## Setup"), code(SETUP),
        code('''from avito_retriever.data.io import load_calibration
from avito_retriever.evaluation.metrics import evaluate_rankings
from avito_retriever.evaluation.selection import metric_on_split
from avito_retriever.reranking.bi_encoder import rerank_with_bi_encoder
from avito_retriever.reranking.cross_encoder import rerank_with_cross_encoder

OUT = result_dir(PROJECT_ROOT, "03_reranker_comparison")
PREV = result_dir(PROJECT_ROOT, "02_dense_knn_fusion_search")
LEX = result_dir(PROJECT_ROOT, "01_sentencepiece_bm25f_search")
calibration = load_calibration(DATA_DIR)
split = pd.read_csv(LEX / "selection_split.csv")
hybrid = pd.read_parquet(PREV / "best_hybrid_rankings.parquet")
candidates = hybrid[hybrid["rank"] <= 50].copy()
contexts_frame = pd.read_parquet(PREV / "reranker_contexts.parquet")
contexts = {(int(r.query_id), int(r.article_id)): r.text for r in contexts_frame.itertuples(index=False)}
CROSS_ENCODERS = ["DiTy/cross-encoder-russian-msmarco", "BAAI/bge-reranker-v2-m3",
                  "Alibaba-NLP/gte-multilingual-reranker-base"]
BI_ENCODERS = ["intfloat/multilingual-e5-base",
               "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"'''),
        md("## Bi-encoder reranker sweep\nThese models score the same top-50 query–context pairs independently, so they are compared fairly with the cross-encoders."),
        code('''rows, rankings_by_model = [], {}
for model_name in BI_ENCODERS:
    started = time.perf_counter()
    try:
        ranking = rerank_with_bi_encoder(calibration, candidates, contexts, model_name,
                                          batch_size=64 if DEVICE=="cuda" else 16,
                                          device=DEVICE, source=model_name)
        metrics, per_query = evaluate_rankings(ranking, calibration)
        trial = model_name.split("/")[-1]
        ranking.to_parquet(OUT / f"rankings_{trial}.parquet", index=False)
        per_query.to_parquet(OUT / f"per_query_{trial}.parquet", index=False)
        rankings_by_model[model_name] = ranking
        rows.append({"model": model_name, "kind": "bi_encoder",
                     "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"),
                     "seconds": time.perf_counter()-started, **metrics})
    except Exception as error:
        rows.append({"model": model_name, "kind": "bi_encoder", "status": f"error: {error}"})
pd.DataFrame(rows).to_csv(OUT / "reranker_progress.csv", index=False)
display(pd.DataFrame(rows))'''),
        md("## Cross-encoder sweep"),
        code('''for model_name in CROSS_ENCODERS:
    started = time.perf_counter()
    try:
        ranking = rerank_with_cross_encoder(calibration, candidates, contexts, model_name,
                                            batch_size=32 if DEVICE=="cuda" else 8, max_length=512,
                                            device=DEVICE, source=model_name)
        metrics, per_query = evaluate_rankings(ranking, calibration)
        trial = model_name.split("/")[-1]
        ranking.to_parquet(OUT / f"rankings_{trial}.parquet", index=False)
        per_query.to_parquet(OUT / f"per_query_{trial}.parquet", index=False)
        rankings_by_model[model_name] = ranking
        rows.append({"model": model_name, "kind": "cross_encoder",
                     "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"),
                     "seconds": time.perf_counter()-started, **metrics})
    except Exception as error:
        rows.append({"model": model_name, "kind": "cross_encoder", "status": f"error: {error}"})
    pd.DataFrame(rows).to_csv(OUT / "reranker_progress.csv", index=False)
table = pd.DataFrame(rows).dropna(subset=["tune_map@10"]).sort_values("tune_map@10", ascending=False).reset_index(drop=True)
assert not table.empty, "Every reranker failed; inspect reranker_progress.csv"
display(highlight_best(table, ["tune_map@10", "map@10", "recall@20"]))'''),
        md("## Optional Qwen3-Reranker-0.6B\nSet `RUN_QWEN=True` on a GPU runtime. The cell uses the model's yes/no relevance probability."),
        code('''RUN_QWEN = DEVICE == "cuda"
QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"

def qwen_scores(pairs, model_name=QWEN_MODEL, batch_size=8, max_length=2048):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                  device_map="auto").eval()
    no_id, yes_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    system = 'Judge whether the Document meets the requirements based on the Query. Answer only "yes" or "no".'
    values = []
    for start in range(0, len(pairs), batch_size):
        prompts = [f"<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n<Query>: {q}\\n<Document>: {d}<|im_end|>\\n<|im_start|>assistant\\n"
                   for q,d in pairs[start:start+batch_size]]
        batch = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad(): logits = model(**batch).logits[:, -1, [no_id, yes_id]]
        values.extend(torch.softmax(logits.float(), dim=-1)[:,1].cpu().tolist())
    return np.asarray(values)

if RUN_QWEN:
    try:
        qmap = dict(zip(calibration.query_id.astype(int), calibration.query_text.astype(str)))
        ordered = candidates.sort_values(["query_id","rank"]).reset_index(drop=True)
        pairs = [(qmap[int(r.query_id)], contexts[(int(r.query_id),int(r.article_id))]) for r in ordered.itertuples()]
        ordered["score"] = qwen_scores(pairs); ordered["source"] = QWEN_MODEL
        ordered["rank"] = ordered.groupby("query_id").score.rank(method="first", ascending=False).astype(int)
        qwen_ranking = ordered[["query_id","article_id","score","rank","source"]].sort_values(["query_id","rank"])
        metrics, per_query = evaluate_rankings(qwen_ranking, calibration)
        qwen_ranking.to_parquet(OUT / "rankings_Qwen3-Reranker-0.6B.parquet", index=False)
        per_query.to_parquet(OUT / "per_query_Qwen3-Reranker-0.6B.parquet", index=False)
        rows.append({"model": QWEN_MODEL, "kind": "llm_reranker",
                     "tune_map@10": metric_on_split(per_query, split, "ap@10", "tune"), **metrics})
        rankings_by_model[QWEN_MODEL] = qwen_ranking
    except Exception as error:
        rows.append({"model": QWEN_MODEL, "kind": "llm_reranker", "status": f"error: {error}"})
table = pd.DataFrame(rows).dropna(subset=["tune_map@10"]).sort_values("tune_map@10", ascending=False).reset_index(drop=True)
display(highlight_best(table, ["tune_map@10", "map@10"]))'''),
        md("## Freeze winner and report confirm"),
        code('''best = table.iloc[0].to_dict(); selected = rankings_by_model[best["model"]]
metrics, per_query = evaluate_rankings(selected, calibration)
selected.to_parquet(OUT / "best_reranked_rankings.parquet", index=False)
per_query.to_parquet(OUT / "best_reranked_per_query.parquet", index=False)
best["confirm_map@10"] = metric_on_split(per_query, split, "ap@10", "confirm")
table.to_csv(OUT / "reranker_leaderboard.csv", index=False); save_json(best, OUT / "best_reranker_config.json")
display(pd.DataFrame([best]))'''),
        md("## Interpretation\nThe winner is selected by tune MAP@10. Notebook 05 tests whether its confirm-set improvement over baseline is statistically credible."),
    ],
)


write(
    "04_ocr_ablation.ipynb",
    [
        md('''# 04 — OCR Ablation

**Question.** Does text recognized from help-center screenshots improve retrieval enough to justify the extra pipeline cost?

The notebook downloads and deduplicates images, runs PaddleOCR, rebuilds only the affected lexical field, and compares with the frozen no-OCR baseline.'''),
        md("## Setup"), code(SETUP),
        code('''import yaml
if IN_COLAB and not module_available("paddleocr"):
    import torch
    cuda = str(torch.version.cuda or "")
    if torch.cuda.is_available():
        paddle_index = "cu126" if cuda.startswith("12") else "cu118"
        subprocess.check_call([sys.executable,"-m","pip","install","-q","paddlepaddle-gpu==3.2.0",
                               "-i",f"https://www.paddlepaddle.org.cn/packages/stable/{paddle_index}/"])
    else:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","paddlepaddle==3.2.0",
                               "-i","https://www.paddlepaddle.org.cn/packages/stable/cpu/"])
    subprocess.check_call([sys.executable,"-m","pip","install","-q","paddleocr[all]"])

from avito_retriever.data.io import load_articles, load_calibration
from avito_retriever.preprocessing.images import extract_image_manifest, download_images
from avito_retriever.preprocessing.ocr import paddle_ocr_manifest, aggregate_ocr_by_article
from avito_retriever.preprocessing.html import FIELD_COLUMNS
from avito_retriever.preprocessing.normalize import normalize_lexical
from avito_retriever.tokenization.sentencepiece import SentencePieceTokenizer
from avito_retriever.retrieval.bm25f import BM25FIndex
from avito_retriever.evaluation.metrics import evaluate_rankings
from avito_retriever.evaluation.selection import metric_on_split
from avito_retriever.fusion.rrf import weighted_rrf

OUT = result_dir(PROJECT_ROOT, "04_ocr_ablation")
LEX = result_dir(PROJECT_ROOT, "01_sentencepiece_bm25f_search")
HYB = result_dir(PROJECT_ROOT, "02_dense_knn_fusion_search")
articles = load_articles(DATA_DIR); calibration = load_calibration(DATA_DIR)
parsed = pd.read_parquet(PROJECT_ROOT / "artifacts/cache/parsed_articles.parquet")
split = pd.read_csv(LEX / "selection_split.csv")'''),
        md("## Download and cache unique images"),
        code('''manifest_path = OUT / "download_manifest.parquet"
if manifest_path.exists():
    downloaded = pd.read_parquet(manifest_path)
else:
    manifest = extract_image_manifest(articles)
    downloaded = download_images(manifest, PROJECT_ROOT / "artifacts/images")
    downloaded.to_parquet(manifest_path, index=False)
display(downloaded.download_status.value_counts().rename_axis("status").to_frame("images"))'''),
        md("## Run PaddleOCR (resumable at manifest level)"),
        code('''ocr_path = OUT / "ocr_manifest.parquet"
if ocr_path.exists():
    ocr_manifest = pd.read_parquet(ocr_path)
else:
    ocr_manifest = paddle_ocr_manifest(downloaded, language="ru", confidence_threshold=0.50)
    ocr_manifest.to_parquet(ocr_path, index=False)
display(ocr_manifest.ocr_status.value_counts().rename_axis("status").to_frame("images"))
print("Recognized non-empty images:", ocr_manifest.ocr_text.fillna("").str.len().gt(0).sum())'''),
        md("## Attach OCR and rerun BM25F"),
        code('''ocr_by_article = aggregate_ocr_by_article(ocr_manifest)
with_ocr = parsed.drop(columns=["image_ocr"]).merge(ocr_by_article, on="article_id", how="left")
with_ocr["image_ocr"] = with_ocr.image_ocr.fillna("")
with_ocr.to_parquet(OUT / "parsed_articles_with_ocr.parquet", index=False)

best_cfg = yaml.safe_load((LEX / "best_bm25f_config.yaml").read_text())
sp_path = json.loads((LEX / "best_artifacts.json").read_text())["sentencepiece_model"]
sp = SentencePieceTokenizer(sp_path)
normalization = {"unicode_form":"NFKC","lowercase":True,"replace_yo":True,"normalize_quotes":True,"normalize_dashes":True}
lexical = with_ocr.copy()
for field in FIELD_COLUMNS: lexical[field] = lexical[field].fillna("").map(lambda x: normalize_lexical(x, normalization))
queries = calibration.copy(); queries["query_text"] = queries.query_text.map(lambda x: normalize_lexical(x, normalization))
index = BM25FIndex(best_cfg["bm25f"]["fields"], sp.encode, k1=best_cfg["bm25f"]["k1"]).fit(lexical)
ocr_bm25 = index.retrieve(queries, top_k=100, source="bm25f_ocr")
ocr_metrics, ocr_per_query = evaluate_rankings(ocr_bm25, calibration)'''),
        md("## OCR fusion weights"),
        code('''hybrid = pd.read_parquet(HYB / "best_hybrid_rankings.parquet")
rows, rankings = [], {}
for weight in [0.0,0.25,0.5,0.75,1.0]:
    ranking = hybrid if weight == 0 else weighted_rrf({"hybrid":hybrid,"ocr":ocr_bm25},
                                                       {"hybrid":1.0,"ocr":weight}, rrf_k=40, top_k=100)
    metrics, per_query = evaluate_rankings(ranking, calibration)
    rows.append({"ocr_weight":weight,"tune_map@10":metric_on_split(per_query,split,"ap@10","tune"),**metrics})
    rankings[weight] = (ranking, per_query)
table = pd.DataFrame(rows).sort_values("tune_map@10",ascending=False).reset_index(drop=True)
display(highlight_best(table,["tune_map@10","map@10","recall@50"]))'''),
        md("## Freeze OCR decision"),
        code('''best = table.iloc[0].to_dict(); ranking, per_query = rankings[best["ocr_weight"]]
ranking.to_parquet(OUT / "best_ocr_rankings.parquet", index=False)
per_query.to_parquet(OUT / "best_ocr_per_query.parquet", index=False)
best["confirm_map@10"] = metric_on_split(per_query,split,"ap@10","confirm")
table.to_csv(OUT / "ocr_leaderboard.csv",index=False); save_json(best,OUT/"best_ocr_config.json")
display(pd.DataFrame([best]))'''),
        md("## Decision rule\nKeep OCR only if tune improves and the confirm result does not reverse the gain. Runtime and failed-download rate remain part of the engineering decision."),
    ],
)


write(
    "05_final_architecture_significance.ipynb",
    [
        md('''# 05 — Final Architecture and Significance

**Question.** Which architecture wins after parameter selection, and is the improvement over BM25F statistically supported on untouched confirm queries?

This notebook is the only place where all architecture-level confirm results are compared.'''),
        md("## Setup"), code(SETUP),
        code('''from avito_retriever.evaluation.statistics import compare_paired_runs
from avito_retriever.evaluation.multiple import holm_bonferroni

R = PROJECT_ROOT / "artifacts/notebook_results"
OUT = result_dir(PROJECT_ROOT, "05_final_architecture_significance")
split = pd.read_csv(R / "01_sentencepiece_bm25f_search/selection_split.csv")
paths = {
 "BM25F": R/"01_sentencepiece_bm25f_search/best_bm25f_per_query.parquet",
 "Dense": R/"02_dense_knn_fusion_search/best_dense_per_query.parquet",
 "kNN": R/"02_dense_knn_fusion_search/best_knn_per_query.parquet",
 "BM25F + Dense": R/"02_dense_knn_fusion_search/best_bm25f_dense_per_query.parquet",
 "BM25F + kNN": R/"02_dense_knn_fusion_search/best_bm25f_knn_per_query.parquet",
 "Dense + kNN": R/"02_dense_knn_fusion_search/best_dense_knn_per_query.parquet",
 "BM25F + Dense + kNN": R/"02_dense_knn_fusion_search/best_bm25f_dense_knn_per_query.parquet",
 "Reranked": R/"03_reranker_comparison/best_reranked_per_query.parquet",
 "OCR fusion": R/"04_ocr_ablation/best_ocr_per_query.parquet",
}
systems = {name: pd.read_parquet(path) for name,path in paths.items() if path.exists()}
required = {"BM25F", "Dense", "kNN", "BM25F + Dense", "BM25F + kNN", "Dense + kNN", "BM25F + Dense + kNN"}
assert required <= systems.keys(), "Run notebooks 01 and 02 with the full architecture ablation first"'''),
        md("## Architecture table"),
        code('''components = {
 "BM25F": (True, False, False, False, False),
 "Dense": (False, True, False, False, False),
 "kNN": (False, False, True, False, False),
 "BM25F + Dense": (True, True, False, False, False),
 "BM25F + kNN": (True, False, True, False, False),
 "Dense + kNN": (False, True, True, False, False),
 "BM25F + Dense + kNN": (True, True, True, False, False),
 "Reranked": (True, True, True, True, False),
 "OCR fusion": (True, True, True, False, True),
}
rows=[]
for name,frame in systems.items():
    joined=frame.merge(split[["query_id","split"]],on="query_id")
    use_bm25f,use_dense,use_knn,use_reranker,use_ocr=components[name]
    rows.append({"architecture":name,"BM25F":use_bm25f,"Dense":use_dense,"kNN":use_knn,
                 "Reranker":use_reranker,"OCR":use_ocr,
                 "tune_map@10":joined.loc[joined.split=="tune","ap@10"].mean(),
                 "confirm_map@10":joined.loc[joined.split=="confirm","ap@10"].mean(),
                 "overall_map@10":joined["ap@10"].mean(),
                 "recall@50":joined["recall@50"].mean()})
architecture_table=pd.DataFrame(rows).sort_values("tune_map@10",ascending=False).reset_index(drop=True)
display(highlight_best(architecture_table,["tune_map@10","confirm_map@10","overall_map@10","recall@50"]))'''),
        md("## Incremental ablation gains"),
        code('''score_by_name=architecture_table.set_index("architecture")
comparisons=[
 ("Add Dense to BM25F","BM25F","BM25F + Dense"),
 ("Add kNN to BM25F","BM25F","BM25F + kNN"),
 ("Combine Dense and kNN","Dense","Dense + kNN"),
 ("Add kNN to BM25F + Dense","BM25F + Dense","BM25F + Dense + kNN"),
 ("Add Dense to BM25F + kNN","BM25F + kNN","BM25F + Dense + kNN"),
]
if "Reranked" in systems: comparisons.append(("Add reranker","BM25F + Dense + kNN","Reranked"))
if "OCR fusion" in systems: comparisons.append(("Add OCR","BM25F + Dense + kNN","OCR fusion"))
ablation_rows=[]
for experiment,parent,candidate in comparisons:
    ablation_rows.append({"experiment":experiment,"parent":parent,"candidate":candidate,
                          "tune_delta":score_by_name.loc[candidate,"tune_map@10"]-score_by_name.loc[parent,"tune_map@10"],
                          "confirm_delta":score_by_name.loc[candidate,"confirm_map@10"]-score_by_name.loc[parent,"confirm_map@10"]})
ablation_table=pd.DataFrame(ablation_rows)
display(ablation_table.style.format({"tune_delta":"{:+.4f}","confirm_delta":"{:+.4f}"})
        .background_gradient(subset=["tune_delta","confirm_delta"],cmap="RdYlGn",vmin=-.03,vmax=.03))'''),
        md("## Paired confirm-set tests against BM25F"),
        code('''confirm_ids=set(split.loc[split.split=="confirm","query_id"])
baseline=systems["BM25F"]; baseline=baseline[baseline.query_id.isin(confirm_ids)]
tests=[]
for name,frame in systems.items():
    if name=="BM25F": continue
    result=compare_paired_runs(baseline,frame[frame.query_id.isin(confirm_ids)],metric="ap@10",
                               bootstrap_samples=10000,permutation_samples=10000,seed=SEED)
    tests.append({"architecture":name,**result})
test_table=pd.DataFrame(tests)
test_table["holm_permutation_p"]=holm_bonferroni(test_table.paired_permutation_p.tolist())
display(test_table[["architecture","baseline_mean","candidate_mean","mean_difference","bootstrap_ci",
                    "paired_permutation_p","holm_permutation_p","wins","ties","losses"]])'''),
        md("## Paired tests for incremental additions"),
        code('''incremental_tests=[]
for experiment,parent,candidate in comparisons:
    result=compare_paired_runs(
        systems[parent][systems[parent].query_id.isin(confirm_ids)],
        systems[candidate][systems[candidate].query_id.isin(confirm_ids)],
        metric="ap@10",bootstrap_samples=10000,permutation_samples=10000,seed=SEED)
    incremental_tests.append({"experiment":experiment,"parent":parent,"candidate":candidate,**result})
incremental_test_table=pd.DataFrame(incremental_tests)
incremental_test_table["holm_permutation_p"]=holm_bonferroni(
    incremental_test_table.paired_permutation_p.tolist())
display(incremental_test_table[["experiment","parent","candidate","mean_difference","bootstrap_ci",
                                "paired_permutation_p","holm_permutation_p","wins","ties","losses"]])'''),
        md("## Distribution diagnostics"),
        code('''best_name=architecture_table.iloc[0].architecture
best_frame=systems[best_name].set_index("query_id"); base_frame=systems["BM25F"].set_index("query_id")
delta=(best_frame.loc[list(confirm_ids),"ap@10"]-base_frame.loc[list(confirm_ids),"ap@10"])
fig,axes=plt.subplots(1,2,figsize=(12,4))
delta.hist(bins=20,ax=axes[0]); axes[0].axvline(0,color="black"); axes[0].set_title(f"Confirm AP@10 delta: {best_name} - BM25F")
pd.DataFrame({name:frame.set_index("query_id").loc[list(confirm_ids),"ap@10"] for name,frame in systems.items()}).boxplot(ax=axes[1],rot=25)
axes[1].set_title("Confirm per-query AP@10"); plt.tight_layout(); plt.show()'''),
        md("## Human-readable conclusion"),
        code('''winner=architecture_table.iloc[0].to_dict()
winner_test=test_table.loc[test_table.architecture==winner["architecture"]].iloc[0] if winner["architecture"]!="BM25F" else None
if winner_test is None:
    conclusion="BM25F remained the strongest tune-selected architecture; added complexity was not justified."
else:
    low,high=winner_test.bootstrap_ci
    significant=(winner_test.holm_permutation_p<0.05 and low>0)
    conclusion=(f"Selected architecture: {winner['architecture']}. Confirm MAP@10 = {winner['confirm_map@10']:.4f}; "
                f"delta vs BM25F = {winner_test.mean_difference:+.4f}; 95% bootstrap CI [{low:.4f}, {high:.4f}]. "
                + ("The improvement is statistically supported." if significant else
                   "The improvement is not yet statistically conclusive after correction."))
display(Markdown("### Conclusion\\n"+conclusion))'''),
        md("## Save final decision"),
        code('''architecture_table.to_csv(OUT/"architecture_comparison.csv",index=False)
ablation_table.to_csv(OUT/"architecture_ablation_deltas.csv",index=False)
test_table.to_json(OUT/"significance_tests.json",orient="records",force_ascii=False,indent=2)
incremental_test_table.to_json(OUT/"incremental_significance_tests.json",orient="records",force_ascii=False,indent=2)
ranking_paths = {
 "BM25F": R/"01_sentencepiece_bm25f_search/best_bm25f_rankings.parquet",
 "Dense": R/"02_dense_knn_fusion_search/best_dense_rankings.parquet",
 "kNN": R/"02_dense_knn_fusion_search/best_knn_rankings.parquet",
 "BM25F + Dense": R/"02_dense_knn_fusion_search/best_bm25f_dense_rankings.parquet",
 "BM25F + kNN": R/"02_dense_knn_fusion_search/best_bm25f_knn_rankings.parquet",
 "Dense + kNN": R/"02_dense_knn_fusion_search/best_dense_knn_rankings.parquet",
 "BM25F + Dense + kNN": R/"02_dense_knn_fusion_search/best_bm25f_dense_knn_rankings.parquet",
 "Reranked": R/"03_reranker_comparison/best_reranked_rankings.parquet",
 "OCR fusion": R/"04_ocr_ablation/best_ocr_rankings.parquet",
}
selected={"architecture":winner["architecture"],"ranking_source":str(ranking_paths[winner["architecture"]]),
 "conclusion":conclusion}
save_json(selected,OUT/"final_selected_architecture.json")
print("Saved final decision and statistical evidence")'''),
    ],
)


write(
    "06_final_fit_submission.ipynb",
    [
        md('''# 06 — Final Fit and Submission

**Objective.** Refit the selected architecture on all 500 calibration queries, predict all test queries, validate the submission contract, and write `answer.csv`.

Run this notebook only after notebook 05 has frozen the architecture and parameters.'''),
        md("## Setup"), code(SETUP),
        code('''from avito_retriever.data.io import load_articles, load_calibration, load_test
from avito_retriever.cli.validate_submission import validate_submission_frame

R=PROJECT_ROOT/"artifacts/notebook_results"
decision_path=R/"05_final_architecture_significance/final_selected_architecture.json"
assert decision_path.exists(), "Run notebook 05 first"
decision=json.loads(decision_path.read_text())
articles=load_articles(DATA_DIR); calibration=load_calibration(DATA_DIR); test=load_test(DATA_DIR)
display(pd.DataFrame([decision]))'''),
        md('''## Final prediction runner

The production runner uses the same cached artifacts and selected YAML/JSON parameters as notebooks 01–04. It must not read test labels or hardcode query IDs.'''),
        code('''from avito_retriever.pipeline.final import fit_predict_selected

rankings, run_manifest = fit_predict_selected(
    project_root=PROJECT_ROOT,
    data_dir=DATA_DIR,
    decision=decision,
    top_k=10,
)
display(pd.DataFrame([run_manifest]))
display(rankings.head(20))'''),
        md("## Build and validate answer.csv"),
        code('''answers=(rankings.sort_values(["query_id","rank"]).groupby("query_id").article_id
         .apply(lambda values:" ".join(map(str,list(dict.fromkeys(values))[:10]))).rename("answer").reset_index())
answers=test[["query_id"]].merge(answers,on="query_id",how="left")
validate_submission_frame(answers,test,set(articles.article_id.astype(int)),max_k=10)
submission_dir=PROJECT_ROOT/"artifacts/submissions"; submission_dir.mkdir(parents=True,exist_ok=True)
answer_path=submission_dir/"answer.csv"; answers.to_csv(answer_path,index=False)
print("Valid answer.csv:",answer_path,"rows=",len(answers))
display(answers.head(10))'''),
        md("## Reproducibility manifest"),
        code('''save_json({**run_manifest,"answer_csv":str(answer_path),"rows":len(answers),
           "max_articles":int(answers.answer.str.split().str.len().max())}, submission_dir/"submission_manifest.json")'''),
        md("## Create a shareable analysis bundle"),
        code('''from avito_retriever.experiments.bundle import build_analysis_bundle
bundle_path, bundle_manifest = build_analysis_bundle(PROJECT_ROOT)
print(f"Send this file for analysis: {bundle_path}")
print(f"Bundle size: {bundle_manifest['bundle_bytes']/1024**2:.2f} MB; files: {bundle_manifest['file_count']}")'''),
        md('''## Expected output

- `artifacts/submissions/answer.csv` — file to submit;
- `submission_manifest.json` — chosen architecture, models, parameters and artifact paths;
- `artifacts/analysis_bundles/avito-results-*.zip` — compact package to send for analysis;
- a validated preview with exactly 500 rows and no invalid article IDs.'''),
    ],
)


if __name__ == "__main__":
    print(f"Built research notebooks in {OUT}")
