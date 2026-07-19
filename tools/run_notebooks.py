from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "output" / "jupyter-notebook"
NOTEBOOKS = [
    "00_data_html_ocr_audit.ipynb",
    "01_sentencepiece_bm25f_search.ipynb",
    "02_dense_knn_fusion_search.ipynb",
    "03_reranker_comparison.ipynb",
    "04_ocr_ablation.ipynb",
    "05_final_architecture_significance.ipynb",
    "06_final_fit_submission.ipynb",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the Avito research notebooks in order.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip notebook 04")
    parser.add_argument("--start", type=int, default=0, choices=range(7))
    parser.add_argument("--stop", type=int, default=6, choices=range(7))
    parser.add_argument("--timeout", type=int, default=21600, help="Per-notebook timeout in seconds")
    args = parser.parse_args()
    if args.start > args.stop:
        parser.error("--start must be less than or equal to --stop")

    selected = NOTEBOOKS[args.start : args.stop + 1]
    if args.skip_ocr:
        selected = [name for name in selected if not name.startswith("04_")]

    environment = os.environ.copy()
    environment.setdefault("AVITO_DATA_DIR", str(PROJECT_ROOT / "data" / "candidate_data"))
    for position, name in enumerate(selected, start=1):
        path = NOTEBOOK_DIR / name
        print(f"[{position}/{len(selected)}] Running {name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(path),
                f"--ExecutePreprocessor.timeout={args.timeout}",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    print("Notebook execution completed.")
    print("Run `python tools/collect_results.py` to create the shareable ZIP.")


if __name__ == "__main__":
    main()
