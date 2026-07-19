from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from avito_retriever.experiments.notebook import resolve_data_dir  # noqa: E402
from avito_retriever.pipeline.fixed import run_fixed_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed full retriever and create a validated answer.csv."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing articles.f, calibration.f and test.f",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "answer.csv",
        help="Destination CSV; defaults to answer.csv in the project root",
    )
    parser.add_argument("--ocr", action="store_true", help="Download images and add PaddleOCR text")
    parser.add_argument("--force", action="store_true", help="Ignore a cached final ranking")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve() if args.data_dir else resolve_data_dir(PROJECT_ROOT)
    answer_path, manifest = run_fixed_submission(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        output_csv=args.output,
        use_ocr=args.ocr,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nValidated submission: {answer_path}")


if __name__ == "__main__":
    main()
