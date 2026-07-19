from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from avito_retriever.experiments.bundle import build_analysis_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect experiment tables, selected rankings and executed notebooks into one ZIP."
    )
    parser.add_argument("--output", type=Path, help="Optional output ZIP path")
    args = parser.parse_args()
    path, manifest = build_analysis_bundle(PROJECT_ROOT, args.output)
    print(f"Analysis bundle: {path}")
    print(f"Files: {manifest['file_count']}; size: {manifest['bundle_bytes'] / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
