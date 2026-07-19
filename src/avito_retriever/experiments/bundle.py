from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _selected_result_files(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    result_root = project_root / "artifacts" / "notebook_results"
    if result_root.exists():
        for path in result_root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if path.suffix.lower() in {".csv", ".json", ".yaml", ".yml"}:
                candidates.append(path)
            elif path.suffix.lower() == ".parquet" and (
                name.startswith("best_") or "per_query" in name
            ):
                candidates.append(path)

    for root in [
        project_root / "artifacts" / "submissions",
        project_root / "artifacts" / "fixed_solution",
        project_root / "artifacts" / "runs",
        project_root / "configs",
        project_root / "output" / "jupyter-notebook",
    ]:
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())

    root_answer = project_root / "answer.csv"
    if root_answer.exists():
        candidates.append(root_answer)

    allowed = {".csv", ".json", ".yaml", ".yml", ".parquet", ".ipynb", ".md"}
    return sorted({path for path in candidates if path.suffix.lower() in allowed})


def build_analysis_bundle(
    project_root: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a compact, shareable archive of metrics, selected rankings and notebooks."""

    root = Path(project_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "artifacts" / "analysis_bundles" / f"avito-results-{timestamp}.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = _selected_result_files(root)
    manifest: dict[str, Any] = {
        "created_utc": timestamp,
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "file_count": len(files),
        "files": [],
        "excluded": [
            "raw dataset",
            "downloaded images",
            "model weights",
            "embedding caches",
            "non-selected dense/reranker candidate rankings",
        ],
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root)
            archive.write(path, relative.as_posix())
            manifest["files"].append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        archive.writestr(
            "analysis_bundle_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    manifest["bundle"] = str(destination)
    manifest["bundle_bytes"] = destination.stat().st_size
    return destination, manifest
