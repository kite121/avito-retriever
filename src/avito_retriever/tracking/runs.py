from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from avito_retriever.contracts import validate_ranking_frame


def _config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _git_revision(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class RunStore:
    def __init__(self, root: str | Path, config: dict[str, Any], project_root: str | Path):
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = str(config.get("project", {}).get("name", "run"))
        self.run_id = f"{now}-{name}-{_config_hash(config)}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.project_root = Path(project_root)

        with (self.path / "config.resolved.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        self.write_json(
            "manifest.json",
            {
                "run_id": self.run_id,
                "created_at_utc": now,
                "git_revision": _git_revision(self.project_root),
                "python": sys.version,
                "platform": platform.platform(),
            },
        )

    def write_rankings(self, rankings: pd.DataFrame) -> Path:
        validate_ranking_frame(rankings)
        destination = self.path / "rankings.parquet"
        rankings.to_parquet(destination, index=False)
        return destination

    def write_per_query(self, per_query: pd.DataFrame) -> Path:
        destination = self.path / "per_query_metrics.parquet"
        per_query.to_parquet(destination, index=False)
        return destination

    def write_json(self, name: str, value: dict[str, Any]) -> Path:
        destination = self.path / name
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        return destination

