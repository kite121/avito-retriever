from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compare_paired_runs(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str = "ap@10",
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two systems on aligned query-level scores.

    Returns a paired mean difference, bootstrap confidence interval, paired t-test,
    Wilcoxon signed-rank test, and a paired random-sign permutation test.
    """

    left = baseline[["query_id", metric]].rename(columns={metric: "baseline"})
    right = candidate[["query_id", metric]].rename(columns={metric: "candidate"})
    aligned = left.merge(right, on="query_id", how="inner", validate="one_to_one")
    if aligned.empty:
        raise ValueError("No shared query_id values between runs")

    differences = (aligned["candidate"] - aligned["baseline"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    n = len(differences)

    bootstrap_indices = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])

    signs = rng.choice((-1.0, 1.0), size=(permutation_samples, n))
    permuted = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    permutation_p = (np.count_nonzero(np.abs(permuted) >= observed) + 1) / (
        permutation_samples + 1
    )

    t_test = stats.ttest_rel(aligned["candidate"], aligned["baseline"])
    try:
        wilcoxon_p = float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    return {
        "metric": metric,
        "n_queries": n,
        "baseline_mean": float(aligned["baseline"].mean()),
        "candidate_mean": float(aligned["candidate"].mean()),
        "mean_difference": float(differences.mean()),
        "bootstrap_ci": [float(ci_low), float(ci_high)],
        "paired_t_p": float(t_test.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "paired_permutation_p": float(permutation_p),
        "wins": int(np.count_nonzero(differences > 0)),
        "ties": int(np.count_nonzero(differences == 0)),
        "losses": int(np.count_nonzero(differences < 0)),
    }

