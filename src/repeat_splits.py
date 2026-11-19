"""Robustness check: repeat the whole procedure over several grouped dev/test splits.

For every seed: grouped 80/20 split -> select (degree, alpha) by grouped 5-fold CV on the
dev part only -> refit -> evaluate on that seed's test part. Reports mean +- std test RMSE.

Usage: python -m src.repeat_splits [n_seeds]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import data
from .models import gradient_boosting, linear_pipeline
from .train import ALPHAS, DEGREES, N_FOLDS, RESULTS, cv_scores, test_metrics


def run(n_seeds: int = 10):
    df = data.load()
    rows = []
    for seed in range(n_seeds):
        dev, test = data.split(df, seed=seed)
        Xd, yd = dev[data.FEATURES].values, dev[data.TARGET].values
        Xt, yt = test[data.FEATURES].values, test[data.TARGET].values
        gd = data.mix_groups(dev)
        cv = GroupKFold(n_splits=N_FOLDS)
        best = min(((cv_scores(linear_pipeline(d, a), Xd, yd, cv, gd)[1], d, a)
                    for d in DEGREES for a in ALPHAS))
        models = {
            "OLS, log(age)": linear_pipeline(1, kind="ols"),
            "Ridge, CV-selected": linear_pipeline(best[1], best[2]),
            "Gradient boosting (ref.)": gradient_boosting(seed),
        }
        for name, m in models.items():
            _, _, tm = test_metrics(m, Xd, yd, Xt, yt)
            rows.append({"seed": seed, "model": name, "degree": best[1] if "Ridge" in name else None,
                         **tm})
        print(f"seed {seed}: selected degree={best[1]}, alpha={best[2]:.3g}")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "repeated_splits.csv", index=False)
    summary = res.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    print(summary.round(3))
    deg_counts = res[res.model == "Ridge, CV-selected"].degree.value_counts().to_dict()
    out = {"n_seeds": n_seeds,
           "summary": {m: {k: float(v) for k, v in r.items()}
                       for m, r in summary.round(4).iterrows()},
           "selected_degree_counts": {str(int(k)): int(v) for k, v in deg_counts.items()}}
    # flatten tuple keys
    out["summary"] = {m: {f"{a}_{b}": v for (a, b), v in zip(summary.columns, summary.loc[m].round(4))}
                      for m in summary.index}
    (RESULTS / "repeated_splits.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
