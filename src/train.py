"""Run all experiments, select the model on validation folds, evaluate once on the test set.

Usage (from the repository root):
    python -m src.train
Outputs: results/metrics.json, results/cv_grid.csv, results/figures/*.png, models/best_model.joblib
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate

from . import data
from .models import gradient_boosting, linear_pipeline

RESULTS = data.ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = data.ROOT / "models"
N_FOLDS = 5
DEGREES = [1, 2, 3, 4, 5]
ALPHAS = np.logspace(-4, 3, 15)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def cv_scores(model, X, y, cv, groups=None):
    """Mean train/validation RMSE over the folds."""
    out = cross_validate(model, X, y, cv=cv, groups=groups,
                         scoring="neg_root_mean_squared_error", return_train_score=True)
    return -out["train_score"].mean(), -out["test_score"].mean(), out["test_score"].std()


def test_metrics(model, X_dev, y_dev, X_test, y_test):
    m = clone(model).fit(X_dev, y_dev)
    p = m.predict(X_test)
    return m, p, {"RMSE": rmse(y_test, p), "MAE": float(mean_absolute_error(y_test, p)),
                  "R2": float(r2_score(y_test, p))}


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df = data.load()
    dev, test = data.split(df)
    X_dev, y_dev = dev[data.FEATURES].values, dev[data.TARGET].values
    X_test, y_test = test[data.FEATURES].values, test[data.TARGET].values
    g_dev = data.mix_groups(dev)
    gkf = GroupKFold(n_splits=N_FOLDS)
    print(f"dev={len(dev)} rows / {len(set(g_dev))} mixes, test={len(test)} rows / "
          f"{len(set(data.mix_groups(test)))} mixes")

    # 1) Ridge grid over (polynomial degree, alpha), grouped 5-fold CV on the dev set.
    rows = []
    for d in DEGREES:
        for a in ALPHAS:
            tr, va, va_sd = cv_scores(linear_pipeline(d, a), X_dev, y_dev, gkf, g_dev)
            rows.append({"degree": d, "alpha": a, "train_rmse": tr, "val_rmse": va, "val_sd": va_sd})
        print(f"  degree {d} done")
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "cv_grid.csv", index=False)
    best = grid.loc[grid.val_rmse.idxmin()]
    best_d, best_a = int(best.degree), float(best.alpha)
    print(f"Best ridge: degree={best_d}, alpha={best_a:.3g}, val RMSE={best.val_rmse:.2f}")

    # 2) Unregularised least squares for each degree (shows over-fitting).
    ols_curve = []
    for d in DEGREES:
        tr, va, _ = cv_scores(linear_pipeline(d, kind="ols"), X_dev, y_dev, gkf, g_dev)
        ols_curve.append({"degree": d, "train_rmse": tr, "val_rmse": va})
    ols_curve = pd.DataFrame(ols_curve)
    ridge_curve = grid.loc[grid.groupby("degree").val_rmse.idxmin()].reset_index(drop=True)

    # 3) Lasso at the selected degree (alpha chosen on the same grouped folds).
    lasso_rows = []
    for a in np.logspace(-3, 0.5, 12):
        _, va, _ = cv_scores(linear_pipeline(best_d, a, kind="lasso"), X_dev, y_dev, gkf, g_dev)
        lasso_rows.append((a, va))
    lasso_a, lasso_va = min(lasso_rows, key=lambda t: t[1])

    # 4) Candidate models: validation RMSE (grouped CV) and test metrics.
    candidates = {
        "Mean predictor": DummyRegressor(),
        "OLS, raw age": linear_pipeline(1, kind="ols", use_log_age=False),
        "OLS, log(age)": linear_pipeline(1, kind="ols"),
        f"OLS, poly deg {best_d}": linear_pipeline(best_d, kind="ols"),
        f"Ridge, poly deg {best_d}": linear_pipeline(best_d, best_a),
        f"Lasso, poly deg {best_d}": linear_pipeline(best_d, lasso_a, kind="lasso"),
        "Gradient boosting (ref.)": gradient_boosting(data.SEED),
    }
    table, fitted, preds = [], {}, {}
    for name, model in candidates.items():
        _, va, va_sd = cv_scores(model, X_dev, y_dev, gkf, g_dev)
        m, p, tm = test_metrics(model, X_dev, y_dev, X_test, y_test)
        fitted[name], preds[name] = m, p
        table.append({"model": name, "val_rmse": va, "val_sd": va_sd, **{f"test_{k}": v for k, v in tm.items()}})
        print(f"  {name:28s} val RMSE {va:6.2f}  test RMSE {tm['RMSE']:6.2f}  R2 {tm['R2']:.3f}")
    table = pd.DataFrame(table)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)

    ridge_name = f"Ridge, poly deg {best_d}"
    lasso_model = fitted[f"Lasso, poly deg {best_d}"].named_steps["reg"]
    n_terms = int(lasso_model.coef_.size)
    n_nonzero = int(np.sum(np.abs(lasso_model.coef_) > 1e-8))

    # 5) Leakage check: same model, random K-fold vs grouped K-fold on the dev set.
    best_model = linear_pipeline(best_d, best_a)
    _, va_random, _ = cv_scores(best_model, X_dev, y_dev, KFold(N_FOLDS, shuffle=True, random_state=data.SEED))
    _, va_group, _ = cv_scores(best_model, X_dev, y_dev, gkf, g_dev)
    gb = gradient_boosting(data.SEED)
    _, gb_random, _ = cv_scores(gb, X_dev, y_dev, KFold(N_FOLDS, shuffle=True, random_state=data.SEED))
    _, gb_group, _ = cv_scores(gb, X_dev, y_dev, gkf, g_dev)

    # 6) Save the selected model (refit on the full development set) and metrics.
    joblib.dump({"model": fitted[ridge_name], "features": data.FEATURES,
                 "degree": best_d, "alpha": best_a}, MODELS / "best_model.joblib")
    metrics = {
        "n_dev": len(dev), "n_test": len(test), "seed": data.SEED, "n_folds": N_FOLDS,
        "selected": {"model": ridge_name, "degree": best_d, "alpha": best_a,
                     "val_rmse": float(best.val_rmse)},
        "lasso": {"alpha": float(lasso_a), "val_rmse": float(lasso_va),
                  "nonzero_terms": n_nonzero, "total_terms": n_terms},
        "leakage_check_val_rmse": {"ridge_random_kfold": va_random, "ridge_group_kfold": va_group,
                                   "gb_random_kfold": gb_random, "gb_group_kfold": gb_group},
        "comparison": table.to_dict(orient="records"),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 7) Figures.
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].plot(ols_curve.degree, ols_curve.train_rmse, "o--", c="tab:blue", label="OLS train")
    ax[0].plot(ols_curve.degree, ols_curve.val_rmse, "o-", c="tab:blue", label="OLS validation")
    ax[0].plot(ridge_curve.degree, ridge_curve.train_rmse, "s--", c="tab:orange", label="Ridge train")
    ax[0].plot(ridge_curve.degree, ridge_curve.val_rmse, "s-", c="tab:orange", label="Ridge validation")
    ax[0].set_xlabel("Polynomial degree")
    ax[0].set_ylabel("RMSE (MPa), grouped 5-fold CV")
    ax[0].set_yscale("log")
    ax[0].set_xticks(DEGREES)
    ax[0].legend(fontsize=7, frameon=False, handlelength=3)
    ax[0].set_title("(a) Model complexity", fontsize=9)
    p = preds[ridge_name]
    lim = [0, max(y_test.max(), p.max()) + 5]
    ax[1].scatter(y_test, p, s=8, alpha=0.6, c="tab:orange", label=ridge_name)
    ax[1].scatter(y_test, preds["OLS, log(age)"], s=8, alpha=0.4, c="tab:gray", marker="x", label="OLS, log(age)")
    ax[1].plot(lim, lim, "k-", lw=0.8)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("Measured strength (MPa)")
    ax[1].set_ylabel("Predicted strength (MPa)")
    ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    ax[1].set_title("(b) Test set", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "results.png", dpi=200)
    fig.savefig(FIGS / "results.pdf")

    # Validation-RMSE heat map over (degree, alpha).
    piv = grid.pivot(index="degree", columns="alpha", values="val_rmse")
    fig, axh = plt.subplots(figsize=(6, 2.4))
    im = axh.imshow(np.minimum(piv.values, 15), aspect="auto", cmap="viridis_r")
    axh.set_yticks(range(len(piv.index)), piv.index)
    axh.set_xticks(range(0, len(piv.columns), 2), [f"{a:.0e}" for a in piv.columns[::2]], fontsize=7)
    axh.set_xlabel("Ridge alpha"); axh.set_ylabel("Degree")
    fig.colorbar(im, label="Validation RMSE (capped at 15)")
    fig.tight_layout()
    fig.savefig(FIGS / "cv_heatmap.png", dpi=200)
    print(f"Saved metrics to {RESULTS/'metrics.json'} and model to {MODELS/'best_model.joblib'}")


if __name__ == "__main__":
    main()
