"""Model definitions. All linear models share the same preprocessing pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

AGE_COL = 7  # position of "age" in data.FEATURES


def log_age(X):
    """Replace age (days) by log(age): strength gain is roughly logarithmic in time."""
    X = np.array(X, dtype=float, copy=True)
    X[:, AGE_COL] = np.log(X[:, AGE_COL])
    return X


def linear_pipeline(degree: int = 1, alpha: float | None = None, kind: str = "ridge",
                    use_log_age: bool = True) -> Pipeline:
    """[log-age] -> standardise -> polynomial expansion -> standardise -> linear model.

    Standardising before and after the expansion keeps X^T X well conditioned.
    kind: "ols" (least squares), "ridge" (L2) or "lasso" (L1).
    """
    if kind == "ols":
        reg = LinearRegression()
    elif kind == "ridge":
        reg = Ridge(alpha=alpha)
    elif kind == "lasso":
        reg = Lasso(alpha=alpha, max_iter=100_000, tol=1e-4)
    else:
        raise ValueError(kind)
    steps = []
    if use_log_age:
        steps.append(("log_age", FunctionTransformer(log_age)))
    steps += [
        ("scale_in", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale_poly", StandardScaler()),
        ("reg", reg),
    ]
    return Pipeline(steps)


def gradient_boosting(seed: int) -> HistGradientBoostingRegressor:
    """Non-linear reference model (default hyper-parameters, not tuned)."""
    return HistGradientBoostingRegressor(random_state=seed)
