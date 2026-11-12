"""Tree SHAP for the LightGBM challenger, on the raw score (log-odds of good)."""

import numpy as np
import pandas as pd
import shap


def challenger_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    expected = explainer.expected_value
    if isinstance(raw, list):  # one array per class; index 1 is the good class (y = 1)
        raw, expected = raw[1], np.ravel(expected)[1]
    values = np.asarray(raw)
    if values.shape != X.shape:
        raise ValueError(f"expected SHAP shape {X.shape}, got {values.shape}")
    return values, float(np.ravel(expected)[-1])


def additivity_error(values: np.ndarray, base: float, raw_score: np.ndarray) -> float:
    """Largest gap between base + sum(SHAP) and the model's raw score."""
    return float(np.max(np.abs(values.sum(axis=1) + base - np.asarray(raw_score))))


def global_importance(values: np.ndarray, columns) -> pd.Series:
    return pd.Series(np.abs(values).mean(axis=0), index=list(columns)).sort_values(ascending=False)


def direction_agreement(values: np.ndarray, X: pd.DataFrame, monotone: dict) -> pd.DataFrame:
    """For each constrained feature, the Spearman correlation between its value and its SHAP value.

    `monotone` is in default-risk direction. The model predicts good, so the expected sign of the
    correlation is the opposite: a risk-increasing feature lowers the good score.
    """
    rows = []
    for j, col in enumerate(X.columns):
        if monotone.get(col, 0) == 0:
            continue
        rho = pd.Series(X[col].to_numpy(dtype=float)).corr(pd.Series(values[:, j]), method="spearman")
        expected = -np.sign(monotone[col])
        rows.append(
            {
                "feature": col,
                "expected_sign": int(expected),
                "spearman_rho": float(rho),
                "agrees": bool(np.sign(rho) == expected),
            }
        )
    return pd.DataFrame(rows)
