"""Reason codes for a points scorecard, plus a hand-style recomputation used to check them."""

import pandas as pd
from matplotlib.axes import Axes

from credit_core.scorecard import Scorecard


def hand_contributions(sc: Scorecard, row: pd.DataFrame) -> pd.DataFrame:
    """Recompute points per feature for one applicant directly from WOE tables and coefficients.

    Independent of `Scorecard.contributions`: it reads the fitted WOE tables and logistic
    coefficients, then takes each feature's points relative to its best bin.
    `row` is a one-row DataFrame with the raw feature values.
    """
    if len(row) != 1:
        raise ValueError("hand_contributions expects exactly one row")
    coef = dict(zip(sc.encoder.selected_, sc.model_.coef_[0]))
    bins = sc.encoder.transform_bins(row).iloc[0]
    out = []
    for feat in sc.encoder.selected_:
        woe_by_bin = sc.encoder.tables_[feat]["woe"]
        points_by_bin = sc.factor_ * coef[feat] * woe_by_bin
        b = bins[feat]
        point_here = points_by_bin.get(b, 0.0)  # unseen bins carry WOE 0
        out.append(
            {
                "feature": feat,
                "bin": b,
                "points": point_here,
                "delta_vs_best": point_here - points_by_bin.max(),
            }
        )
    return pd.DataFrame(out).set_index("feature")


def top_reasons(sc: Scorecard, row: pd.DataFrame, top: int = 3) -> list[str]:
    """Features with the largest negative points gap, most negative first."""
    contrib = hand_contributions(sc, row)["delta_vs_best"]
    negative = contrib[contrib < 0].sort_values()
    return list(negative.index[:top])


def reason_frequency(codes: pd.DataFrame) -> pd.Series:
    """How often each feature appears as any reason across the sample."""
    flat = codes.to_numpy().ravel()
    flat = flat[pd.notna(flat)]
    return pd.Series(flat).value_counts()


def plot_applicant(sc: Scorecard, row: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Horizontal bars: each feature's points gap vs. its best bin for one applicant."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    h = hand_contributions(sc, row).sort_values("delta_vs_best")
    colors = ["tab:red" if v < 0 else "tab:gray" for v in h["delta_vs_best"]]
    ax.barh(h.index, h["delta_vs_best"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("points vs best bin")
    ax.set_title("Points gap by feature for one applicant")
    return ax


def check_reason_codes(sc: Scorecard, X: pd.DataFrame, codes: pd.DataFrame, top: int = 3) -> float:
    """Share of applicants whose reason list matches the independent recomputation."""
    matches = 0
    for idx in X.index:
        expected = top_reasons(sc, X.loc[[idx]], top=top)
        got = [c for c in codes.loc[idx].tolist() if pd.notna(c)]
        matches += int(expected == got)
    return matches / len(X)
