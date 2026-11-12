# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CR04 - Reason codes for a scorecard (German Credit)
#
# Goal: explain each applicant's score. For each holdout applicant, the top-3 features that pull
# the points down relative to the best bin of each feature. The reason list is checked against an
# independent recomputation from the WOE tables and coefficients, and a sample is checked by hand.
#
# Model: the CR01 scorecard (same split, same `C`). Target `Risk` (1 = good). Holdout n = 300.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reason_codes.explain import (
    check_reason_codes,
    hand_contributions,
    plot_applicant,
    reason_frequency,
    top_reasons,
)
from scorecard_woe.pipeline import FEATURES, fit_scorecard, load_split

FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Fit the scorecard and generate reason codes

# %%
split = load_split()
sc, C = fit_scorecard(split)
print(f"C = {C}, train n = {len(split.X_train)}, holdout n = {len(split.X_test)}")

X_hold = split.X_test[FEATURES]
codes = sc.reason_codes(X_hold, top=3)
codes.head(10)

# %% [markdown]
# ## 2. Automated check against an independent recomputation

# %%
match_rate = check_reason_codes(sc, X_hold, codes, top=3)
print(f"Share of holdout applicants whose reason list matches the recomputation: {match_rate:.1%}")

# %%
freq = reason_frequency(codes)
freq

# %% [markdown]
# ## 3. Hand check on applicants
#
# For three applicants, the raw bin, points gap, and reason codes are shown side by side, so each
# code can be checked against the raw value. Applicants are taken at the 5th, 50th and 95th
# percentile of the holdout score, which covers low, middle and high scores.

# %%
score = pd.Series(sc.score(X_hold), index=X_hold.index)
picks = [score.sort_values().index[int(len(score) * q)] for q in (0.05, 0.50, 0.95)]

for idx in picks:
    row = X_hold.loc[[idx]]
    print(f"\n=== applicant index {idx}: score {score[idx]:.0f}, reasons {top_reasons(sc, row)}")
    print(row.T.to_string())
    print(hand_contributions(sc, row).round(2).to_string())

# %% [markdown]
# Reading the first applicant: each reason is a feature whose bin is worse than its best bin.
# A code is correct when that bin's points gap is among the largest negative gaps.

# %% [markdown]
# ## 4. Plots

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
freq.sort_values().plot(kind="barh", ax=ax, color="tab:red")
ax.set_xlabel("times listed as a top-3 reason (holdout)")
ax.set_title("How often each feature is a reason code")
fig.tight_layout()
fig.savefig(FIG / "cr04_reason_frequency.png", dpi=120)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, idx, label in zip(axes, picks, ["low score", "median score", "high score"]):
    plot_applicant(sc, X_hold.loc[[idx]], ax=ax)
    ax.set_title(f"{label}: score {score[idx]:.0f}")
fig.tight_layout()
fig.savefig(FIG / "cr04_applicant_points.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Summary
#
# - Reason codes are generated for every holdout applicant. The automated check matches the
#   independent recomputation for all of them (see section 2).
# - The hand-checked cases are in section 3.
# - SHAP for the LightGBM challenger (CR02) is not included yet. It needs the `shap` dependency,
#   which the plan requires approval for.

# %%
codes.head(3)
