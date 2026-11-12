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
# # CR04b - Tree SHAP for the LightGBM challenger (LendingClub)
#
# Goal: explain the CR02 monotone LightGBM challenger with tree SHAP. SHAP values are computed on
# the raw score (log-odds of good). Negative values push an applicant toward default, so the most
# negative values are the reasons for a risky score.
#
# Checks: (1) SHAP is additive to the raw score, (2) the sign of each constrained feature's effect
# agrees with its monotone constraint.
#
# Data: LendingClub, test window 2015 to 2016 (CR02 split). A random sample of 50,000 test loans is
# explained, with seed 42.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_core.datasets import LC_MONOTONE
from credit_core.models import lgbm_frame
from gbm_monotone.pipeline import fit_challenger, load_oot
from reason_codes.shap_explain import (
    additivity_error,
    challenger_shap,
    direction_agreement,
    global_importance,
)

FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
SAMPLE_N = 50_000
SEED = 42

# %% [markdown]
# ## 1. Challenger and sample

# %%
d = load_oot()
model = fit_challenger(d)
X_all = lgbm_frame(d.X_test, d.categories)
sample = X_all.sample(n=SAMPLE_N, random_state=SEED)
y_sample = d.y_test.loc[sample.index]
print(f"Explained loans: {len(sample):,}, good share: {y_sample.mean():.3f}")

# %% [markdown]
# ## 2. SHAP values and additivity check

# %%
values, base = challenger_shap(model, sample)
raw = model.predict(sample, raw_score=True)
err = additivity_error(values, base, raw)
print(f"Base value (log-odds of good): {base:.3f}")
print(f"Max |base + sum(SHAP) - raw score|: {err:.2e}")

# %% [markdown]
# ## 3. Global importance

# %%
imp = global_importance(values, sample.columns)
print(imp.round(3).to_string())

# %%
fig, ax = plt.subplots(figsize=(7, 5))
imp.sort_values().plot.barh(ax=ax)
ax.set(xlabel="Mean |SHAP| (log-odds of good)", title="LightGBM challenger: global importance")
fig.savefig(FIG / "cr04b_shap_importance.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Monotone constraints vs SHAP direction

# %%
agree = direction_agreement(values, sample, LC_MONOTONE)
print(agree.round(3).to_string(index=False))
print(f"\nConstrained features whose SHAP direction agrees: {agree['agrees'].sum()} of {len(agree)}")

# %% [markdown]
# ## 5. Reasons for one risky applicant
#
# Take a defaulted loan with a low predicted good score. Its top-3 reasons are the three features
# with the most negative SHAP values.

# %%
p_good = model.predict_proba(sample)[:, 1]
cand = pd.DataFrame({"p_good": p_good, "y": y_sample.to_numpy()}, index=sample.index)
pick = cand[cand["y"] == 0].nsmallest(1, "p_good").index[0]
row = sample.index.get_loc(pick)
contrib = pd.Series(values[row], index=sample.columns)
top3 = contrib.nsmallest(3)
print(f"Loan index {pick}: predicted good {p_good[row]:.3f}, observed default")
print("Top-3 reasons (most negative SHAP):")
print(top3.round(3).to_string())

# %%
fig, ax = plt.subplots(figsize=(7, 5))
show = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10]).sort_values()
colors = ["firebrick" if v < 0 else "seagreen" for v in show]
ax.barh(show.index, show.to_numpy(), color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(xlabel="SHAP value (log-odds of good)", title=f"Applicant {pick}: top 10 contributions")
fig.savefig(FIG / "cr04b_applicant_shap.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Summary

# %%
print(f"Additivity error: {err:.2e}")
print(f"Constrained features agreeing with SHAP: {agree['agrees'].sum()} of {len(agree)}")
print(imp.head(3).round(3).to_string())
