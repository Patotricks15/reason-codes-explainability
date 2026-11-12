import numpy as np
import pandas as pd

from credit_core.models import make_encoder
from credit_core.scorecard import Scorecard
from reason_codes.explain import check_reason_codes, hand_contributions, reason_frequency, top_reasons


def _synthetic(n: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "income": rng.lognormal(10, 0.5, n),
            "age": rng.integers(18, 70, n).astype(float),
            "purpose": rng.choice(["car", "home", "edu"], n),
        }
    )
    logit = -0.5 + 0.0 * X["age"] + 1.2 * (np.log(X["income"]) - 10) + np.where(X["purpose"] == "edu", -0.8, 0.0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), index=X.index)
    return X, y


def _fitted():
    X, y = _synthetic()
    enc = make_encoder(["income", "age"], ["purpose"], n_bins=5, min_count=20, iv_min=0.0, iv_max=10.0)
    sc = Scorecard(enc, C=1.0).fit(X, y)
    return sc, X


def test_hand_contributions_match_scorecard_contributions():
    sc, X = _fitted()
    sample = X.iloc[:50]
    contrib = sc.contributions(sample)
    for idx in sample.index:
        hand = hand_contributions(sc, sample.loc[[idx]])["delta_vs_best"]
        np.testing.assert_allclose(hand[contrib.columns].to_numpy(), contrib.loc[idx].to_numpy(), atol=1e-9)


def test_deltas_are_never_positive():
    sc, X = _fitted()
    for idx in X.index[:200]:
        assert (hand_contributions(sc, X.loc[[idx]])["delta_vs_best"] <= 1e-12).all()


def test_reason_codes_match_independent_recomputation():
    sc, X = _fitted()
    sample = X.iloc[:300]
    codes = sc.reason_codes(sample, top=3)
    assert check_reason_codes(sc, sample, codes, top=3) == 1.0


def test_applicant_at_best_bins_has_no_reasons():
    sc, X = _fitted()
    contrib = sc.contributions(X)
    best_idx = (contrib.sum(axis=1)).idxmax()
    assert top_reasons(sc, X.loc[[best_idx]]) == []


def test_reason_frequency_counts_only_reasons():
    sc, X = _fitted()
    codes = sc.reason_codes(X.iloc[:200], top=3)
    freq = reason_frequency(codes)
    assert freq.sum() == codes.notna().sum().sum()
