import numpy as np
import pandas as pd

from credit_core.models import fit_monotone_lgbm
from reason_codes.shap_explain import additivity_error, challenger_shap, direction_agreement, global_importance


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    # Higher x0 means lower risk (good more likely), so the default-risk constraint on x0 is -1.
    p_good = 1 / (1 + np.exp(-(1.5 * X["x0"] - 0.8 * X["x1"])))
    y = rng.binomial(1, p_good)
    return X, pd.Series(y)


def test_shap_is_additive_to_raw_score():
    X, y = _synthetic()
    model = fit_monotone_lgbm(X, y, {"x0": -1, "x1": 1}, {"n_estimators": 100, "random_state": 42})
    values, base = challenger_shap(model, X)
    raw = model.predict(X, raw_score=True)
    assert additivity_error(values, base, raw) < 1e-6


def test_monotone_directions_agree_with_shap():
    X, y = _synthetic()
    model = fit_monotone_lgbm(X, y, {"x0": -1, "x1": 1}, {"n_estimators": 100, "random_state": 42})
    values, _ = challenger_shap(model, X)
    table = direction_agreement(values, X, {"x0": -1, "x1": 1})
    assert table["agrees"].all()


def test_global_importance_ranks_informative_features_first():
    X, y = _synthetic()
    model = fit_monotone_lgbm(X, y, {"x0": -1, "x1": 1}, {"n_estimators": 100, "random_state": 42})
    values, _ = challenger_shap(model, X)
    imp = global_importance(values, X.columns)
    assert imp.index[-1] == "noise"
