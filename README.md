# CR04 - Reason codes for a scorecard (German Credit)

Explains each applicant's score. For every applicant, the top-3 features whose bin pulls the
points down most, relative to the best bin of that feature, are listed as reason codes.

Notebook: [notebooks/04_reason_codes.ipynb](notebooks/04_reason_codes.ipynb)
(source: `04_reason_codes.py`, percent format). Figures are in `reports/figures/`.

## Method

- Points per feature bin: `factor · β_feature · WOE_bin`, with `factor = PDO / ln 2`.
- Gap for a feature: its points in the applicant's bin minus the best (highest) points that
  feature can give. Gaps are zero or negative.
- Reason codes: the features with the most negative gaps, top 3. Features at their best bin are
  not reasons.
- Check: `hand_contributions` recomputes the gaps from the fitted WOE tables and coefficients,
  independently of `Scorecard.contributions`. `check_reason_codes` compares the two.

## Data and model

- German Credit (`kabure/german-credit-data-with-risk`), target `Risk` (1 = good).
- Same split and model as CR01: 70% train (n = 700), 30% holdout (n = 300), seed 42, `C = 1.0`.
- Explained applicants: the 300 holdout rows.

## Results

- The reason list matches the independent recomputation for 100% of holdout applicants.
- Most frequent reason codes on the holdout: Saving accounts (280), Duration (245),
  Purpose (171), Age (102), Credit amount (68), Housing (18).
- Hand check (notebook section 3): for applicant 378, the gaps are Saving accounts "little"
  −45.3, Duration 36 months −38.5, Credit amount 14,318 −16.9, Purpose "car" −15.0, Housing
  "free" −11.7. The top 3 are Saving accounts, Duration, and Credit amount, which matches the
  output.

## Limitations

- The check confirms that the code is consistent with the scorecard arithmetic. It does not show
  that the reason is the true driver of the decision.
- Reason codes come from binned features, so a reason names a bin, not the exact value.
  "Duration" for a 36-month loan means the 36-month bin.
- Saving accounts "little" appears as a reason for most applicants. That is a feature of this
  small dataset (1,000 rows). It does not generalise to other portfolios.
- SHAP for the LightGBM challenger is not included. The plan requires approval for the `shap`
  dependency.

## Reproduce

```bash
pip install -e ../credit-scoring-core -e ../scorecard-woe-logistic -e .
pytest
make notebook
```
