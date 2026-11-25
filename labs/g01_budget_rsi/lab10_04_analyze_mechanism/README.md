# 10.04 · Analyse the Scoring Mechanism

**Purpose:** understand *why* the linear readout picks what it picks.
**Run:** `python labs/g01_budget_rsi/lab10_04_analyze_mechanism/run.py` (≈5 s) · **Evidence:** [`results/logs/lab10_04.txt`](../../../results/logs/lab10_04.txt)

## Walkthrough

1. **Coefficients across 12 calibrations.** `LinearReadout` standardises its inputs, so the coefficients are comparable. Instead of one run (the course's code snippet), we show the mean, the spread and the **sign agreement** over 12 independent calibrations.
2. **Ranking fresh edits.** 200 *new* edits (not used for fitting), from programs in calibration 0's archive. We report the Spearman correlation between predicted and actual capture, and the quality of the readout's top 20.
3. **Ablation.** `ColumnReadout` fits the same ridge model on a subset of the inputs (parent rate only / diff only / both), and the full search is re-run with it for 12 runs.
4. **Training-set size.** Recalibrate with 24, 48, 96 and 192 labelled edits.
5. **8 → 4 coordinates.** Keep lines added/deleted, rules added and constants changed.

## What we observed

| feature | mean coef | sd | sign agreement |
|---|---|---|---|
| constants_changed | +0.252 | 0.053 | 1.00 |
| conds_delta | −0.153 | 0.050 | 1.00 |
| reordered | +0.044 | 0.021 | 0.92 |
| rules_added | +0.031 | 0.024 | 0.92 |
| ... | | | |
| parent_rate | +0.008 | 0.033 | **0.58** |

* **Interpretation.** The readout learned "new rules with thresholds help (`constants_changed` +), extra conditions hurt (`conds_delta` −)". From a no-rule start, adding a templated rule is usually good, and piling conditions onto it makes it fire too rarely. Parent rate is near zero and unstable in sign. That contradicts the course text's expectation ("parent rate often has significant weight"): in our archives the parents' rates vary little, so they cannot explain the variation in children.
* **Correlated inputs:** corr(lines_added, rules_added) = 0.61. Credit is shared, so treat single coefficients as "edit size and kind", not as causal effects (the course's Q2).
* **Fresh edits:** Spearman 0.442. The readout's top 20 average 0.755 actual capture vs 0.345 for all fresh edits. It is a useful filter, not a precise predictor.

| readout inputs | held-out capture (12 runs) |
|---|---|
| parent rate only | 0.756 ± 0.069 |
| diff coordinates only | 0.817 ± 0.074 |
| both (paper) | 0.817 ± 0.074 |

| labelled edits | fresh-edit Spearman | held-out | calibration day-sims |
|---|---|---|---|
| 24 | 0.111 | 0.823 | 330 |
| 48 | 0.358 | 0.806 | 648 |
| 96 | 0.286 | 0.806 | 1,250 |
| 192 | 0.376 | 0.806 | 2,424 |

* More labels improve the readout's *ranking* of fresh edits, but **not** the search outcome. With 4 nodes, the readout only needs to avoid obviously bad edits.
* 4 coordinates: 0.815 vs 0.817, no loss.

*Reproducibility note:* two identical edits get exactly equal predictions. An early version broke those ties by floating-point noise, which made one run differ between labs. Ties are now broken by proposal order (`np.round(..., 9)` in `search()`), and labs 10.03 and 10.04 agree run for run.

## Quiz

**Q1. Why linear and not a neural network?** It fits 96 rows in microseconds and scores hundreds of proposals per step, and its coefficients are inspectable (this lab). With 96 labels a bigger model would mostly fit noise: fresh-edit Spearman is only 0.29–0.44 even for the linear model.

**Q2. Two highly correlated features?** Their coefficients trade off (lines_added vs rules_added here) while the combined prediction stays stable. Ridge regularisation (`ridge=1.0`) keeps the split from blowing up.

Next: [10.05 · Extend to your own edits](../lab10_05_extend/README.md)
