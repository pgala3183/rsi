# 10.08 · Apply to a Debugging Task

**Purpose:** use SO-RSI's anomaly detector and probes on a debugging session, where the bug is not where the developer looks.
**Run:** `python labs/g02_second_order_diagnostics/lab10_08_apply_debugging/run.py` (<1 s) · **Evidence:** [`results/logs/lab10_08.txt`](../../../results/logs/lab10_08.txt)

## The scenario

```python
def test_spike_dump(policy, b):     # on price-spike days, export >= 6 kWh in the truly high-price hours
```

The fixture is 37 spike days, generated separately from the experiment splits. With the hand-written expert program the test **fails** (4.31 kWh). A bug has been planted in `buggy_pipeline()`: every price-derived feature is read from the *next* hour's row, the classic hour-ending vs hour-beginning timestamp bug. The policy is fine.

## Walkthrough

1. **The developer's session.** Five plausible first-order fixes to the policy, each recorded as an `Attempt` (what changed, the expected effect, what happened). They're exactly what a person would try: lower the dump threshold twice, undo, add a second spike rule keyed on import price, delete it.
2. **Detector.** `detect()` from `so_rsi.py` runs on the recorded trace, plus an explicit expected-effect check: each patch promised more spike energy and delivered none.
3. **Probes:**
   * *recurrence → sweep:* the test metric across thresholds 0.1–0.9. Every threshold is ≤ 4.31 kWh, so **the threshold is not the binding lever.**
   * *mismatch → "did it do what I expected, WHEN I expected?":* compare the hour the dump rule first fires with the first truly high-price hour, on every spike day.
   * *confirm:* cross-correlate the `export_price` feature with the raw tariff at lags −2..+2.
4. **Fix the environment, not the policy.** Rebuild the features with `_static_features()` and re-run the test with the **original** program.

## What we observed

| # | what changed | expected | what happened | anomalies flagged |
|---|---|---|---|---|
| 1 | lower spike threshold 0.50 → 0.30 | more spike energy | FAIL (4.31 kWh) | mismatch |
| 2 | lower spike threshold 0.30 → 0.20 | more spike energy | FAIL (4.31 kWh) | mismatch |
| 3 | raise it back to 0.50 (undo) | fewer false dumps | FAIL (4.31 kWh) | opposing |
| 4 | add `discharge_max if price >= 1.0` | catch spikes by import price | FAIL (4.31 kWh) | recurrence, mismatch |
| 5 | delete that rule again | back to baseline | FAIL (4.31 kWh) | opposing |

* The sweep is flat at ≤ 4.31 kWh for every threshold, so it's not the lever.
* First dump hour minus first hot hour: **−1 on all 37 spike days**. The rule fires one hour early, every time.
* Lag error is 0.000 at +1 and positive at every other lag, so `feature[t] == raw[t+1]`.
* After fixing the pipeline, the **original policy passes** (6.52 kWh). Steering capture of the same program goes from 0.496 to 0.840.

## Lessons

* Retrospective analysis surfaces patterns the developer missed in the moment: three anomaly types in five attempts.
* The second-order question ("did it happen *when* I expected?") moved the search from the policy to the environment model. That's the course's "test setup wrong, not function".
* Constructing this lab also exposed a *test* bug of my own. The first version required only 3 kWh on 3 spike days and **passed despite the planted bug**. The fixture and threshold were then set from what the correct pipeline delivers.

## Quiz

**Q1. How does SO-RSI differ from standard debugging?** Standard debugging iterates change → test. SO-RSI also watches the *history* of changes for recurrence, opposing edits and unmet expectations, and treats them as evidence that the change is being made in the wrong place.

**Q2. Can it be automated?** Detection, yes: `detect()` is a pure function of the attempt log, and the sweep, timing and lag probes here are generic. Choosing *which* probe reveals a pipeline bug still needed domain knowledge (that features have a time index), so a practical tool would detect automatically and prompt the developer with the diagnostic question.

Next: [Group 03 · Scaling RSI](../../g03_scaling_coordination/README.md)
