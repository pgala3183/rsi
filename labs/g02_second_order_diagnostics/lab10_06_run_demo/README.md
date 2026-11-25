# 10.06 · Run the SO-RSI Demo

**Purpose:** watch anomaly detection change what the improver does, and measure whether it helps.
**Run:** `python labs/g02_second_order_diagnostics/lab10_06_run_demo/run.py` (≈6 s) · **Evidence:** [`results/logs/lab10_06.txt`](../../../results/logs/lab10_06.txt) · inquiry memory: [`results/so_rsi_inquiry_memory_tou.json`](../../../results/so_rsi_inquiry_memory_tou.json)

## Walkthrough of the code

* **Symptoms** (`diagnose.symptoms`) compare the policy's battery power hour by hour with the oracle's. Each mismatch is binned as `uncovered_peak`, `early_discharge`, `missed_grid_charge`, `excess_grid_charge` or `wasted_solar` and weighted by the hour's price. For the factory program on TOU: uncovered_peak 74, wasted_solar 43, early_discharge 33.
* **`FirstOrderImprover.run()`**, Naive RSI: pick a symptom (usually the largest), look up `PATCHES[symptom]` → `(slot, direction)`, create the slot's rule from its template or nudge its threshold, simulate, and keep the patch only if the steering score rises.
* **`SecondOrderImprover.after_attempt()`**: after every attempt, `detect(trace)` checks the **last attempt against earlier ones only** (the course demo compared each trace with itself; see 10.01):
  * *recurrence*: the same `(slot, direction)` has failed ≥ 2 times in the last 8 attempts → `sweep_probe` → `lever_probe`
  * *opposing*: the same slot was nudged in the opposite direction within the last 4 attempts → `split_probe`
  * *mismatch*: the targeted symptom moved, but the score fell → `attribution_probe`
* **Probes cost day-sims from the same budget** (`Evaluator(charge_repeats=True)` also bills a memoryless improver for re-running a patch it already tried).

## The annotated trace (TOU, seed 0)

```
[  2] wasted_solar       -> create hold      dir=+1  ACCEPT dScore=+0.0813
...
[  9] wasted_solar       -> create absorb    dir=-1  reject dScore=-0.0017
      !! anomaly=recurrence on slot=absorb
      ?? sweep probe: absorb.surplus_fc is NOT a lever (no threshold beats the incumbent; spread=0.020)
      ?? lever probe: no lever found (shadowed 0%, empty 1%): ban this patch
[ 12] missed_grid_charge -> create charge    dir=+1  reject dScore=-0.0396
      !! anomaly=mismatch on slot=charge
      ?? attribution probe: cost rose most at hour 1 (+$31.52 over 48 days):
         grid charging filled the battery before free solar arrived
      -> guard: keep the patch, add guard surplus_left_fc <= 4.0  dScore=+0.0686 ACCEPT
```

Attempt 12 is the core of SO-RSI. The first-order patch ("charge from the grid at night") was *rejected* because it made things worse. Instead of discarding it, the attribution probe asked where the money went: at 01:00 on sunny days, the battery was filled from the grid and then had no room for free solar. That licenses a **structural** edit, the same patch plus a guard "only when the forecast solar surplus is small". The guard is now in inquiry memory, so later charge rules inherit it. The final program:

```python
def policy(o):
    if o.export_price >= 0.60:
        return DISCHARGE_MAX
    if o.price_rank <= 0.15 and o.soc <= 0.90 and o.surplus_left_fc <= 4.00:
        return CHARGE_MAX
    if o.peak_ahead >= 0.40 and o.price_rank <= 0.60 and o.surplus_fc <= 0.00:
        return IDLE
    return SELF_CONSUME
```

## Matched-budget comparison (8 seeds, 4,800 day-sims per run, held-out capture)

| tariff | root | Naive RSI | SO-RSI | SO-RSI − naive | recurrences n / so | longest zero-progress n / so |
|---|---|---|---|---|---|---|
| tou | 0.712 | 0.804 | **0.926** | **+0.122** | 60 / 6 | 89 / 9 |
| vpp_events | 0.841 | 0.959 | 0.959 | +0.000 | 66 / 5 | 92 / 14 |
| critical_peak | 0.790 | 0.839 | **0.922** | **+0.083** | 68 / 4 | 69 / 7 |
| wholesale | 0.501 | 0.776 | 0.763 | −0.013 | 60 / 4 | 61 / 9 |
| solar_glut | 0.779 | 0.863 | 0.822 | −0.041 | 64 / 4 | 82 / 15 |

* **Thrashing is suppressed everywhere**: about 10× fewer recurrences and much shorter zero-progress streaks. This is the paper's behavioural claim.
* **The score gain is conditional.** It's large on TOU and critical-peak, where the right fix is structural (the peak rule is a no-op until a hold rule exists; night charging needs a solar guard). On wholesale and solar_glut, ordinary patches already make progress (naive accepts 14 patches per run on wholesale), so the probes' budget is better spent patching.
* Paper: +21.8 and +25.8 points (Lean 4, Verus). Ours: +0.030 on average, +0.12 at best. Different scale, same sign where its premise holds.

## Quiz

**Q1. The key difference?** Naive RSI changes the policy in response to symptoms. SO-RSI also changes *itself*: its memory bans levers, learns guards and re-maps symptoms, and those edits govern every later attempt.

**Q2. Why not synthetic traces?** The course demo's synthetic traces hard-code the result (lab 10.01). Here every outcome is a simulation, so SO-RSI can lose, and on two tariffs it does.

Next: [10.07 · Analyse diagnostic probes](../lab10_07_analyze_probes/README.md)
