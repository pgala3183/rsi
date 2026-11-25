# 10.15 · Calculate RSI-Index

**Purpose:** compute RSIGym's RSI-Index for our three methods on five benchmarks.
**Run:** `python labs/g05_environments_metrics/lab10_15_rsi_index/run.py` (≈30 s) · **Evidence:** [`results/logs/lab10_15.txt`](../../../results/logs/lab10_15.txt)

## The formula, and a sign error in the course page

```
RSI-Index = mean over benchmarks of  (final - initial) / (ideal - initial)
```

The course page prints the numerator as `(initial - final)`, but its own worked example (50 → 75 and 20 → 60 out of 100, giving 0.5) uses `(final - initial)`. `run.py` computes both: the printed formula gives −0.50 on the worked example.

## Walkthrough

* **Benchmarks:** the five tariffs. **Initial** = the factory `SELF_CONSUME` program. **Ideal** = the oracle (capture 1.0, accurate to one SoC step; lab 10.02). **Final** = held-out capture of the program each method returns.
* **Matched budget:** every method gets 4,800 day-simulations per benchmark. `budget_rsi()` spends what's left after calibration on a longer archive. `so_rsi()` runs `SecondOrderImprover` with `charge_repeats=True`. `swarm_rsi()` runs the full swarm.
* 4 seeds per method per benchmark.

## What we observed

| benchmark | initial | Budget RSI | SO-RSI | SwarmRSI |
|---|---|---|---|---|
| tou | 0.712 | 0.874 | 0.926 | **0.937** |
| vpp_events | 0.841 | 0.900 | 0.959 | **0.962** |
| critical_peak | 0.790 | 0.903 | 0.922 | **0.945** |
| wholesale | 0.501 | 0.674 | **0.766** | 0.756 |
| solar_glut | 0.779 | 0.957 | 0.866 | **0.972** |

| gap closed | Budget RSI | SO-RSI | SwarmRSI |
|---|---|---|---|
| tou | 0.564 | 0.744 | 0.782 |
| vpp_events | 0.367 | 0.740 | 0.759 |
| critical_peak | 0.538 | 0.627 | 0.736 |
| wholesale | 0.346 | 0.531 | 0.511 |
| solar_glut | 0.806 | 0.393 | 0.872 |
| **RSI-Index** | **0.524** | **0.607** | **0.732** |
| course formula as printed | −0.524 | −0.607 | −0.732 |

Day-sims actually used: Budget RSI 4,833, SO-RSI **2,719**, SwarmRSI 4,843. SO-RSI stops early once its memory has banned every patch for the current symptoms. It used 57% of its budget, which matters when reading its index.

## Interpretation

* **SwarmRSI has the highest index** with a full budget. It is the most *general* improver here: its proposer can write any rule, whereas SO-RSI is limited to its hand-designed slots and patch table, which is why it scores 0.393 on solar_glut.
* **Budget RSI is lowest:** its readout spends 26% of the budget on calibration and then searches by score. Lab 10.05 shows that on GridWise plain search with that budget would do better.
* Paper: Opus 5 reaches RSI-Index **0.4809** on five real benchmarks with $500 each. Our numbers are a different system on a different task; only the metric is shared.

## Quiz

**Q1. Why use RSI-Index instead of raw scores?** It normalises by the *remaining* gap. Closing 0.12 on vpp_events (gap 0.16) counts as 0.74, closing 0.27 on wholesale (gap 0.50) as 0.53. The flip side: near-solved benchmarks can dominate the mean, so report the per-benchmark rows alongside it.

Next: [10.16 · Compare environments](../lab10_16_environments/README.md)
