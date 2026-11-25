# 10.03 · Run the Budget RSI Demo

**Purpose:** see the mechanism run end to end, on real program edits scored by a simulator.
**Run:** `python labs/g01_budget_rsi/lab10_03_run_demo/run.py` (≈2 s) · **Evidence:** [`results/logs/lab10_03.txt`](../../../results/logs/lab10_03.txt)

## Walkthrough

**Step 1 — the task** (`gridwise/env.py`). `make_task("tou", 0)` generates 164 days and splits them 16 fit / 48 steering / 100 held-out, the same sizes as the paper's HumanEval+ split. Each day has prices, solar, load and forecasts. `oracle()` solves the day exactly by dynamic programming over the state of charge, and `capture()` scores a program by the share of the oracle's savings it achieved. The starting program is `return SELF_CONSUME` (steering 0.787, 30/48 days solved).

**Step 2 — calibration** (`calibrate()` in `budget_rsi.py`). Ninety-six times: pick a parent from a growing pool, ask the proposer (`edits.propose`) for an edit, and simulate parent and child on the 16 fit days. The row is `X = [parent's capture, 8 diff coordinates]` and `y = child's capture`. `LinearReadout.fit()` is ridge regression on standardised inputs, and it is **frozen** afterwards. Measured: R² 0.32, 1,152 fit day-sims.

**Step 3 — one scored proposal.** The script prints a real unified diff, for example:
```diff
 def policy(o):
+    if o.peak_ahead >= 0.50 and o.price_rank <= 0.50:
+        return IDLE
     return SELF_CONSUME
```
`diff_features()` reads only that text: lines_added=2, rules_added=1, conds_delta=2, constants_changed=2, returns_changed=1. The readout predicts 0.879. The simulated truth is 0.537, a reminder that this is a ranking heuristic, not an oracle.

**Step 4 — one search** (`search(..., "readout")`). Each of the archive's nodes proposes 4 edits. The readout scores all of them and only the top one is simulated on the 48 steering days, repeated until 4 nodes exist. The return rule is the paper's: the earliest node with maximal steering solve count. Here node 3 (36/48 solved) has held-out capture 0.832.

**Step 5 — twelve runs, all selectors** (same calibration and search seeds for every selector, so comparisons are paired).

**Step 6 — measured cost**, read from `Evaluator.ledger`.

## What we observed

| method | held-out capture | W/T/L vs HGM | paper (GPT-4o, HumanEval+) |
|---|---|---|---|
| readout (Budget RSI) | **0.817 ± 0.074** | 5/3/4 | 87.0 |
| HGM selection | 0.787 ± 0.083 | – | 85.1 |
| parent-rate ablation | 0.770 ± 0.076 | 0/8/4 | 84.2 |
| random | 0.733 ± 0.056 | 3/3/6 | – |
| root | 0.712 | 1/5/6 | 81.2 |
| calibration-best | 0.887 ± 0.030 | 12/0/0 | 82.8 |

| phase | day-sims per run |
|---|---|
| calibration (96 edits × 16 fit days) | 1,263 |
| search with readout (4 nodes × 48 steering days) | 192 |
| search evaluating every proposal (upper bound) | ≤ 2,304 |

* **Reproduced:** the ordering readout > HGM > parent-rate > root.
* **Not reproduced:** calibration-best beats everything here. Our evaluations are deterministic and the fit split is informative, so the calibration phase (a 96-candidate search costing 6.6× the search) is itself a strong optimiser. The paper's calibration-best (82.8%) is weak, consistent with its noisier 4-sample LLM labels.

## Check your understanding

1. **Why a steering split instead of all test data?** Selection must not see the data you report on. The `Evaluator` enforces it: the search cannot read held-out (lab 10.02).
2. **What if you change 8 features to 4?** Lab 10.04: 0.815 vs 0.817, essentially no change on this task.
3. **Fewer training edits?** Lab 10.04: held-out capture is flat from 24 to 192 edits; the readout only needs to separate "plausible small edit" from "large risky edit".

## Quiz

**Q1. Purpose of the steering split?** It is a proxy for held-out performance that the improver *may* use, so the held-out split stays a clean final measurement. Lab 10.17 measures how faithful the proxy is: Spearman 0.94 with held-out, but the selected node is optimistic by +0.037.

**Q2. Why not synthetic edits?** The course demo's synthetic edits made the result true by construction (lab 10.01). Here every edit is a real program change and every label a simulation.

Next: [10.04 · Analyse the scoring mechanism](../lab10_04_analyze_mechanism/README.md)
