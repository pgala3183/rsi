# 10.02 · Audit an October 2026 Paper (and our own adaptation)

**Purpose:** check what was actually measured, before trusting a number.
**Run:** `python labs/g00_reading_frontier_research/lab10_02_audit_paper/run.py` (≈2 s) · **Evidence:** [`results/logs/lab10_02.txt`](../../../results/logs/lab10_02.txt)

**Paper audits (the lab's expected output):**
[`paper_audit_budget_rsi.md`](paper_audit_budget_rsi.md) · [`paper_audit_so_rsi.md`](paper_audit_so_rsi.md) · [`paper_audit_swarm_rsi.md`](paper_audit_swarm_rsi.md)

## Why this lab

The audit template asks what changed, what feedback, what evaluation protocol, what cost, and what's untested. The same questions apply to *our* adaptation. A laptop result is only worth discussing if its protocol holds, so `run.py` turns the template's "evaluation protocol" row into nine executable checks.

## Walkthrough of `run.py`

| # | Check | How it is tested |
|---|-------|------------------|
| 1 | Split sizes and disjointness | `make_task("tou")` is compared with the paper's 16 / 48 / 100; days are fingerprinted (day-of-year + price vector) |
| 2 | Held-out boundary is enforced | an improver asks for held-out data and gets `BoundaryViolation`; a full Budget RSI run is replayed through `Evaluator.access_log` |
| 3 | The ledger is honest | `ledger["fit"] == 16 × distinct programs labelled`, `ledger["steering"] == 48 × archive nodes` |
| 4 | Determinism | the same seed twice gives the same program and the same score |
| 5 | Source = executed program | `check_source_matches()` exec's the rendered Python on 32 programs × 100 random states and compares with the vectorised interpreter |
| 6 | Oracle is an upper bound | idle / seed / expert programs are simulated on all 5 tariffs; the excess over the DP oracle is measured |

## What we observed

All 9 checks pass. Check 6 found a real limitation of our "ideal score": in 139 of 1,500 policy-days a continuous policy beat the DP oracle, by at most **1.01 SoC steps of energy at that day's peak price**. The oracle is optimal on a 0.25 kWh grid, not in continuous space. So `capture()` clips at 1.0, and RSI-Index's "ideal" should be read as *oracle-level*, not as a proven optimum. The audit records the bound instead of hiding it.

## Quiz

**Q1. Why identify untested assumptions before implementing?**
So the implementation includes the check that would expose them. Here, "the oracle is a true upper bound" was an assumption; check 6 tested it and found the 0.25 kWh caveat.

**Q2. Can a laptop adaptation prove a paper's headline?**
No. It can show the mechanism works (or fails) in a different setting with a known ceiling. Lab 10.03 reproduces Budget RSI's *ordering* of methods, not its 87.0%.

Next: [Group 01 · Budget RSI](../../g01_budget_rsi/README.md)
