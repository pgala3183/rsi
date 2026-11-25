# 10.10 · Simulate Leader-Follower Coordination (AReaL-RSI)

**Purpose:** see conditional memory sharing at work, and test the paper's "sharing beats independent exploration".
**Run:** `python labs/g03_scaling_coordination/lab10_10_leader_follower/run.py` (≈7 s) · **Evidence:** [`results/logs/lab10_10.txt`](../../../results/logs/lab10_10.txt)

## Walkthrough of `areal_rsi.py`

* **`ORIENT`** defines four research orientations (`peak`, `solar`, `arbitrage`, `events`). Each is a set of preferred rule templates plus the oracle symptoms it targets.
* **Leader** (top of each campaign in `run_team`): it traces the current best program, computes its symptoms and ranks orientations by `priority[o] × unresolved $-symptom`. The top two go to the two researchers.
* **Researchers** (`investigate`): a short hill-climb whose proposer prefers the orientation's templates (`soft_templates`, 3× weight) without excluding the others.
* **`ConditionalMemory`** stores `{campaign, orientation, problem, status: supported | unresolved, evidence}`. `supported()` and `unresolved()` are the two views. A problem leaves the unresolved list once a later finding supports it.
* **Evidence-based review**: a validated gain raises that orientation's priority ×1.2; a failure halves it.
* **Ablations**: `conditional=False` passes each researcher's result to the next *whether or not it improved* (sharing without the resolved/unresolved filter). `run_independent` has two researchers who never share; the better one is reported.

## What we observed

The TOU campaign log and memory (excerpt):

| campaign | orientation | problem | status | evidence |
|---|---|---|---|---|
| 1 | events | uncovered_peak | supported | +0.1356 |
| 2 | solar | wasted_solar | supported | +0.0153 |
| 2 | peak | uncovered_peak | unresolved | +0.0000 |
| 4 | arbitrage | excess_grid_charge | supported | +0.0016 |

The best program reached **0.954** held-out (99/100 days solved). It learned a cloudy-day grid-charge rule, a guarded peak discharge and solar absorption.

| tariff | conditional memory | unconditional sharing | independent |
|---|---|---|---|
| tou | **0.929** (+1.8%) | 0.912 (−0.1%) | 0.913 |
| critical_peak | 0.942 (+0.5%) | **0.945** (+0.8%) | 0.937 |
| wholesale | 0.736 (+1.0%) | **0.771** (+5.7%) | 0.729 |

* **Sharing beats independent exploration on 3/3 tariffs** (paper: 1.9–27.5%; here the best sharing variant gains +0.8% to +5.7%).
* **The conditional filter wins on 1/3.** On wholesale, passing on unvalidated results acts like a random walk that escapes a plateau, and that helps. The filter's value depends on how deceptive small regressions are.

## Quiz

**Q1. Why distinguish resolved from unresolved findings?** So that researchers build only on validated improvements, while the leader keeps aiming effort at problems still open. The TOU run shows this: `peak` failed in campaign 2 and was de-prioritised, while `solar` kept producing small supported gains.

Next: [10.11 · Swarm evolution](../lab10_11_swarm/README.md)
