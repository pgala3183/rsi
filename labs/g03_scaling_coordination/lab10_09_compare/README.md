# 10.09 · Compare Scaling Approaches

**Purpose:** distinguish AReaL-RSI (leader-follower) from SwarmRSI (swarm evolution) by mechanism *and* by measured result.
**Run:** `python labs/g03_scaling_coordination/lab10_09_compare/run.py` (≈14 s) · **Evidence:** [`results/logs/lab10_09.txt`](../../../results/logs/lab10_09.txt)

## Walkthrough

The same task, proposer and budget (4,800 day-sims) for four arms, 8 seeds, 3 tariffs:
`run_team()` (AReaL-RSI), `run_swarm()` (SwarmRSI), `run_independent()` (2 researchers, no sharing), and `run_single()` (one hill climber).

## Mechanism comparison (from the code)

| aspect | AReaL-RSI | SwarmRSI |
|---|---|---|
| coordination | leader assigns orientations to 2 researchers | 5 peers, no leader |
| memory | conditional: supported vs unresolved | shared feedback board + archive of successful edits |
| evolution | evidence-based review re-weights orientations | mutation + complementary recombination + hypothesis restarts |
| selection | leader keeps only validated improvements | each agent keeps its best; the weakest is replaced by recombinants |
| diversity | low: researchers restart from the shared best | high, measured every generation (≈0.9 Jaccard distance, lab 10.17) |

## What we observed (held-out capture, 8 seeds)

| tariff | AReaL-RSI team | SwarmRSI | independent researchers | single agent |
|---|---|---|---|---|
| tou | 0.930 ± 0.031 | **0.933 ± 0.017** | 0.921 ± 0.014 | 0.923 ± 0.020 |
| critical_peak | 0.948 ± 0.012 | 0.942 ± 0.010 | 0.948 ± 0.015 | 0.941 ± 0.009 |
| wholesale | 0.756 ± 0.024 | 0.737 ± 0.035 | **0.770 ± 0.036** | 0.769 ± 0.022 |

On TOU both coordinated methods edge out the baselines, by less than one standard deviation. On wholesale, where improvement is many small independent threshold gains, coordination overhead costs more than it returns.

## Advantages, limitations, when to choose

| approach | key advantage | key limitation | choose it when |
|---|---|---|---|
| AReaL-RSI | cheap coordination; only validated findings propagate; an interpretable campaign log | orientations are hand-defined; a bad leader prior wastes campaigns | a known research agenda with distinct sub-problems |
| SwarmRSI | parallel diverse hypotheses; recombines complementary programs; a reusable archive | the budget is split 5 ways; agents converge over time | open-ended search, or many campaigns that can reuse experience |

## Quiz

**Q1. When is leader-follower better than a swarm?** When the sub-problems are known and distinct, so a leader can aim researchers at them. On critical-peak there is one dominant failure (an empty battery when the CPP event starts), and the team ties the best arm (0.948) while the swarm is lower (0.942).

Next: [10.10 · Leader-follower coordination](../lab10_10_leader_follower/README.md)
