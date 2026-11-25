# 10.17 · Analyse Metrics

**Purpose:** for each new October 2026 metric, say what problem it solves, when it's useful and where it fails, *with a measurement*.
**Run:** `python labs/g05_environments_metrics/lab10_17_metrics/run.py` (≈6 s) · **Evidence:** [`results/logs/lab10_17.txt`](../../../results/logs/lab10_17.txt)

## Walkthrough

1. **RSI-Index:** loaded from lab 10.15's JSON.
2. **Steering-split solve rate (Budget RSI):** 6 HGM searches with 30 nodes = 180 evaluated programs. For each node, compare its steering score (what the search saw) with its held-out score (via the auditor-side `final_report`). Also measure the **winner's curse**: how optimistic the *selected* node's steering score is.
3. **Recursive Learnability (R2Orch):** a population of 31 programs gives each steering day a solve rate *p*. Train a hill climber on 12 days chosen four ways (max p(1−p), hardest, easiest, random) with the same 80 evaluations, then report held-out capture.
4. **Swarm diversity (SwarmRSI):** mean pairwise rule-set Jaccard distance between agents, per generation, against the number of successful recombinations.

## What we observed

* **RSI-Index:** Budget RSI 0.524, SO-RSI 0.607, SwarmRSI 0.732.
* **Steering proxy:** Spearman(steering capture, held-out) = **0.944**; using solve *rate* instead = 0.852. Counts lose information. The selected node is **optimistic by +0.037 ± 0.024**: the winner's curse of selecting a maximum on 48 days.
* **Learnability:**

| 12 training days chosen by | held-out capture |
|---|---|
| learnable (max p(1−p)) | 0.794 ± 0.101 |
| hardest (lowest p) | 0.877 ± 0.035 |
| easiest (highest p) | 0.739 ± 0.077 |
| **random** | **0.913 ± 0.032** |

  The most "learnable" days are a narrow slice of day types (those whose outcome *differs* across programs). Optimising on that slice alone transfers worse than a random sample. Learnability tells you what to *add* to a curriculum, not what to replace it with.
* **Diversity:** 0.00 → 0.97 after the first generation, then 0.87–0.99 throughout; 2–6 successful recombinations per run. Diversity stays high, but only a few pairs are complementary enough to recombine. High syntactic diversity does not imply complementary behaviour.

## Summary table (the lab's expected output)

| metric | problem it solves | useful when | limitation (measured here) |
|---|---|---|---|
| RSI-Index | compares improvement across benchmarks of different difficulty | reporting one number over a suite | near-solved benchmarks weigh as much as hard ones; needs a trustworthy "ideal" (ours is accurate to one SoC step) |
| steering-split solve rate | a cheap proxy for selection during search | it ranks like held-out (Spearman 0.94 here) | the selected node is optimistic by +0.037; solve counts rank worse than means (0.85) |
| Recursive Learnability | chooses practice tasks that separate good from bad solvers | self-play curricula without labels | a learnable-only curriculum scored 0.794 vs 0.913 for random days: unrepresentative |
| swarm diversity | detects premature convergence | deciding when to inject new hypotheses | stayed ≈0.9 while recombination succeeded only 2–6 times: diversity ≠ complementarity |

## Quiz

**Q1. Why do we need so many RSI metrics?** Each improver changes a different object: an edit selector, a workflow, a swarm, a curriculum. Each metric watches one failure mode of that object. A single headline number (even RSI-Index) can hide the failure mode that matters: winner's-curse optimism, an unrepresentative curriculum, or a converged swarm.

Back to the [repository overview](../../../README.md).
