# 10.12 · Compare Workflow Adaptation: RSIFlow vs Self-HPO

**Purpose:** distinguish meta-agent orchestration from harness-policy alternation, then decide when each is preferable.
**Run:** `python labs/g04_adaptive_workflows/lab10_12_compare/run.py` (≈14 s) · **Evidence:** [`results/logs/lab10_12.txt`](../../../results/logs/lab10_12.txt)

## What each method adapts

| aspect | RSIFlow | Self-HPO |
|---|---|---|
| changed object | the routing of improvement actions (and the workflow graph itself) | the harness (curriculum, runtime guards) **and** the policy |
| feedback | credit per (bottleneck, branch) | the policy's own failure symptoms on validation days |
| iteration | continuous: route → evaluate → credit | alternating H-step / P-step |
| deployed artefact | a policy program | a policy program + a runtime wrapper |

## What we observed (8 seeds, 2,880 day-sims, held-out capture)

| tariff | RSIFlow | Self-HPO | RSIFlow graph edits / run | Self-HPO guards kept / run |
|---|---|---|---|---|
| tou | 0.888 ± 0.029 | **0.916 ± 0.015** | 1.0 | 1.1 |
| critical_peak | 0.934 ± 0.005 | 0.936 ± 0.009 | 1.0 | 1.9 |
| wholesale | **0.756 ± 0.047** | 0.714 ± 0.045 | 0.5 | 3.2 |
| vpp_events | 0.962 | 0.961 | 1.4 | 1.1 |

Self-HPO wins where one runtime guard captures a real rule ("no grid charging on sunny days" on TOU). It loses on wholesale, where it keeps 3.2 guards chosen on 16 validation days that don't generalise.

## When is each preferable?

| method | preferable when | weak when |
|---|---|---|
| RSIFlow | the bottleneck shifts over time and different bottlenecks need different actions | the bottleneck is stable (as on most GridWise tariffs): routing reduces to round-robin plus exploration cost |
| Self-HPO | failures can be fixed by wrapping the policy (a missing guard, a skewed curriculum) | there are few validation days, so guards overfit, and you must ship and maintain the wrapper too |

Next: [10.13 · RSIFlow meta-agent](../lab10_13_rsiflow/README.md)
