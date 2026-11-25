# 10.13 · Simulate the RSIFlow Meta-Agent

**Purpose:** a workflow graph with adaptive routing, and a meta-agent that edits the graph.
**Run:** `python labs/g04_adaptive_workflows/lab10_13_rsiflow/run.py` (≈9 s) · **Evidence:** [`results/logs/lab10_13.txt`](../../../results/logs/lab10_13.txt)

## Walkthrough of `rsiflow.py`

* **The graph:** `diagnose → {tune | extend | restructure | prune} → evaluate`. Each branch is a set of edit operators (`BRANCHES`). `extend` uses rule templates biased toward the current bottleneck (`SYMPTOM_TEMPLATES` + `soft_templates`).
* **`MetaAgent.route(bottleneck)`:** UCB over branches, using credit **conditioned on the bottleneck**, so the same branch can be good for one symptom and useless for another.
* **`MetaAgent.credit()`** records the gain each routing produced.
* **`MetaAgent.evolve()`**, where the meta-agent rewrites the workflow itself:
  * prune a branch with ≥ 12 visits and zero total gain
  * after 10 stalled steps, **grow a composite branch** `extend+tune` (add a rule, then tune it immediately)
* **Static baseline:** the same branches, visited round-robin, with no bottleneck-specific templates.

## What we observed

The TOU run's graph edit: `step 23: progress stalled -> grew composite branch 'extend+tune'`.

| bottleneck | branch | visits | mean gain ×100 |
|---|---|---|---|
| uncovered_peak | restructure | 1 | 7.62 |
| wasted_solar | restructure | 30 | 0.28 |
| wasted_solar | tune | 8 | 0.05 |
| wasted_solar | extend / extend+tune / prune | 6 each | 0.00 |

| tariff | static workflow | RSIFlow meta-agent | adaptive − static (wins) |
|---|---|---|---|
| tou | 0.896 | 0.895 | −0.000 (3/8) |
| critical_peak | 0.943 | 0.932 | −0.011 (1/8) |
| wholesale | 0.778 | 0.773 | −0.005 (5/8) |
| solar_glut | 0.974 | 0.972 | −0.002 (3/8) |

## Interpretation

**No measurable gain on GridWise (−0.004 on average).** The run shows why: the bottleneck was `wasted_solar` in **56 of 58 steps**. RSIFlow's premise, *evolving* bottlenecks that need different actions, barely holds here. With one fixed bottleneck, routing has nothing to adapt to, and UCB still spends steps exploring branches for that bottleneck. A negative result like this is informative: it says when *not* to pay for a meta-agent.

## Quiz-style question

**What change to the task would make RSIFlow pay off?** Bottlenecks that move. For example, a curriculum that rotates tariffs during the run, so the dominant symptom shifts from `wasted_solar` to `uncovered_peak` to `missed_grid_charge` and the bottleneck-conditioned credit table becomes useful.

Next: [10.14 · Self-HPO alternation](../lab10_14_self_hpo/README.md)
