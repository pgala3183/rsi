# 10.07 · Analyse Diagnostic Probes

**Purpose:** for each anomaly type, what triggers it, what the probe concludes, and whether acting on the conclusion helps.
**Run:** `python labs/g02_second_order_diagnostics/lab10_07_analyze_probes/run.py` (≈2 s) · **Evidence:** [`results/logs/lab10_07.txt`](../../../results/logs/lab10_07.txt)

## Walkthrough

The script runs SO-RSI for 8 seeds on TOU and critical-peak and collects:
* **triggers**: `detect()` replayed over SO-RSI's own first-order attempts
* **findings**: every `InquiryMemory` entry, classified by its text
* **interventions**: every probe-licensed edit (`sweep`, `reorder`, `reroute`, `split`, `guard`) and whether it was accepted
* **cost**: `Evaluator.by_purpose`, which bills every day-sim to the purpose that requested it (`search`, `probe:sweep`, ...)

It also runs Naive RSI with the **same detector switched to passive**, so we can see how much it thrashes when nobody acts on the anomalies.

## The three probes

| anomaly | detected when | probe | what it can conclude |
|---|---|---|---|
| **recurrence** | the same (slot, direction) failed ≥ 2× in the last 8 attempts | `sweep_probe`: set the threshold to 6 values across its range | *is a lever* → jump to the best value; *not a lever* → `lever_probe` replays the trace: is the rule **shadowed** by an earlier rule (→ reorder) or is the battery **empty** when it fires (→ re-route the symptom to the hold slot)? |
| **opposing** | the slot was nudged the other way within 4 attempts | `split_probe`: evaluate ±2 steps *per day*; search day-level context splits | days disagree → split the rule into two context-specific rules; otherwise use finer steps |
| **mismatch** | the targeted symptom moved but the score fell | `attribution_probe`: per-hour cost change, SoC and exported surplus | *blocks absorbing* → add an absorb rule above it; *grid charge displaced solar* → guard `surplus_left_fc ≤ v`; *energy spent early* → guard `peak_ahead ≤ v` |

## What we observed (16 runs)

| anomaly | count |
|---|---|
| mismatch | 91 |
| recurrence | 78 |
| opposing | 30 |
| two or more at once | 40 |

| anomaly | probe: finding | count |
|---|---|---|
| mismatch | attribution: inconclusive | 29 |
| mismatch | attribution: grid charge displaced solar | 16 |
| mismatch | attribution: blocks absorbing | 2 |
| recurrence | lever: no lever found | 64 |
| recurrence | sweep: IS a lever | 7 |
| opposing | split: no conflict | 7 |
| opposing | split: conflicting constraints | 4 |

| anomaly | interventions tried | accepted | capture gained |
|---|---|---|---|
| mismatch | 18 | 16 (89%) | 0.949 |
| opposing | 4 | 0 | 0.000 |
| recurrence | 7 | 7 | 0.022 |

Naive RSI's trace, analysed passively: 63.9 recurrences, 60.6 mismatches and 2.5 accepted patches out of 99 attempts, with a longest zero-progress streak of 78.8.

**Probes consume 64% of SO-RSI's day-sims.**

## Interpretation

* **Most findings are negative (78%), and that's still useful.** "No lever found" bans the patch, which is exactly what drives recurrences from ~64 to ~5. Negative evidence stops thrashing.
* **The productive path is mismatch → attribution → guard.** A rejected patch whose cost is concentrated in particular hours often has a clean cause ("grid charge displaced solar"), and the guard turns a bad patch into the best edit of the run.
* **Opposing edits rarely meant conflicting constraints here.** 4 of 11 split probes found a conflict, and none of the 4 resulting split rules improved the steering score. The split probe is the weakest of the three on this task.
* **Multiple anomalies at once:** 40 times. A rejected repeat of a harmful patch is both a recurrence and a mismatch, and both probes run (with cooldowns so the same pair isn't re-probed every attempt).

## Quiz

**Q1. Can multiple anomalies trigger simultaneously?** Yes, 40 times in these runs. `after_attempt()` runs every relevant probe, each with its own cooldown.

**Q2. A probe finds no clear root cause?** It is recorded as such (29 inconclusive attributions, 64 "no lever found"). The memory still bans the exhausted lever. Inconclusive evidence is the main reason SO-RSI's budget share for probes is so high, and on two tariffs it costs more than it returns (lab 10.06).

Next: [10.08 · Apply to a debugging task](../lab10_08_apply_debugging/README.md)
