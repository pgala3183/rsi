# 10.01 · Distinguish Claim from Evidence

**Purpose:** a headline score does not prove the mechanism.
**Run:** `python labs/g00_reading_frontier_research/lab10_01_claim_vs_evidence/run.py` (≈1 s) · **Evidence:** [`results/logs/lab10_01.txt`](../../../results/logs/lab10_01.txt)
**Written analysis (the lab's expected output):** [`claim_evidence_analysis.md`](claim_evidence_analysis.md)

## Why this lab

The course gives two "full laptop demos" (vendored unchanged in [`reference/course_demos/`](../../../reference/course_demos)). Each prints a striking result: SO-RSI +20 points over Naive RSI, and Budget RSI +40% over random. Before reusing them, ask: *what in the code produces the number?*

## Walkthrough of `run.py`

1. **`load()`** imports the vendored demos by file path, so we run the course's exact code.
2. **A1 – the SO-RSI demo, 300 times.** `run_pair(diagnostics_on)` replays the demo's 20-attempt comparison. The ablation monkey-patches `StructuralAnomalyDetector.monitor_trace` to report no anomalies. If diagnostics caused the gain, the gap should vanish. The script then greps the demo for the success probabilities.
3. **A2 – do the detectors detect?** It feeds a single failure into a fresh detector (recurrence should *not* fire), then "increase lr" followed by "decrease lr" (opposing *should* fire).
4. **B1 – is held-out independent evidence?** It correlates the generated steering and held-out counts.
5. **B2 – does selection survive uninformative features?** It shuffles the diff features across edits (keeping everything else) and re-runs the readout's selection 200 times.

## What we observed

| SO-RSI demo, 300 trials | naive success | SO-RSI success | gap |
|---|---|---|---|
| diagnostics ON (as shipped) | 0.297 | 0.501 | 0.204 |
| diagnostics OFF (ablation) | 0.297 | 0.501 | **0.204** |

* The gap is identical with diagnostics off. It comes from `so_rsi_demo.py:203` (`random.random() < 0.3`) vs `:249` (`< 0.5`).
* `monitor_trace()` appends the trace *before* checking. So **recurrence fires on the very first failure** (the trace matches itself), and **opposing never fires** (each edit is compared with itself).
* Budget RSI demo: corr(steering, held-out) = 0.92, **by construction** (`held_out = steering*6 + noise`). Shuffling the diff features cuts the selection gain from 40.7 to 8.4 solves per 100; the rest comes from `parent_rate`, which the generator also wires into the outcome. `HuxleyGodelSelector` is defined but never used, and the "88% cost reduction" is printed arithmetic.

## Interpretation

The demos show that the code paths run. They do **not** show that either mechanism causes improvement. That's why the rest of this repo rebuilds the methods on GridWise: outcomes come from simulating the programs, the ceiling is an oracle, and every gain is checked against an ablation at a matched budget.

## Quiz

**Q1. Can the Budget RSI paper claim linear readout scoring works for all code-editing tasks?**
No. The abstract reports HumanEval+ and Polyglot with GPT-4o and Claude Sonnet 4.5. Lab 10.05 shows it partly transfers to GridWise, and that at matched total cost plain HGM search is as good. Generality has to be tested per task.

**Q2. Why does the laptop demo use synthetic edits?**
To run without GPT-4o or HumanEval+. The cost is that synthetic outcomes can encode the answer. Here they did, which is why our adaptation uses real program edits scored by a simulator.

Next: [10.02 · Audit a paper](../lab10_02_audit_paper/README.md)
