# Claim–evidence analysis: Budget RSI (arXiv:2610.29121)

Source: the paper's abstract (Darzi, Balija, He; submitted 21 Oct 2026), plus the course summary. Laptop numbers come from labs 10.03–10.05 of this repo.

## Claim 1 — "A frozen linear readout over the parent's steering solve rate and eight diff coordinates picks better edits than HGM selection."

| | |
|---|---|
| **Supporting experiment** | GPT-4o, twelve runs, HumanEval+ held-out: readout 87.0% vs HGM 85.1%, parent-rate ablation 84.2%, root 81.2%, calibration-best 82.8%. Claude Sonnet 4.5: 86.6% vs HGM 86.5%. |
| **Conditions that limit it** | Two benchmarks (HumanEval+, Polyglot), two base models, an archive of only **four** nodes, 96 calibration edits labelled by four evaluations of a 16-instance fit split. With Sonnet 4.5 the margin over HGM is 0.1 points, inside run-to-run noise. |
| **Laptop adaptation preserves** | The 16 / 48 / 100 split, the 96-edit calibration, the frozen readout, the 4-node archive, the "earliest node with maximal steering solve count" return rule, and all baselines (HGM Thompson sampling, parent-rate, root, calibration-best). |
| **Laptop adaptation changes** | Scaffold edits become edits to a battery-control program. Solve rate becomes the mean share of oracle savings captured. Deterministic simulation replaces four LLM evaluations. A seeded edit generator replaces the LLM proposer. |
| **What we found** | Same ordering: readout 0.817 > HGM 0.787 > parent-rate 0.770 > root 0.712 (12 runs). But paired against HGM the readout wins only 5/12 (3 ties, 4 losses). |

## Claim 2 — "Scoring before running reduces the cost of self-improvement."

| | |
|---|---|
| **Supporting experiment** | The readout ranks candidates without running them; only the selected child is evaluated. |
| **Conditions that limit it** | The calibration labels (96 edits × 16 instances × 4 evaluations) are a cost in their own right. Whether the method is cheap overall depends on amortising that cost across runs. |
| **Laptop adaptation preserves** | Day-simulations are billed by `Evaluator.ledger`, including calibration. |
| **What we found** | Calibration costs 6.6× the search (1,263 vs 192 day-sims). At the *same total budget*, plain HGM search reaches 0.863 vs the readout's 0.817. A readout calibrated once and reused for 12 runs costs 296 day-sims per run and reaches 0.833, about the same as HGM on the same budget (0.838). The cost claim holds for the search step, not for the whole method unless calibration is amortised. |

## Claim 3 — "Parent solve rate alone is not enough; the diff coordinates add information."

| | |
|---|---|
| **Supporting experiment** | The parent-rate ablation (84.2%) is below the full readout (87.0%). |
| **Conditions that limit it** | The diff coordinates are syntactic (lines, files, imports), so their meaning depends on the codebase and the proposer. |
| **Laptop adaptation preserves** | Eight coordinates computed from the unified diff text only: lines ±, rules ±, conditions Δ, constants changed, returns changed, reordered. |
| **What we found** | Supported, more strongly than in the paper. Diff-only matches the full readout (0.817) and parent-rate-only drops to 0.756. On GridWise the parent-rate coefficient is near zero and unstable in sign (58% agreement across 12 calibrations). Adding six task-aware coordinates raises held-out capture to 0.847 (lab 10.05). |

## What remains untested (paper and adaptation)

* Real LLM proposers on GridWise (an LLM's edit distribution differs from our generator's, and the readout is fitted to the proposer).
* Larger archives. The paper's 4-node setting is where a scorer helps most; by 16+ nodes HGM catches up (lab 10.03 trial and lab 10.05).
* Noisy evaluations. The paper averages four LLM runs; our simulator is deterministic, which makes the calibration-best baseline unusually strong here (0.887).
