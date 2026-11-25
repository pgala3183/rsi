# Paper audit — Second-Order Problem Solving for RSI in Formal Verification (SO-RSI, arXiv:2610.05701)

Note: the arXiv title includes "... in Formal Verification", which the course summary omits.

| Aspect | Paper's setup | Laptop adaptation (this repo) | What's preserved |
|--------|---------------|-------------------------------|------------------|
| Changed object | The RSI *workflow* (second-order), not just surface parameters | The improver's own workflow: symptom→patch table (`remap`), banned levers, learned guards, step sizes (`InquiryMemory`) | The improver edits *how it improves*, not only the policy |
| Feedback type | Execution traces of Lean 4 proof generation / Verus code generation | Trace of the improver's own attempts (`Attempt`), plus hourly simulator traces for probes | Anomalies are read from the improver's trace |
| Anomalies | Recurrence, opposing edits, expectation mismatch | Same three (`detect()`), judged only against *earlier* attempts (fixing the course demo's self-match bug) | All three |
| Probes | Lightweight diagnostic probes; persistent inquiry memory | Sweep + lever probe (recurrence), split probe (opposing), attribution probe (mismatch); memory saved to JSON | Probe → finding → structural edit |
| Evaluation protocol | Held-out pass rate vs Naive RSI under **matched 24-hour budgets** | Held-out capture vs `FirstOrderImprover` under matched day-sim budgets (probes are paid for); 8 seeds × 5 tariffs | Matched budget, same first-order loop on both arms |
| Compute budget | 24 h per arm | 4,800 day-sims per arm, under 1 s per run | Relative: probes compete with patches for budget |
| Untested assumptions | Generalisation beyond formal verification; how much of the budget probes consume; probe accuracy | Patch table and probes are hand-designed for this domain | — |

## Assumptions the paper makes but doesn't test (as reported in the abstract)

1. **Probes are cheap relative to the patches they save.** On GridWise, probes consume **64%** of SO-RSI's budget (lab 10.07). On tariffs where first-order patches already work, that cost makes SO-RSI lose (wholesale −0.013, solar_glut −0.041).
2. **Probes usually find a cause.** On GridWise, 78% of findings are negative or inconclusive. They are still useful (they ban exhausted levers), but they rarely license an edit. Nearly all of the gain comes from one path: mismatch → attribution → guard (16 of 18 accepted, +0.949 total capture).
3. **Matched wall-clock = matched opportunity.** A 24-hour budget hides how many evaluations each arm ran; we report day-sims and attempts (99 vs ~25).

## What would be needed to test them at paper scale

Log per-probe cost and outcome on Lean 4 / Verus runs (the lab 10.07 table at scale), and repeat with a cheaper first-order baseline that has its own memory (e.g. a tabu list), to separate "diagnosis" from "not repeating yourself".
