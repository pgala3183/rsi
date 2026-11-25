# Paper audit — Budget RSI: Scoring Code Edits Before They Run (arXiv:2610.29121)

| Aspect | Paper's setup | Laptop adaptation (this repo) | What's preserved |
|--------|---------------|-------------------------------|------------------|
| Changed object | Coding-agent scaffold (self-modifying agent code) | The battery-control program (`Policy` rules) | An agent editing the program it runs |
| Feedback type | Steering-split solve counts; diff of each proposed edit | Steering-split capture and solve counts; unified diff of the rendered source | The readout sees the parent's rate + 8 diff coordinates, never the child's result |
| Selection | Frozen linear readout picks which proposal to expand; archive of 4 nodes; returns earliest node with maximal steering solve count | Identical rules (`search()` in `gridwise/methods/budget_rsi.py`) | Exactly |
| Evaluation protocol | HumanEval+: 16 fit / 48 steering / 100 held-out; Polyglot: 64 steering / 145 held-out; 12 runs | 16 fit / 48 steering / 100 held-out days; 12 runs; held-out guarded by `BoundaryViolation` | Split sizes, run count, frozen boundary |
| Baselines | HGM selection, parent-rate ablation, root, calibration-best | All four, plus random | All |
| Compute budget | GPT-4o / Sonnet 4.5 calls; 96 edits × 16 instances × 4 evaluations for calibration | 1,263 fit day-sims (calibration) + 192 steering day-sims (search) per run; about 2 s for all 12 runs | Relative cost structure (calibration ≫ search) |
| Untested assumptions | That the readout transfers across tasks/models beyond the two tested; that calibration cost is amortised; behaviour at larger archives | Our proposer is a seeded generator, not an LLM; evaluations are deterministic | — |

## Assumptions the paper makes but doesn't test (as reported in the abstract)

1. **The calibration bill is worth paying.** The abstract compares selection rules at a fixed search budget, not at a fixed *total* budget including the 96 labelled edits.
   *To test:* compare at matched total cost. Lab 10.05 does this: on GridWise, HGM with the same total budget reaches 0.863 vs the readout's 0.817.
2. **Diff coordinates carry the same meaning across codebases.** "Lines added" means different things in a 30-line policy and a 3,000-line scaffold.
   *To test:* freeze a readout from one task and use it on another. Lab 10.05 shows a TOU-calibrated readout used on three other tariffs performs within 0.012 of one calibrated on each target.
3. **Four nodes is the relevant regime.** With more nodes, a cheap scorer matters less.

## Laptop adaptation constraints

No LLM, so edit *proposal* quality is fixed by `gridwise/edits.py`. The readout is fitted to that proposer's distribution, and conclusions about an LLM proposer would need re-calibration.
