# GridWise RSI — Section 10 codelabs (October 2026 Research Studio)

Progressive, executable implementations of **all 17 codelabs in Section 10** of the
[RSI Masterclass course map](https://github.com/dlmastery/simple-coding-harness/blob/main/COURSE-MAP.md#theme-10),
applied to one problem area of my own: **a home solar + battery controller that improves its own control program.**

Three state-of-the-art (October 2026) recursive self-improvement methods are rebuilt in full and compared under one protocol:

| # | Method | Paper | What it changes | Where |
|---|--------|-------|-----------------|-------|
| 1 | **Budget RSI**: score code edits before they run | arXiv:2610.29121 | *which* proposed edit gets evaluated (a frozen linear readout over diff features) | [`gridwise/methods/budget_rsi.py`](gridwise/methods/budget_rsi.py), labs 10.03–10.05 |
| 2 | **SO-RSI**: second-order diagnostic inquiry | arXiv:2610.05701 | the *improver's own workflow* (anomaly → probe → inquiry memory → re-mapped patches, bans, guards) | [`gridwise/methods/so_rsi.py`](gridwise/methods/so_rsi.py), labs 10.06–10.08 |
| 3 | **SwarmRSI**: evolving scientific agent swarm | arXiv:2610.20431 | a *population of hypotheses* (feedback-guided mutation, complementary recombination, hypothesis evolution, retained experience) | [`gridwise/methods/swarm_rsi.py`](gridwise/methods/swarm_rsi.py), labs 10.09–10.11 |

The other papers in Section 10 are implemented as smaller mechanisms on the same task: AReaL-RSI (leader-follower, 10.10), RSIFlow (meta-agent workflow routing, 10.13), Self-HPO (harness-policy alternation, 10.14) and RSIGym (Everything-as-a-Service environment + RSI-Index, 10.15–10.16).

---

## The problem area: GridWise

Each **instance is one day**: 24 hourly decisions for a 13.5 kWh / 5 kW home battery next to a 7 kW solar array. The controller sees day-ahead prices, noisy solar and load forecasts, and its state of charge. It picks one of 7 actions per hour (charge, discharge, absorb surplus, hold, ...). The controller is a **small program**:

```python
def policy(o):
    if o.export_price >= 0.50:                                   # dump into price spikes / VPP events
        return DISCHARGE_MAX
    if o.price_rank <= 0.25 and o.surplus_left_fc <= 8.00 and o.soc <= 0.85:
        return CHARGE_MAX                                        # grid-charge only before cloudy days
    if o.peak_ahead >= 0.35 and o.price_rank <= 0.60:
        return IDLE                                              # hold energy for the evening peak
    return SELF_CONSUME
```

The improvers edit that program. Why this domain makes a good RSI testbed:

* **There is a ground-truth ceiling.** A dynamic-programming oracle with perfect foresight gives the cheapest possible day. Every day gets a score: the *fraction of the oracle's savings the program captured*. That gives RSI-Index a real "ideal" (accurate to one 0.25 kWh step; see lab 10.02).
* **Good behaviour is conditional.** Grid-charging at night helps on cloudy days and hurts on sunny ones, and holding energy only pays if the peak gets covered. Threshold patches therefore thrash, and structural edits are needed: exactly the regime the October 2026 papers target.
* **Five tariffs act as five benchmarks** (`tou`, `vpp_events`, `critical_peak`, `wholesale`, `solar_glut`).
* **It runs on a laptop.** All of it is NumPy and seeded: the 17 labs take about 3.5 minutes on a CPU, with no GPU and no API key.

The shipped starting program (`return SELF_CONSUME`, what most home batteries do) captures 0.71 of the oracle's savings on TOU days. A hand-written expert program captures 0.92.

## Quick start

```powershell
pip install -r requirements.txt
```

```powershell
python run_all.py
```

```powershell
python labs/g02_second_order_diagnostics/lab10_06_run_demo/run.py
```

`run_all.py` runs all 17 labs in order and writes each console log to [`results/logs/`](results/logs) and structured results to [`results/`](results). `python run_all.py --fast` is a fewer-seed smoke test; `python run_all.py 10.06 10.07` runs selected labs.

## Results at a glance (all numbers are held-out capture on 100 unseen days)

**RSI-Index across the five tariffs, same budget (4,800 day-simulations) for every method** (lab 10.15):

| | Budget RSI | SO-RSI | SwarmRSI |
|---|---|---|---|
| RSI-Index (mean fraction of remaining gap closed) | 0.524 | 0.607 | **0.732** |

**Each method's own claim, tested** (each with ablations and seeds; details in its lab):

| Claim from the paper | Our result on GridWise | Lab |
|---|---|---|
| Budget RSI: readout > HGM > parent-rate > root (at 4-node archive) | **Same ordering**: 0.817 > 0.787 > 0.770 > 0.712 (12 runs) | 10.03 |
| ...and this saves compute | Only if calibration is amortised. At equal total cost, plain HGM search matches or beats it (0.863 vs 0.817) | 10.05 |
| SO-RSI beats Naive RSI under matched budgets | +0.122 (TOU), +0.083 (critical-peak), tie (VPP), −0.013 / −0.041 on two tariffs where cheap patches already work | 10.06 |
| SO-RSI suppresses recurrence / zero-progress loops | **Yes**: recurrences 60 → 6 per run; longest zero-progress streak 89 → 9 (TOU) | 10.06, 10.07 |
| SwarmRSI: mutation + recombination matter | On TOU, removing recombination costs 0.022 and removing the feedback board 0.026. On critical-peak and wholesale, no measurable effect | 10.11 |
| SwarmRSI: accumulated experience improves later research | Yes: fresh 0.907 → 0.929 on a new tariff. But a control (single agent + the retained program) gets 0.933, so the gain is mostly program transfer | 10.11 |
| AReaL-RSI: sharing beats independent exploration (1.9–27.5%) | Sharing wins on 3/3 tariffs (best variant +0.8% to +5.7%). The conditional filter itself wins on 1/3 | 10.10 |
| Self-HPO: joint gain > sum of parts | Holds on 1/4 tariffs (TOU, +0.032) | 10.14 |
| RSIFlow: adaptive routing beats a fixed workflow | Not here: −0.011 to 0.000. The bottleneck barely changes on this task | 10.13 |

Some results don't reproduce. That's a finding, not a failure: each lab says *why*, and the evidence is in its log.

**What the course's own demos show** (lab 10.01): the reference SO-RSI demo's "+20 points" comes from hard-coded success probabilities (0.3 vs 0.5). It is identical with diagnostics switched off, and its detectors compare each trace with itself. The reference Budget RSI demo generates held-out scores *from* the steering score, so the readout cannot fail. That is why everything here runs on a simulator with an oracle.

## The 17 codelabs

| Lab | Codelab | What you see when you run it |
|-----|---------|------------------------------|
| **Group 00 · Read a frontier claim** | [README](labs/g00_reading_frontier_research/README.md) | |
| 10.01 | [Distinguish claim from evidence](labs/g00_reading_frontier_research/lab10_01_claim_vs_evidence/README.md) | Ablating the course demos: their headline gains are built into the data generators |
| 10.02 | [Audit a paper](labs/g00_reading_frontier_research/lab10_02_audit_paper/README.md) | Paper audits for all three methods, and 9 automated protocol checks on this harness |
| **Group 01 · Budget RSI** | [README](labs/g01_budget_rsi/README.md) | |
| 10.03 | [Run the Budget RSI demo](labs/g01_budget_rsi/lab10_03_run_demo/README.md) | Calibration, one scored diff, one search, 12-run selector comparison, measured cost |
| 10.04 | [Analyse the scoring mechanism](labs/g01_budget_rsi/lab10_04_analyze_mechanism/README.md) | Coefficient stability, fresh-edit ranking (Spearman 0.44), input ablations, training-set size |
| 10.05 | [Extend to your own edits](labs/g01_budget_rsi/lab10_05_extend/README.md) | Domain diff features (+0.030), cross-tariff transfer, matched-cost bottom line |
| **Group 02 · SO-RSI** | [README](labs/g02_second_order_diagnostics/README.md) | |
| 10.06 | [Run the SO-RSI demo](labs/g02_second_order_diagnostics/lab10_06_run_demo/README.md) | Full annotated trace (anomaly → probe → finding → edit), 5-tariff matched-budget comparison |
| 10.07 | [Analyse diagnostic probes](labs/g02_second_order_diagnostics/lab10_07_analyze_probes/README.md) | What each probe concluded, which interventions survived, cost share of probes (64%) |
| 10.08 | [Apply to a debugging task](labs/g02_second_order_diagnostics/lab10_08_apply_debugging/README.md) | A planted off-by-one-hour pipeline bug, diagnosed from a 5-attempt debugging trace |
| **Group 03 · Scaling: coordination & swarms** | [README](labs/g03_scaling_coordination/README.md) | |
| 10.09 | [Compare scaling approaches](labs/g03_scaling_coordination/lab10_09_compare/README.md) | AReaL-RSI vs SwarmRSI vs baselines at one budget |
| 10.10 | [Leader-follower coordination](labs/g03_scaling_coordination/lab10_10_leader_follower/README.md) | Campaign log, conditional memory table, sharing vs independent |
| 10.11 | [Swarm evolution](labs/g03_scaling_coordination/lab10_11_swarm/README.md) | Recombination events, diversity curve, ablations, budget curve, experience transfer + control |
| **Group 04 · Adaptive workflows** | [README](labs/g04_adaptive_workflows/README.md) | |
| 10.12 | [Compare workflow adaptation](labs/g04_adaptive_workflows/lab10_12_compare/README.md) | RSIFlow vs Self-HPO on four tariffs |
| 10.13 | [RSIFlow meta-agent](labs/g04_adaptive_workflows/lab10_13_rsiflow/README.md) | Routing log, credit table, graph edits, adaptive vs static |
| 10.14 | [Self-HPO alternation](labs/g04_adaptive_workflows/lab10_14_self_hpo/README.md) | H-step/P-step log, learned guards, synergy computation |
| **Group 05 · Environments & metrics** | [README](labs/g05_environments_metrics/README.md) | |
| 10.15 | [Calculate RSI-Index](labs/g05_environments_metrics/lab10_15_rsi_index/README.md) | All three methods on 5 benchmarks; the course formula's sign error |
| 10.16 | [Compare environments](labs/g05_environments_metrics/lab10_16_environments/README.md) | EaaS sandbox / permissions / budget meter, RSIGym's 3 tracks, terminal CLI, learnability |
| 10.17 | [Analyse metrics](labs/g05_environments_metrics/lab10_17_metrics/README.md) | Steering proxy validity (Spearman 0.94, winner's curse +0.037), learnability curricula, diversity |

## Repository layout

```
gridwise/                    the task harness (shared by every lab)
  env.py                     days, tariffs, battery simulator, perfect-foresight oracle (DP), splits
  policy.py                  policies as rule programs: readable source + vectorised interpreter
  edits.py                   the proposer (edit operators, rule templates) + diff features
  evaluate.py                Evaluator: cost ledger, held-out boundary guard, caching
  diagnose.py                oracle-relative failure symptoms
  eaas.py, cli.py            RSIGym-style services (sandbox, permissions, budget) + terminal CLI
  methods/                   budget_rsi.py  so_rsi.py  swarm_rsi.py  areal_rsi.py  rsiflow.py  self_hpo.py
labs/g00..g05/lab10_XX_*/    one folder per codelab: run.py + README.md walkthrough
reference/course_demos/      the course's two laptop demos, vendored unchanged (MIT) for lab 10.01
results/                     lab10_XX.json (structured) + logs/lab10_XX.txt (console evidence)
run_all.py                   runs every lab, saves logs
VIDEO_GUIDE.md               recording script for each part
```

## How the harness keeps results honest

* **Frozen splits** mirror the Budget RSI paper's HumanEval+ split: 16 fit / 48 steering / 100 held-out days (164 total).
* **The held-out boundary is enforced in code.** An improver that reads held-out data gets a `BoundaryViolation`; only the experimenter's `final_report()` can (lab 10.02 checks this).
* **Every simulated day is billed** to a ledger, so all method comparisons use matched day-simulation budgets. Probes, calibration and repeat evaluations by memoryless improvers are paid for too.
* **The code you read is the code that runs.** The readable policy source is exec'd and compared with the vectorised interpreter (0 mismatches).
* **Every run is reproducible**: everything is seeded and re-running gives identical results.
* **Ablations and controls** accompany every claimed mechanism.

## What is and is not claimed

These are **laptop adaptations**. They preserve each paper's *mechanism* (what is changed, what feedback drives it, how it's evaluated), not its scale, models or headline numbers. The proposer is a seeded program generator, not an LLM, so runs are free and reproducible. A paper's result transferring to GridWise is evidence that the mechanism works outside its original benchmark. A result *not* transferring is evidence about the mechanism's preconditions, not a refutation of the paper.

## Videos

One video per part (see [`VIDEO_GUIDE.md`](VIDEO_GUIDE.md) for the scripts):

| Part | Topic | YouTube |
|------|-------|---------|
| 0 | Overview: GridWise harness, protocol, how to run | _link to add_ |
| 1 | Group 00: claim vs evidence, paper audits (10.01–10.02) | _link to add_ |
| 2 | Group 01: Budget RSI (10.03–10.05) | _link to add_ |
| 3 | Group 02: SO-RSI (10.06–10.08) | _link to add_ |
| 4 | Group 03: AReaL-RSI and SwarmRSI (10.09–10.11) | _link to add_ |
| 5 | Group 04: RSIFlow and Self-HPO (10.12–10.14) | _link to add_ |
| 6 | Group 05: RSIGym, RSI-Index, metrics (10.15–10.17) | _link to add_ |

## Credits

* Course: [dlmastery/simple-coding-harness](https://github.com/dlmastery/simple-coding-harness), Theme 10 (Research Studio). The two reference demos in `reference/course_demos/` are vendored unchanged under their MIT license.
* Papers: Budget RSI (arXiv:2610.29121), SO-RSI (arXiv:2610.05701), SwarmRSI (arXiv:2610.20431), AReaL-RSI (arXiv:2610.18083), RSIFlow (arXiv:2610.21791), Self-HPO (arXiv:2610.22791), RSIGym (arXiv:2610.10310), Terminal-RSI (arXiv:2610.22192), R2Orch (arXiv:2610.22740).
* Built with AI coding assistance (Claude Code), as the assignment permits. The walkthroughs explain each codelab step by step.
