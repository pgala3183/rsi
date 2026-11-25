# 10.05 · Extend to Your Own Edits

**Purpose:** apply the scorer to our own editing task, and find where it does and doesn't pay off.
**Run:** `python labs/g01_budget_rsi/lab10_05_extend/run.py` (≈15 s) · **Evidence:** [`results/logs/lab10_05.txt`](../../../results/logs/lab10_05.txt)

## Walkthrough

**A. Task-aware diff coordinates.** `domain_features()` in `gridwise/edits.py` adds six coordinates, still computed from the diff *text* only: does the edit add discharge logic, add charge logic, touch price logic, touch the SoC guard, touch solar logic, and how high in the priority order is it? `featurize(..., "diff+domain")` gives 15 inputs; calibration and search are otherwise unchanged.

**B. Transfer.** Calibrate a readout on TOU, freeze it, and use it unchanged to steer search on critical-peak, solar_glut and wholesale. Compare it with a readout calibrated on the target, with HGM, and with the root.

**C. Matched total cost** (the theme 08 question: *did we measure cost fairly?*). For each run, the Budget RSI bill is calibration plus search. The same number of day-sims is then given to plain HGM search (≈30 nodes instead of 4). We also test one readout **amortised** over all 12 runs.

## What we observed

| features | readout R² | held-out capture |
|---|---|---|
| 8 diff coordinates | 0.390 | 0.817 ± 0.074 |
| + 6 domain coordinates | 0.470 | **0.847 ± 0.062** |

The top new coefficient is `touches_solar_logic` (+0.15): edits to the absorb/solar rules are the safest wins from the factory default.

| target tariff | frozen TOU readout | calibrated on target | HGM | root |
|---|---|---|---|---|
| critical_peak | 0.823 | 0.835 | 0.790 | 0.790 |
| solar_glut | 0.878 | 0.878 | 0.850 | 0.779 |
| wholesale | 0.569 | 0.573 | 0.557 | 0.501 |

The frozen readout transfers: within 0.012 of a target-calibrated one, and better than HGM on all three tariffs. The diff coordinates describe *how risky an edit is*, which is fairly task-independent.

| method | day-sims / run | held-out capture |
|---|---|---|
| Budget RSI, calibrated per run | 1,455 | 0.817 ± 0.074 |
| calibration-best edit (no search) | 1,263 | 0.887 ± 0.030 |
| **HGM search, same total budget** | 1,500 | **0.863 ± 0.061** |
| Budget RSI, readout amortised over 12 runs | 296 | 0.833 ± 0.069 |
| HGM search, amortised budget | 336 | 0.838 ± 0.074 |

## Bottom line for the Budget RSI group

* At the paper's operating point (4 nodes, fixed search budget), the readout beats HGM, parent-rate and random.
* Counting the calibration bill, the same day-sims spent on plain HGM search are **as good or better**. Amortising one calibration closes the cost gap but doesn't overtake HGM.
* The readout earns its keep with **task-aware coordinates** and **cross-task reuse**: one calibration, many searches.

## Quiz

**Q1. What if success doesn't correlate with the diff features?** The readout becomes random selection with extra cost. Lab 10.01 showed this directly on the course demo: shuffling its features removed about 80% of its gain.

**Q2. Can Budget RSI work for non-code tasks?** Here it does: the "code" is a controller and the features come from its diff. The requirements are (1) a steering split, (2) features of a proposed change that can be read without running it, and (3) a correlation between those features and quality. Part B suggests (3) can be fairly task-independent.

Next: [Group 02 · SO-RSI](../../g02_second_order_diagnostics/README.md)
