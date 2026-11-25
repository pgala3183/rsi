"""Lab 10.15 - Calculate RSIGym's RSI-Index for our three methods.

RSI-Index = mean over benchmarks of the fraction of the REMAINING gap closed:

        (final - initial) / (ideal - initial)

Benchmarks = the five GridWise tariffs. initial = the shipped self-consumption
program, ideal = the oracle (capture 1.0), final = the program each method returns.
All three methods get the SAME budget of day-simulations per benchmark.

Note: the course page writes the numerator as (initial - final). Its own worked
example uses (final - initial); we show both so the sign error is visible.

Run:  python labs/g05_environments_metrics/lab10_15_rsi_index/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import TARIFFS, make_task  # noqa: E402
from gridwise.evaluate import Evaluator, score_batch  # noqa: E402
from gridwise.methods.budget_rsi import calibrate, search  # noqa: E402
from gridwise.methods.so_rsi import SecondOrderImprover, run_improver  # noqa: E402
from gridwise.methods.swarm_rsi import run_swarm  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.15", "RSI-Index across five tariff benchmarks", "RSIGym, arXiv:2610.10310")
BUDGET = 48 * 100


def rsi_index(initial, final, ideal=1.0):
    return float(np.mean([(f - i) / (ideal - i) for i, f in zip(initial, final)]))


def course_formula(initial, final, ideal=1.0):
    return float(np.mean([(i - f) / (ideal - i) for i, f in zip(initial, final)]))


section("1. Worked example from the course page")
ini, fin, ide = [0.50, 0.20], [0.75, 0.60], 1.0
print(f"  benchmarks: initial {ini}, final {fin}, ideal {ide}")
print(f"  (final - initial)/(ideal - initial) -> {rsi_index(ini, fin):.2f}   (matches the page's worked answer 0.5)")
print(f"  (initial - final)/(ideal - initial) -> {course_formula(ini, fin):.2f}   (the formula as printed: sign error)")


def budget_rsi(ev, seed):
    cal = calibrate(ev, SEED_POLICY, np.random.default_rng(seed))
    nodes = 1 + max(3, (BUDGET - ev.day_sims) // 48)          # spend the rest of the matched budget on search
    return search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + seed), readout=cal.readout,
                  archive_size=nodes).best.policy


def so_rsi(ev, seed):
    ev.charge_repeats = True
    return run_improver(SecondOrderImprover, ev, SEED_POLICY, seed, BUDGET)[1]


def swarm_rsi(ev, seed):
    return run_swarm(ev, SEED_POLICY, np.random.default_rng(seed), BUDGET).best.policy


METHODS = {"Budget RSI": budget_rsi, "SO-RSI": so_rsi, "SwarmRSI": swarm_rsi}

section(f"2. Run each method on each benchmark ({BUDGET} day-sims each, 4 seeds)")
initial, finals, used = {}, {m: {} for m in METHODS}, {m: [] for m in METHODS}
dollars = {m: {} for m in METHODS}
for tariff in TARIFFS:
    task = make_task(tariff, seed=0)
    initial[tariff] = score_batch(SEED_POLICY, task.heldout).mean
    for m, fn in METHODS.items():
        caps, costs = [], []
        for seed in seeds(4):
            ev = Evaluator(task)
            pol = fn(ev, seed)
            used[m].append(ev.day_sims)
            caps.append(ev.final_report(pol).mean)
        finals[m][tariff] = float(np.mean(caps))
rows = [[t, f"{initial[t]:.3f}"] + [f"{finals[m][t]:.3f}" for m in METHODS] for t in TARIFFS]
table(rows, ["benchmark", "initial (seed)"] + [f"final: {m}" for m in METHODS])
print("  mean day-sims actually used per run: " + ", ".join(f"{m} {np.mean(v):.0f}" for m, v in used.items()))
print("  (SO-RSI can stop early: once its inquiry memory has banned every patch for the current symptoms")
print("   its first-order loop stalls. Unspent budget is reported, not redistributed.)")

section("3. Per-benchmark gap closed and RSI-Index")
rows = []
for t in TARIFFS:
    rows.append([t] + [f"{(finals[m][t] - initial[t]) / (1 - initial[t]):.3f}" for m in METHODS])
idx = {m: rsi_index([initial[t] for t in TARIFFS], [finals[m][t] for t in TARIFFS]) for m in METHODS}
wrong = {m: course_formula([initial[t] for t in TARIFFS], [finals[m][t] for t in TARIFFS]) for m in METHODS}
rows.append(["RSI-Index (mean)"] + [f"{idx[m]:.3f}" for m in METHODS])
rows.append(["course formula as printed"] + [f"{wrong[m]:.3f}" for m in METHODS])
table(rows, ["benchmark"] + list(METHODS))
print(f"\n  Paper: Opus 5 reaches RSI-Index 0.4809 under a $500 budget per benchmark (5 benchmarks).")
print("  Ours is a different system on a different task; only the METRIC is shared.")
print("  The index weighs a benchmark with little remaining gap (vpp_events, solar_glut) as heavily as one")
print("  with a large gap (wholesale): a small absolute gain on an almost-solved benchmark counts a lot.")
save("10.15", {"initial": initial, "final": finals, "rsi_index": idx, "course_formula": wrong,
               "day_sims_used": {m: float(np.mean(v)) for m, v in used.items()}}, t0)
