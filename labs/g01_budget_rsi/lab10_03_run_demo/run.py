"""Lab 10.03 - Run Budget RSI (arXiv:2610.29121) on GridWise.

Walkthrough
  1. the task, the starting program and the oracle
  2. calibration: 96 edits labelled on the 16-day fit split -> one frozen linear readout
  3. one proposal, its diff, its 8 coordinates and its predicted score
  4. one search run (archive of four nodes, the paper's setting)
  5. twelve runs: readout vs HGM vs parent-rate ablation vs random vs root vs calibration-best
  6. what it cost (measured day-simulations, not arithmetic)

Run:  python labs/g01_budget_rsi/lab10_03_run_demo/run.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.edits import DIFF_FEATURE_NAMES, diff_features, propose, unified_diff  # noqa: E402
from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator, score_batch  # noqa: E402
from gridwise.methods.budget_rsi import SELECTORS, calibrate, featurize, search  # noqa: E402
from gridwise.policy import EXPERT_POLICY, SEED_POLICY  # noqa: E402

t0 = banner("10.03", "Run Budget RSI: score edits before they run", "arXiv:2610.29121")
task = make_task("tou", seed=0)

section("1. Task: one day = 24 hourly battery decisions; score = share of the oracle's savings captured")
b = task.steering
print(f"  splits: fit={task.fit.n} steering={task.steering.n} held-out={task.heldout.n} days (paper: 16 / 48 / 100)")
print(f"  steering: no battery ${b.cost_none.mean():.2f}/day, oracle ${b.cost_oracle.mean():.2f}/day")
print("  starting program (what most home batteries ship with):")
print("    " + SEED_POLICY.source().replace("\n", "\n    "))
for name, p in (("seed", SEED_POLICY), ("hand-written expert", EXPERT_POLICY)):
    print(f"  {name:20s} steering {score_batch(p, task.steering)}")

section("2. Calibration: label 96 random edits on the 16-day fit split, fit ONE readout, freeze it")
ev = Evaluator(task)
cal = calibrate(ev, SEED_POLICY, np.random.default_rng(0))
print(f"  labelled edits: {len(cal.y)}, fit-split day-sims spent: {ev.ledger['fit']}")
print(f"  readout R^2 on its own training edits: {cal.readout.r2:.3f}")
print("  standardised coefficients (largest first):")
for n_, w in cal.readout.importance():
    print(f"    {n_:18s} {w:+.4f}")

section("3. One proposal, scored before it runs")
rng = np.random.default_rng(7)
child, info = propose(SEED_POLICY, rng)
while info.operator != "add_rule":
    child, info = propose(SEED_POLICY, rng)
print(unified_diff(SEED_POLICY, child))
x = featurize(ev.score(SEED_POLICY).mean, SEED_POLICY, child)
print("  8 diff coordinates: " + ", ".join(f"{n_}={v:g}" for n_, v in zip(DIFF_FEATURE_NAMES, x[1:])))
print(f"  predicted capture (no simulation): {cal.readout.predict(x)[0]:.3f}")
print(f"  actual steering capture (simulated afterwards, for the demo): {ev.score(child).mean:.3f}")

section("4. One search: archive of four nodes, expand ONE proposal per step")
res = search(ev, SEED_POLICY, "readout", np.random.default_rng(100), readout=cal.readout)
print("\n".join("  " + l for l in res.log))
print(f"  proposals scored by the readout without running: {res.proposals_scored}")
print(f"  returned: node {res.best.order} (earliest with maximal steering solve count {res.best.score.solved}/48)")
print("    " + res.best.policy.source().replace("\n", "\n    "))
print(f"  held-out (reported once, after search): {ev.final_report(res.best.policy)}")

section("5. Twelve runs (paper: twelve runs per method), held-out capture on 100 unseen days")
R = {k: [] for k in ("readout (Budget RSI)", "HGM selection", "parent-rate ablation", "random", "root", "calibration-best")}
cost = {"fit": [], "steering": []}
for run in seeds(12):
    e = Evaluator(task)
    c = calibrate(e, SEED_POLICY, np.random.default_rng(run))
    cost["fit"].append(e.ledger["fit"])
    for sel, key in zip(("readout", "hgm", "parent_rate", "random"), list(R)[:4]):
        before = e.ledger["steering"]
        r = search(e, SEED_POLICY, sel, np.random.default_rng(100 + run), readout=c.readout)
        if sel == "readout":
            cost["steering"].append(e.ledger["steering"] - before)
        R[key].append(e.final_report(r.best.policy).mean)
    R["root"].append(e.final_report(SEED_POLICY).mean)
    R["calibration-best"].append(e.final_report(c.best_child).mean)
paper = {"readout (Budget RSI)": "87.0", "HGM selection": "85.1", "parent-rate ablation": "84.2", "random": "-",
         "root": "81.2", "calibration-best": "82.8"}
hgm = np.array(R["HGM selection"])


def wtl(v):
    d = np.array(v) - hgm
    return f"{int((d > 1e-9).sum())}/{int((abs(d) <= 1e-9).sum())}/{int((d < -1e-9).sum())}"


rows = [[k, ms(v), wtl(v) if k != "HGM selection" else "-", paper[k]] for k, v in R.items()]
table(rows, ["method", "held-out capture (mean +/- sd)", "W/T/L vs HGM (paired runs)", "paper GPT-4o HumanEval+ %"])
print("\n  Same ordering as the paper for readout > HGM > parent-rate > root.")
print("  NOT as in the paper: calibration-best is strongest here, because calibration itself is a")
print(f"  96-candidate search on a low-noise fit split -- and it costs "
      f"{np.mean(cost['fit']) / np.mean(cost['steering']):.1f}x the search (see 6).")

section("6. Measured cost per run (day-simulations)")
pool = 4 * 4                                      # proposals per step = archive nodes x 4
full = 3 * pool * 48                              # evaluate every proposal instead of scoring it
table([["calibration (96 edits x 16 fit days, cached repeats free)", f"{np.mean(cost['fit']):.0f}"],
       ["search with readout (4 nodes x 48 steering days)", f"{np.mean(cost['steering']):.0f}"],
       ["search if every proposal were evaluated (upper bound)", f"<= {full}"]],
      ["phase", "day-sims"])
print("  The readout makes SEARCH cheap. Whether the whole method is cheap depends on amortising")
print("  calibration across runs (lab 10.05 tests this).")
save("10.03", {"heldout": R, "cost": cost, "readout_r2": cal.readout.r2,
               "coefficients": cal.readout.importance(), "example_search": res.log}, t0)
