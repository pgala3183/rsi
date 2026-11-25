"""Lab 10.05 - Extend the scorer to our own edits.

Three extensions, each with a measured answer:
  A. Domain coordinates: add 6 task-aware features read from the diff text
     (does it touch discharge logic? the soc guard? solar logic? which priority?)
  B. Transfer: freeze a readout calibrated on the TOU tariff and use it, unchanged,
     to steer search on two other tariffs.
  C. Matched cost: is Budget RSI cheaper than simply searching more? Compare at equal
     day-simulation budgets, including the calibration bill.

Run:  python labs/g01_budget_rsi/lab10_05_extend/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.edits import DOMAIN_FEATURE_NAMES, domain_features, propose  # noqa: E402
from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.budget_rsi import calibrate, featurize, search  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.05", "Extend Budget RSI to task-aware edits, new tariffs and matched cost", "arXiv:2610.29121")
task = make_task("tou", seed=0)
out = {}

section("A. Domain coordinates (computed from the diff text only, still no simulation)")
print("  new coordinates: " + ", ".join(DOMAIN_FEATURE_NAMES))
rng = np.random.default_rng(3)
child, _ = propose(SEED_POLICY, rng)
print("  example diff -> " + ", ".join(f"{n}={v:g}" for n, v in zip(DOMAIN_FEATURE_NAMES, domain_features(SEED_POLICY, child))))
A = {}
for fs in ("diff", "diff+domain"):
    hs, r2 = [], []
    for run in seeds(12):
        ev = Evaluator(task)
        c = calibrate(ev, SEED_POLICY, np.random.default_rng(run), feature_set=fs)
        r = search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=c.readout, feature_set=fs)
        hs.append(ev.final_report(r.best.policy).mean); r2.append(c.readout.r2)
    A[fs] = dict(heldout=hs, r2=float(np.mean(r2)), coef=c.readout.importance()[:5])
table([[fs, f"{v['r2']:.3f}", ms(v["heldout"])] for fs, v in A.items()], ["features", "readout R^2", "held-out capture"])
print("  top coefficients with domain features: " + ", ".join(f"{n} {w:+.2f}" for n, w in A["diff+domain"]["coef"]))
out["domain"] = A

section("B. Transfer: a readout frozen on TOU, used on other tariffs (no recalibration)")
B = []
for target in ("critical_peak", "solar_glut", "wholesale"):
    tt = make_task(target, seed=0)
    res = {"frozen TOU readout": [], "readout calibrated on target": [], "HGM (no readout)": [], "root": []}
    for run in seeds(12):
        src = calibrate(Evaluator(task), SEED_POLICY, np.random.default_rng(run))
        ev = Evaluator(tt)
        own = calibrate(ev, SEED_POLICY, np.random.default_rng(run))
        for name, sel, ro in (("frozen TOU readout", "readout", src.readout),
                              ("readout calibrated on target", "readout", own.readout),
                              ("HGM (no readout)", "hgm", None)):
            r = search(ev, SEED_POLICY, sel, np.random.default_rng(100 + run), readout=ro)
            res[name].append(ev.final_report(r.best.policy).mean)
        res["root"].append(ev.final_report(SEED_POLICY).mean)
    for k, v in res.items():
        B.append([target, k, ms(v)])
table(B, ["target tariff", "selector", "held-out capture"])
out["transfer"] = B

section("C. Matched cost: Budget RSI vs just searching more (day-simulations per run, calibration included)")
C = {}
amort = calibrate(Evaluator(task), SEED_POLICY, np.random.default_rng(12345))      # calibrated ONCE, reused by every run
for run in seeds(12):
    ev = Evaluator(task)
    c = calibrate(ev, SEED_POLICY, np.random.default_rng(run))
    cal_cost = ev.day_sims
    r = search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=c.readout)
    C.setdefault("Budget RSI, calibrated per run", []).append((ev.day_sims, ev.final_report(r.best.policy).mean))
    C.setdefault("calibration-best edit (no search)", []).append((cal_cost, ev.final_report(c.best_child).mean))
    nodes = 1 + int(round(ev.day_sims / 48))              # same total budget spent on plain HGM search
    e2 = Evaluator(task)
    r2 = search(e2, SEED_POLICY, "hgm", np.random.default_rng(100 + run), archive_size=nodes)
    C.setdefault("HGM search, same total budget (~30 nodes)", []).append((e2.day_sims, e2.final_report(r2.best.policy).mean))
    e3 = Evaluator(task)
    r3 = search(e3, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=amort.readout)
    C.setdefault("Budget RSI, readout amortised over 12 runs", []).append(
        (e3.day_sims + 1250 / 12, e3.final_report(r3.best.policy).mean))
    e4 = Evaluator(task)
    r4 = search(e4, SEED_POLICY, "hgm", np.random.default_rng(100 + run), archive_size=1 + int(round((e3.day_sims + 104) / 48)))
    C.setdefault("HGM search, amortised budget", []).append((e4.day_sims, e4.final_report(r4.best.policy).mean))
table([[k, f"{np.mean([c for c, _ in v]):.0f}", ms([h for _, h in v])] for k, v in C.items()],
      ["method", "day-sims / run", "held-out capture"])
print("\n  Honest bottom line for the Budget RSI group on GridWise:")
print("  * at the paper's tiny search budget (4 nodes) the readout beats HGM / parent-rate / random (lab 10.03)")
print("  * counting the calibration bill, spending the same day-sims on plain HGM search is as good or")
print("    better; amortising one calibration over many runs closes the cost gap without overtaking HGM")
print("  * task-aware diff coordinates (A) and transfer across tariffs (B) are where the readout earns its keep")
out["matched_cost"] = {k: v for k, v in C.items()}
save("10.05", out, t0)
