"""Lab 10.04 - Analyse the scoring mechanism.

Questions
  1. Which coefficients matter, and are they stable across calibrations?
  2. Does the readout actually rank FRESH edits (not the ones it was fit on)?
  3. Ablation: parent-rate only vs diff-only vs both.
  4. How many labelled edits does the readout need? (24 / 48 / 96 / 192)
  5. What if we keep only 4 of the 8 diff coordinates?

Run:  python labs/g01_budget_rsi/lab10_04_analyze_mechanism/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.edits import propose  # noqa: E402
from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.budget_rsi import LinearReadout, calibrate, feature_names, featurize, search  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.04", "Analyse the linear readout", "arXiv:2610.29121")
task = make_task("tou", seed=0)
NAMES = feature_names("diff")


class ColumnReadout:
    """A readout that only sees some input columns (for ablations)."""

    def __init__(self, cols, X, y):
        self.cols = list(cols)
        self.inner = LinearReadout([NAMES[c] for c in cols]).fit(X[:, self.cols], y)

    def predict(self, X):
        return self.inner.predict(np.atleast_2d(X)[:, self.cols])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


section("1. Coefficients across 12 independent calibrations (standardised, so comparable)")
W, cals = [], []
for run in seeds(12):
    ev = Evaluator(task)
    c = calibrate(ev, SEED_POLICY, np.random.default_rng(run))
    W.append(c.readout.w); cals.append(c)
W = np.array(W)
order = np.argsort(-np.abs(W.mean(0)))
table([[NAMES[i], f"{W[:, i].mean():+.3f}", f"{W[:, i].std():.3f}", f"{np.mean(np.sign(W[:, i]) == np.sign(W[:, i].mean())):.2f}"]
       for i in order], ["feature", "mean coef", "sd", "sign agreement"])
X0 = cals[0].X
print(f"\n  correlated inputs: corr(lines_added, rules_added) = {np.corrcoef(X0[:, 1], X0[:, 3])[0, 1]:.2f}, "
      f"corr(lines_added, constants_changed) = {np.corrcoef(X0[:, 1], X0[:, 6])[0, 1]:.2f}")
print("  -> correlated coordinates share credit, so single coefficients are unstable even when the")
print("     combined prediction is not; read the table as 'edit size and kind', not as causal effects.")

section("2. Does it rank FRESH edits? (200 new edits from the archive programs of calibration 0)")
ev = Evaluator(task)
rng = np.random.default_rng(999)
parents = [SEED_POLICY] + [p for p in cals[0].children[:30]]
Xf, yf = [], []
for _ in range(200):
    par = parents[int(rng.integers(len(parents)))]
    ch, _ = propose(par, rng)
    Xf.append(featurize(ev.score(par, "fit").mean, par, ch)); yf.append(ev.score(ch, "fit").mean)
Xf, yf = np.array(Xf), np.array(yf)
pred = cals[0].readout.predict(Xf)
rho = spearman(pred, yf)
top = np.argsort(-pred)[:20]
print(f"  Spearman(predicted, actual) on fresh edits = {rho:.3f}")
print(f"  mean actual capture: all fresh edits {yf.mean():.3f} | readout's top-20 {yf[top].mean():.3f}")

section("3. Ablation: what does each input group contribute to SELECTION? (held-out, 12 runs)")
groups = {"parent rate only": [0], "diff coordinates only": list(range(1, 9)), "both (paper)": list(range(9))}
abl = {}
for g, cols in groups.items():
    hs = []
    for run, c in zip(seeds(12), cals):
        ev = Evaluator(task)
        ro = ColumnReadout(cols, c.X, c.y)
        r = search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=ro)
        hs.append(ev.final_report(r.best.policy).mean)
    abl[g] = hs
table([[g, ms(v)] for g, v in abl.items()], ["readout inputs", "held-out capture"])

section("4. How many labelled edits does the readout need?")
size_rows = {}
for n in (24, 48, 96, 192):
    rhos, hs, fitcost = [], [], []
    for run in seeds(8):
        ev = Evaluator(task)
        c = calibrate(ev, SEED_POLICY, np.random.default_rng(run), n_edits=n)
        fitcost.append(ev.ledger["fit"])
        rhos.append(spearman(c.readout.predict(Xf), yf))
        r = search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=c.readout)
        hs.append(ev.final_report(r.best.policy).mean)
    size_rows[n] = dict(spearman=float(np.mean(rhos)), heldout=hs, fit_cost=float(np.mean(fitcost)))
table([[n, f"{v['spearman']:.3f}", ms(v["heldout"]), f"{v['fit_cost']:.0f}"] for n, v in size_rows.items()],
      ["labelled edits", "fresh-edit Spearman", "held-out capture", "calibration day-sims"])

section("5. Keep only 4 of the 8 diff coordinates")
four = [0, 1, 2, 3, 6]      # parent_rate, lines_added, lines_deleted, rules_added, constants_changed
hs4 = []
for run, c in zip(seeds(12), cals):
    ev = Evaluator(task)
    r = search(ev, SEED_POLICY, "readout", np.random.default_rng(100 + run), readout=ColumnReadout(four, c.X, c.y))
    hs4.append(ev.final_report(r.best.policy).mean)
table([["8 coordinates (paper)", ms(abl["both (paper)"])], ["4 coordinates", ms(hs4)]], ["readout", "held-out capture"])
save("10.04", {"coef_mean": dict(zip(NAMES, W.mean(0))), "coef_sd": dict(zip(NAMES, W.std(0))),
               "fresh_spearman": rho, "ablation": abl, "training_size": size_rows, "four_features": hs4}, t0)
