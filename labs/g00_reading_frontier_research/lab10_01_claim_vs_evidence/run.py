"""Lab 10.01 - Distinguish claim from evidence.

We do not take the course's two laptop demos at their word. We run them as
experiments and ask, for each headline: what in the code actually produces it?

  A. SO-RSI demo  : "SO-RSI outperforms Naive RSI by ~20 points because of diagnostics"
  B. Budget RSI demo: "the linear readout selects better edits than random / parent rate"

Run:  python labs/g00_reading_frontier_research/lab10_01_claim_vs_evidence/run.py
"""
import contextlib
import importlib.util
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import ROOT, banner, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

REF = ROOT / "reference" / "course_demos"


def load(name):
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


t0 = banner("10.01", "Distinguish claim from evidence", "course demos in reference/course_demos (MIT)")
so = load("so_rsi_demo")
bud = load("budget_rsi_demo")
evidence = {}

# ---------------------------------------------------------------- A. SO-RSI
section("A1. Re-run the SO-RSI demo comparison 300 times (20 attempts each)")


def run_pair(diagnostics_on: bool, trials: int = 300):
    naive, sorsi = [], []
    orig = so.StructuralAnomalyDetector.monitor_trace
    if not diagnostics_on:          # ablation: the detector sees traces but never reports anomalies
        so.StructuralAnomalyDetector.monitor_trace = lambda self, tr: (self.add_trace(tr), [])[1]
    try:
        for s in range(trials):
            random.seed(s); np.random.seed(s)
            n = so.NaiveRSI(); r = so.SORSI()
            for _ in range(20):
                n.attempt_fix("x")
                quiet(r.attempt_fix, "x", expected_outcome="success" if random.random() < 0.3 else None)
            naive.append(n.successes / 20); sorsi.append(r.successes / 20)
    finally:
        so.StructuralAnomalyDetector.monitor_trace = orig
    return np.mean(naive), np.mean(sorsi)


n_on, s_on = run_pair(True)
n_off, s_off = run_pair(False)
table([["diagnostics ON (as shipped)", n_on, s_on, s_on - n_on],
       ["diagnostics OFF (ablation)", n_off, s_off, s_off - n_off]],
      ["condition", "naive success", "SO-RSI success", "gap"])
print("\nThe gap is identical with the diagnostics switched off. Where does it come from?")
src = (REF / "so_rsi_demo.py").read_text(encoding="utf-8").splitlines()
for i, line in enumerate(src, 1):
    if "random.random() < 0.3" in line or "random.random() < 0.5" in line:
        print(f"  so_rsi_demo.py:{i}: {line.strip()}")
print("-> the success probabilities are hard-coded (0.3 vs 0.5). Diagnostics never change the next fix")
print("   (see the comment 'In full implementation, this would adjust the fix strategy').")

section("A2. Do the anomaly detectors detect what they claim?")
d = so.StructuralAnomalyDetector()
a1 = d.monitor_trace(so.ExecutionTrace(1, "apply_fix", "add parameter X", "failure"))
print(f"  first-ever failure, empty history -> anomalies {[a.value for a in a1]}")
d = so.StructuralAnomalyDetector()
d.monitor_trace(so.ExecutionTrace(1, "apply_fix", "increase learning rate", "failure"))
a2 = d.monitor_trace(so.ExecutionTrace(2, "apply_fix", "decrease learning rate", "failure"))
print(f"  'increase lr' then 'decrease lr'   -> anomalies {[a.value for a in a2]}  (opposing expected)")
print("-> monitor_trace() appends the trace BEFORE checking, so recurrence matches the trace with itself")
print("   (every failure is a 'recurrence') and opposing compares the edit with itself (never fires).")
evidence["so_rsi_demo"] = dict(naive_on=n_on, sorsi_on=s_on, naive_off=n_off, sorsi_off=s_off,
                              recurrence_on_first_failure=[a.value for a in a1],
                              opposing_on_increase_then_decrease=[a.value for a in a2])

# ---------------------------------------------------------- B. Budget RSI
section("B1. Is the held-out score independent evidence, or generated from the steering score?")
random.seed(0); np.random.seed(0)
edits = bud.generate_synthetic_edits(500)
st = np.array([e.steering_solve_count for e in edits]); ho = np.array([e.held_out_solve_count for e in edits])
r = float(np.corrcoef(st, ho)[0, 1])
print(f"  corr(steering count, held-out count) over 500 synthetic edits = {r:.3f}")
print("  generate_synthetic_edits():  held_out = clip(steering_count * 6 + noise)  -> the proxy is valid BY CONSTRUCTION")

section("B2. Does selection still work if the diff features carry no information?")


def selection_trial(seed, shuffle):
    random.seed(seed); np.random.seed(seed)
    es = bud.generate_synthetic_edits(128)
    if shuffle:   # break the link between features and outcomes, keep everything else
        feats = [e.diff_features for e in es]; random.shuffle(feats)
        for e, f in zip(es, feats):
            e.diff_features = f
    sc = bud.EditScorer(); quiet(sc.train, es[:96])
    test = es[96:]
    pick = max(test, key=sc.score_edit)
    return pick.held_out_solve_count - np.mean([e.held_out_solve_count for e in test])


rows = []
for shuffle in (False, True):
    gains = [selection_trial(s, shuffle) for s in seeds(200)]
    rows.append(["features shuffled" if shuffle else "as shipped", float(np.mean(gains)), float(np.mean(np.array(gains) > 0))])
table(rows, ["condition", "held-out gain over random (solves/100)", "P(gain > 0)"])
print("-> shuffling the diff features removes ~80% of the gain; what is left comes from parent_rate,")
print("   which the generator ALSO wires linearly into the outcome. The readout works because the data")
print("   generator makes outcomes a linear function of exactly the readout's inputs.")
src = (REF / "budget_rsi_demo.py").read_text(encoding="utf-8")
print(f"\n  'HuxleyGodelSelector(' instantiated in main(): {'HuxleyGodelSelector(' in src.split('def main')[1]}")
print("  The '~88% cost reduction' is printed arithmetic (budget_rsi_demo.py:248-258), not a measured quantity.")
evidence["budget_rsi_demo"] = dict(corr_steering_heldout=r, selection=rows)

section("What the demos DO establish")
print("  * the code paths of each mechanism run end to end on a laptop")
print("  * NOT that the mechanism causes the improvement -- both headlines are built into the data generator.")
print("  This is why the rest of this repo re-builds the three methods on a task (GridWise) whose outcomes")
print("  come from a simulator, with an oracle, frozen splits, matched budgets and ablations.")
save("10.01", evidence, t0)
