"""Lab 10.08 - Apply SO-RSI to a debugging task.

Scenario. A unit test guards the most valuable behaviour of the controller:

    test_spike_dump: on price-spike days the battery must export >= 3 kWh during
                     the hours when the export price is actually high.

It fails. A developer tries five plausible fixes to the POLICY. We record each
attempt (what changed, what was expected, what happened), then run the SO-RSI
anomaly detector and probes over that debugging trace.

The real cause is planted outside the policy: the feature pipeline reads
hour-ending timestamps as hour-beginning, so every price feature the policy sees
is one hour ahead of the simulator's clock. (A classic off-by-one / timezone bug.)

Run:  python labs/g02_second_order_diagnostics/lab10_08_apply_debugging/run.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, save, section, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import ACTION_ID, Obs, _static_features, make_task, simulate  # noqa: E402
from gridwise.methods.so_rsi import Attempt, detect  # noqa: E402
from gridwise.policy import EXPERT_POLICY, C, Policy, R  # noqa: E402

t0 = banner("10.08", "Apply second-order diagnosis to a failing test")
from gridwise.env import make_days  # noqa: E402
_fixture = make_days(300, "tou", seed=42)                 # a dedicated test fixture, not an experiment split
clean = _fixture.subset(np.flatnonzero(_fixture.tags["spike"]))
NEED = 6.0                                                # kWh: about half the battery dumped into the spike


def buggy_pipeline(b):
    """BUG: price-derived features are taken from the NEXT hour's row."""
    feats = {k: v.copy() for k, v in b.feats.items()}
    for k in ("price", "export_price", "price_rank", "peak_ahead"):
        feats[k][:, :-1] = b.feats[k][:, 1:]
    return replace(b, feats=feats)


def spike_energy(policy, b):
    sim = simulate(b, policy.act, record=True)
    hot = b.export >= 0.5                                         # truly high export hours
    exported = np.maximum(-(b.load - b.solar + sim.power), 0) * hot
    days = b.tags["spike"]
    return float(exported[days].sum(axis=1).mean()), sim


def test_spike_dump(policy, b):
    e, _ = spike_energy(policy, b)
    return e >= NEED, e


b = buggy_pipeline(clean)
section("1. The failing test")
ok, e = test_spike_dump(EXPERT_POLICY, b)
print(f"  test_spike_dump(EXPERT_POLICY): {'PASS' if ok else 'FAIL'} -- {e:.2f} kWh exported in spike hours "
      f"(need >= {NEED}), fixture of {clean.n} spike days")

section("2. The developer's first-order debugging session (recorded)")


def with_spike_thr(p, v):
    rules = list(p.rules); rules[0] = R("discharge_max", C("export_price", ">=", v))
    return p.with_rules(rules)


p0 = EXPERT_POLICY
p_extra = p0.with_rules((R("discharge_max", C("price", ">=", 1.0)),) + p0.rules)
session = [  # (what changed, expectation, program, slot, direction, kind, expected sign of energy change)
    ("lower spike threshold 0.50 -> 0.30", "more spike energy", with_spike_thr(p0, 0.30), "spike", -1, "tweak", +1),
    ("lower spike threshold 0.30 -> 0.20", "more spike energy", with_spike_thr(p0, 0.20), "spike", -1, "tweak", +1),
    ("raise it back to 0.50 (undo)", "fewer false dumps", p0, "spike", +1, "tweak", 0),
    ("add 'discharge_max if price >= 1.0'", "catch spikes by import price", p_extra, "spike", -1, "create", +1),
    ("delete that rule again", "back to baseline", p0, "spike", +1, "tweak", 0),
]
trace, rows = [], []
prev_e = e
for i, (what, expect, pol, slot, d, kind, sign) in enumerate(session):
    ok_i, e_i = test_spike_dump(pol, b)
    trace.append(Attempt(i, "uncovered_peak", slot, d, kind, ok_i, e_i - prev_e, -(e_i - prev_e), 0))
    found = detect(trace)
    if sign > 0 and e_i - prev_e <= 1e-9 and "mismatch" not in found:
        found.append("mismatch")          # expected more spike energy, observed none
    rows.append([i + 1, what, expect, f"{'PASS' if ok_i else 'FAIL'} ({e_i:.2f} kWh)", ", ".join(found) or "-"])
table(rows, ["#", "what changed", "expected", "what happened", "anomalies flagged"])
print("\n  Five patches, zero progress. Retrospectively the trace shows recurrence (the same lever pulled")
print("  repeatedly), opposing edits (raise / lower, add / delete) and an outcome that never moves as expected.")

section("3. Probes")
print("  recurrence -> sweep the lever itself:")
sweep = [(v, test_spike_dump(with_spike_thr(p0, v), b)[1]) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
print("    " + "  ".join(f"thr={v:.1f}: {x:.2f} kWh" for v, x in sweep))
best_sweep = max(x for _, x in sweep)
if best_sweep < NEED:
    print(f"    best {best_sweep:.2f} kWh < {NEED} at every threshold -> the threshold is NOT the binding lever")
else:
    print(f"    some threshold passes ({best_sweep:.2f} kWh) -> the threshold IS a lever")

print("  mismatch -> expected the dump rule to fire in spike hours. WHEN does it fire?")
_, sim = spike_energy(p0, b)
days = np.flatnonzero(clean.tags["spike"])
offsets = []
for d in days:
    fired = np.flatnonzero(sim.actions[d] == ACTION_ID["discharge_max"])
    hot = np.flatnonzero(clean.export[d] >= 0.5)
    if len(fired) and len(hot):
        offsets.append(int(fired.min() - hot.min()))
vals, counts = np.unique(offsets, return_counts=True)
print(f"    first dump hour minus first truly-hot hour, per spike day: {dict(zip(vals.tolist(), counts.tolist()))}")
print("    -> the rule fires exactly ONE HOUR EARLY on every spike day: the policy is reacting correctly")
print("       to observations that are wrong. The assumption 'features describe the current hour' is broken.")

print("  confirm -> cross-correlate the export_price feature with the raw export tariff at lags -2..+2:")
lag_err = {}
for lag in range(-2, 3):
    a = b.feats["export_price"][:, max(0, -lag):24 - max(0, lag)]
    r = clean.export[:, max(0, lag):24 - max(0, -lag)]
    lag_err[lag] = float(np.abs(a - r).mean())
print("    " + "  ".join(f"lag {k:+d}: err {v:.3f}" for k, v in lag_err.items()))
best_lag = min(lag_err, key=lag_err.get)
print(f"    feature[t] == raw[t{best_lag:+d}]  -> root cause is in the FEATURE PIPELINE, not in the policy")

section("4. Fix the pipeline, keep the original policy")
fixed = replace(b, feats=_static_features(b))
ok_f, e_f = test_spike_dump(EXPERT_POLICY, fixed)
print(f"  test_spike_dump(EXPERT_POLICY) after rebuilding features: {'PASS' if ok_f else 'FAIL'} ({e_f:.2f} kWh)")
from gridwise.env import capture  # noqa: E402
cb = capture(b, simulate(b, EXPERT_POLICY.act).cost).mean()
cf = capture(fixed, simulate(fixed, EXPERT_POLICY.act).cost).mean()
print(f"  steering capture of the unchanged policy: buggy pipeline {cb:.3f} -> fixed pipeline {cf:.3f}")
print("\n  Lesson: five first-order patches could never pass this test. The second-order questions")
print("  ('is this the lever?', 'did it do what I expected, WHEN I expected?') moved the search from")
print("  the policy to the environment model -- the course's 'test setup wrong, not function'.")
save("10.08", {"session": rows, "sweep": sweep, "offsets": dict(zip(vals.tolist(), counts.tolist())),
               "lag_error": lag_err, "fixed_pass": ok_f, "capture_buggy": float(cb), "capture_fixed": float(cf)}, t0)
