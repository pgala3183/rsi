"""Lab 10.14 - Simulate Self-HPO harness-policy alternation (arXiv:2610.22791).

Harness = training curriculum + runtime guards (action filters around the policy).
H-step: the policy analyses ITS OWN failures on validation days and proposes
        guards aimed at its top symptoms; its hardest training days become the
        next curriculum (hard-task identification).
P-step: the policy is hill-climbed on the curriculum, running inside the harness.

Three arms at the same budget: policy-only, harness-only, joint (alternating).
The paper's synergy claim: joint gain > policy gain + harness gain.

Run:  python labs/g04_adaptive_workflows/lab10_14_self_hpo/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.self_hpo import run_self_hpo  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.14", "Self-HPO: alternate harness and policy optimisation", "arXiv:2610.22791")
BUDGET = 48 * 60
task = make_task("tou", seed=0)

section("1. One joint run (TOU, seed 0): the alternation log")
ev = Evaluator(task)
st = run_self_hpo(ev, SEED_POLICY, np.random.default_rng(0), BUDGET, mode="joint")
print("\n".join("  " + l for l in st.log[:12]))
print(f"  ... {len(st.log)} steps")
print("  final policy (runs inside the harness):")
print("    " + st.policy.source().replace("\n", "\n    "))
print(f"  final harness: {st.harness.describe()}")
for g in st.harness.guards:
    print(f"    guard {g.name}: if {' and '.join(f'{f} {op} {t}' for f, op, t in g.conds)} -> override")
print(f"  held-out (policy + harness) {ev.final_report(st.policy, wrapper=st.harness)}")

section("2. Policy-only vs harness-only vs joint (8 seeds, same budget)")
rows, data = [], {}
for tariff in ("tou", "critical_peak", "wholesale", "vpp_events"):
    t = make_task(tariff, 0)
    root = Evaluator(t).final_report(SEED_POLICY).mean
    res = {"policy_only": [], "harness_only": [], "joint": []}
    for seed in seeds(8):
        for mode in res:
            e = Evaluator(t)
            s = run_self_hpo(e, SEED_POLICY, np.random.default_rng(seed), BUDGET, mode=mode)
            res[mode].append(e.final_report(s.policy, wrapper=s.harness).mean)
    gp, gh, gj = (np.mean(res[m]) - root for m in ("policy_only", "harness_only", "joint"))
    rows.append([tariff, f"{root:.3f}", ms(res["policy_only"]), ms(res["harness_only"]), ms(res["joint"]),
                 f"{gj - (gp + gh):+.3f}"])
    data[tariff] = dict(root=root, **res, synergy=gj - (gp + gh))
table(rows, ["tariff", "root", "policy-only", "harness-only", "joint", "synergy = joint gain - (P + H gains)"])
pos = sum(d["synergy"] > 0 for d in data.values())
best_joint = sum(np.mean(d["joint"]) >= max(np.mean(d["policy_only"]), np.mean(d["harness_only"])) for d in data.values())
print(f"\n  Paper: 57.8% -> 64.8% resolve rate; joint gain > sum of individual gains.")
print(f"  Here: synergy > 0 on {pos}/{len(data)} tariffs; joint is the best arm on {best_joint}/{len(data)}.")
print("  harness-only gain per tariff: " + ", ".join(f"{k} {np.mean(d['harness_only']) - d['root']:+.3f}" for k, d in data.items()))
print("  Where one guard alone already fixes the main failure (e.g. a VPP event rule), joint has nothing")
print("  left to add and the additive 'synergy' baseline double-counts the same fix -> large negative synergy.")
print("  Guards are chosen on only 16 validation days, so they can also overfit: judge by the held-out columns.")
save("10.14", {"example_log": st.log, "harness": st.harness.describe(), "comparison": data}, t0)
