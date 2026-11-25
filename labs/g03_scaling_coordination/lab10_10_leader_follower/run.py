"""Lab 10.10 - Simulate AReaL-RSI leader-follower coordination (arXiv:2610.18083).

1 leader + 2 researchers. The leader reads the conditional memory (which
problems are still unresolved, which orientations produced supported findings)
and assigns research orientations each campaign. Researchers run short,
orientation-biased searches. After each campaign an evidence-based review
re-weights orientations.

Comparison at the same budget:
  conditional memory   : build only on SUPPORTED findings (the paper's mechanism)
  unconditional sharing: pass on whatever a researcher returned, validated or not
  independent          : no sharing, best researcher at the end (the paper's baseline)

Run:  python labs/g03_scaling_coordination/lab10_10_leader_follower/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.areal_rsi import ORIENT, run_independent, run_team  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.10", "Leader-follower coordination with conditional memory", "AReaL-RSI, arXiv:2610.18083")
BUDGET = 48 * 60
task = make_task("tou", seed=0)

section("1. Orientations the leader can assign")
for o, d in ORIENT.items():
    print(f"  {o:10s} templates={list(d['templates'])}  targets symptoms={list(d['symptoms'])}")

section("2. One team run (TOU, seed 0): campaign log")
ev = Evaluator(task)
best, best_s, mem, log = run_team(ev, SEED_POLICY, np.random.default_rng(0), BUDGET)
print("\n".join("  " + l for l in log))
section("3. Conditional memory after the run")
table([[e["campaign"], e["orientation"], e["problem"], e["status"], f"{e['evidence']:+.4f}"] for e in mem.entries],
      ["campaign", "orientation", "problem", "status", "evidence (dCapture)"])
print(f"\n  supported findings: {len(mem.supported())}, still-unresolved problems: "
      f"{sorted({e['problem'] for e in mem.unresolved()})}")
print("  best program found:")
print("    " + best.source().replace("\n", "\n    "))
print(f"  held-out {ev.final_report(best)}")

section("4. Does conditional sharing beat independent exploration? (8 seeds, same budget)")
rows, data = [], {}
for tariff in ("tou", "critical_peak", "wholesale"):
    t = make_task(tariff, 0)
    res = {"conditional memory": [], "unconditional sharing": [], "independent": []}
    for seed in seeds(8):
        e = Evaluator(t); p, *_ = run_team(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["conditional memory"].append(e.final_report(p).mean)
        e = Evaluator(t); p, *_ = run_team(e, SEED_POLICY, np.random.default_rng(seed), BUDGET, conditional=False)
        res["unconditional sharing"].append(e.final_report(p).mean)
        e = Evaluator(t); p, _ = run_independent(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["independent"].append(e.final_report(p).mean)
    ind = np.mean(res["independent"])
    for k, v in res.items():
        rows.append([tariff, k, ms(v), f"{(np.mean(v) - ind) / ind:+.1%}"])
    data[tariff] = res
table(rows, ["tariff", "coordination", "held-out capture", "vs independent"])
share_wins = sum(max(np.mean(d["conditional memory"]), np.mean(d["unconditional sharing"])) > np.mean(d["independent"])
                 for d in data.values())
cond_wins = sum(np.mean(d["conditional memory"]) > np.mean(d["unconditional sharing"]) for d in data.values())
print("\n  Paper: experience sharing beats independent exploration by 1.9-27.5%.")
print(f"  Here: some form of sharing beats independent researchers on {share_wins}/{len(data)} tariffs;")
print(f"  conditional memory beats unconditional sharing on {cond_wins}/{len(data)} -- so the benefit of")
print("  SHARING replicates, while the benefit of the resolved/unresolved FILTER is task-dependent.")
print("  Seed spread (+/-) is comparable to several gaps: treat single-tariff differences with caution.")
save("10.10", {"example_log": log, "memory": [{k: v for k, v in e.items() if k != "policy"} for e in mem.entries],
               "comparison": data}, t0)
