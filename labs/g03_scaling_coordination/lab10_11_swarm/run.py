"""Lab 10.11 - Simulate SwarmRSI swarm evolution (arXiv:2610.20431).

Five agents, each a (hypothesis, implementation) pair. One run is printed in
full (recombination events, hypothesis restarts, best score and swarm diversity
per generation). Then three experiments:

  A. ablations at a fixed budget: full swarm vs no recombination / no feedback /
     no hypothesis evolution, and a single-agent hill climber with the same budget
  B. budget curve: does the swarm's advantage depend on how much it may spend?
  C. experience accumulation (the paper's claim): campaign 2 starts with the archive
     from campaign 1 -- with a control that gives a single agent the same retained program

Run:  python labs/g03_scaling_coordination/lab10_11_swarm/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.swarm_rsi import ORIENTATIONS, run_single, run_swarm  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.11", "Swarm evolution: mutation, recombination, recursive evolution", "SwarmRSI, arXiv:2610.20431")
BUDGET = 48 * 150
task = make_task("tou", seed=0)
out = {}

section("1. Initial research directions (hypotheses)")
for k, v in ORIENTATIONS.items():
    print(f"  {k:13s} prefers {list(v) if k != 'generalist' else 'all templates'}")

section("2. One swarm run (TOU, seed 0)")
ev = Evaluator(task)
r = run_swarm(ev, SEED_POLICY, np.random.default_rng(0), BUDGET)
print("\n".join("  " + e for e in r.events))
print("\n  generation curve (day-sims, best steering capture, swarm diversity = mean rule-set Jaccard distance):")
for c in r.curve:
    print(f"    {c[0]:5d}  {c[1]:.3f}  {c[2]:.2f}")
print(f"\n  retained experience: {len(r.archive)} successful edits; by agent: "
      + ", ".join(f"{a.name}={sum(1 for x in r.archive if x['agent'] == a.name)}" for a in r.agents))
print(f"  best agent: {r.best.name}; program:")
print("    " + r.best.policy.source().replace("\n", "\n    "))
print(f"  held-out {ev.final_report(r.best.policy)}")
out["example"] = dict(events=r.events, curve=r.curve, best=r.best.policy.source())

section("A. Ablations at a fixed budget (10 seeds; paired against the single-agent baseline)")
rows, A = [], {}
for tariff in ("tou", "critical_peak", "wholesale"):
    t = make_task(tariff, 0)
    res = {k: [] for k in ("SwarmRSI (full)", "no recombination", "no feedback board", "no hypothesis evolution", "single agent")}
    for seed in seeds(10):
        for k, kw in (("SwarmRSI (full)", {}), ("no recombination", dict(recombination=False)),
                      ("no feedback board", dict(feedback=False)), ("no hypothesis evolution", dict(evolution=False))):
            e = Evaluator(t); rr = run_swarm(e, SEED_POLICY, np.random.default_rng(seed), BUDGET, **kw)
            res[k].append(e.final_report(rr.best.policy).mean)
        e = Evaluator(t); p, _ = run_single(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["single agent"].append(e.final_report(p).mean)
    single = np.array(res["single agent"])
    for k, v in res.items():
        d = np.array(v) - single
        rows.append([tariff, k, ms(v), "-" if k == "single agent" else f"{d.mean():+.3f} ({int((d > 1e-9).sum())}/{len(d)})"])
    A[tariff] = res
table(rows, ["tariff", "variant", "held-out capture", "minus single (wins)"])
out["ablations"] = A

section("B. Budget curve (TOU and critical_peak, 10 seeds)")
rows, Bc = [], {}
for tariff in ("tou", "critical_peak"):
    t = make_task(tariff, 0)
    for b in (48 * 30, 48 * 60, 48 * 150):
        sw, si = [], []
        for seed in seeds(10):
            e = Evaluator(t); rr = run_swarm(e, SEED_POLICY, np.random.default_rng(seed), b); sw.append(e.final_report(rr.best.policy).mean)
            e = Evaluator(t); p, _ = run_single(e, SEED_POLICY, np.random.default_rng(seed), b); si.append(e.final_report(p).mean)
        d = np.array(sw) - np.array(si)
        rows.append([tariff, b, ms(sw), ms(si), f"{d.mean():+.3f}"])
        Bc[f"{tariff}_{b}"] = dict(swarm=sw, single=si)
table(rows, ["tariff", "day-sims", "swarm", "single agent", "swarm - single"])
out["budget_curve"] = Bc

section("C. Experience accumulation: campaign 1 (TOU, 7200 day-sims) -> campaign 2 (1440 day-sims)")
src = make_task("tou", 0)
e = Evaluator(src); r1 = run_swarm(e, SEED_POLICY, np.random.default_rng(0), BUDGET)
experience = r1.archive
retained = experience[-1]["policy"]
rows, C = [], {}
for name, tgt in (("TOU, new days", make_task("tou", 7)), ("critical_peak (new tariff)", make_task("critical_peak", 7)),
                  ("solar_glut (new tariff)", make_task("solar_glut", 7))):
    res = {"swarm, fresh": [], "swarm + experience": [], "single agent + retained program": []}
    for seed in seeds(10):
        e = Evaluator(tgt); rr = run_swarm(e, SEED_POLICY, np.random.default_rng(seed), 48 * 30)
        res["swarm, fresh"].append(e.final_report(rr.best.policy).mean)
        e = Evaluator(tgt); rr = run_swarm(e, SEED_POLICY, np.random.default_rng(seed), 48 * 30, experience=experience)
        res["swarm + experience"].append(e.final_report(rr.best.policy).mean)
        e = Evaluator(tgt); p, _ = run_single(e, retained, np.random.default_rng(seed), 48 * 30)
        res["single agent + retained program"].append(e.final_report(p).mean)
    for k, v in res.items():
        rows.append([name, k, ms(v)])
    C[name] = res
table(rows, ["campaign 2 task", "start", "held-out capture"])
print("\n  The control row matters: if 'single agent + retained program' matches 'swarm + experience',")
print("  the benefit is transfer of the retained PROGRAM, not of the swarm's accumulated hypotheses.")
out["experience"] = C
save("10.11", out, t0)
