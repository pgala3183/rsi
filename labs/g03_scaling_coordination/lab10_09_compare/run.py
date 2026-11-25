"""Lab 10.09 - Compare scaling approaches: AReaL-RSI (leader-follower) vs SwarmRSI (swarm).

Same task, same proposer, same day-simulation budget. We fill the course's
comparison table with what each mechanism does in code AND what it achieved.

Run:  python labs/g03_scaling_coordination/lab10_09_compare/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.areal_rsi import run_independent, run_team  # noqa: E402
from gridwise.methods.swarm_rsi import diversity, run_single, run_swarm  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.09", "Compare scaling approaches at a matched budget", "AReaL-RSI 2610.18083 vs SwarmRSI 2610.20431")
BUDGET = 48 * 100

section("1. Mechanism comparison (from the code in gridwise/methods/)")
table([
    ["coordination", "leader assigns orientations to 2 researchers", "5 peers, no leader"],
    ["memory", "conditional: supported vs unresolved", "shared feedback board + archive of successful edits"],
    ["evolution", "evidence-based review re-weights orientations", "mutation + complementary recombination + hypothesis restarts"],
    ["selection", "leader keeps only validated improvements", "each agent keeps its own best; weakest replaced by recombinants"],
    ["diversity", "low: researchers restart from the shared best", "high early, measured each generation"],
], ["aspect", "AReaL-RSI (areal_rsi.py)", "SwarmRSI (swarm_rsi.py)"])

section("2. Matched-budget results (8 seeds, held-out capture)")
rows, data = [], {}
for tariff in ("tou", "critical_peak", "wholesale"):
    t = make_task(tariff, 0)
    res = {"AReaL-RSI team": [], "SwarmRSI": [], "independent researchers": [], "single agent": []}
    div = []
    for seed in seeds(8):
        e = Evaluator(t); p, *_ = run_team(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["AReaL-RSI team"].append(e.final_report(p).mean)
        e = Evaluator(t); r = run_swarm(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["SwarmRSI"].append(e.final_report(r.best.policy).mean); div.append(diversity(r.agents))
        e = Evaluator(t); p, _ = run_independent(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["independent researchers"].append(e.final_report(p).mean)
        e = Evaluator(t); p, _ = run_single(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["single agent"].append(e.final_report(p).mean)
    for k, v in res.items():
        rows.append([tariff, k, ms(v)])
    data[tariff] = dict(res, swarm_final_diversity=float(np.mean(div)))
table(rows, ["tariff", "approach", "held-out capture"])

section("3. Advantages, limitations, when to choose (grounded in the rows above)")
wins = {k: sum(np.mean(d[k]) >= max(np.mean(d[j]) for j in ("AReaL-RSI team", "SwarmRSI", "independent researchers",
                                                               "single agent")) for d in data.values())
        for k in ("AReaL-RSI team", "SwarmRSI", "independent researchers", "single agent")}
print("  tariffs won: " + ", ".join(f"{k}={v}" for k, v in wins.items()))
table([
    ["AReaL-RSI", "cheap coordination; validated findings only; interpretable campaign log",
     "orientations are hand-defined; a bad leader prior wastes campaigns", "a known research agenda with distinct sub-problems"],
    ["SwarmRSI", "parallel diverse hypotheses; recombines complementary programs; reusable archive",
     "budget split 5 ways; diversity collapses as agents converge", "open-ended search, or many campaigns that can reuse experience"],
], ["approach", "advantage", "limitation", "choose it when"])
save("10.09", data, t0)
