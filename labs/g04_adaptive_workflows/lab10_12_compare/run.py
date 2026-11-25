"""Lab 10.12 - Compare workflow adaptation: RSIFlow vs Self-HPO.

Both papers adapt the improvement process, but they change different objects:
  RSIFlow  changes WHICH improvement action runs next (routing over a workflow graph)
  Self-HPO changes WHAT SURROUNDS the policy (curriculum + runtime guards) and alternates
           that with policy optimisation
We run both at the same budget on the same tariffs and record what each one changed.

Run:  python labs/g04_adaptive_workflows/lab10_12_compare/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.rsiflow import run_flow  # noqa: E402
from gridwise.methods.self_hpo import run_self_hpo  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.12", "Compare workflow adaptation: RSIFlow vs Self-HPO", "arXiv:2610.21791 vs arXiv:2610.22791")
BUDGET = 48 * 60

section("1. What each method adapts")
table([
    ["changed object", "the routing of improvement actions (+ the graph itself)", "harness (curriculum, guards) AND policy"],
    ["feedback", "credit per (bottleneck, branch)", "the policy's own failure symptoms on validation days"],
    ["iteration", "continuous: route -> evaluate -> credit", "alternating H-step / P-step"],
    ["deployed artefact", "a policy program", "a policy program + a runtime wrapper"],
], ["aspect", "RSIFlow", "Self-HPO"])

section("2. Same budget, same tariffs (8 seeds, held-out capture)")
rows, data = [], {}
for tariff in ("tou", "critical_peak", "wholesale", "vpp_events"):
    t = make_task(tariff, 0)
    res = {"RSIFlow (adaptive)": [], "Self-HPO (joint)": []}
    guards, edits = [], []
    for seed in seeds(8):
        e = Evaluator(t); p, _, meta, route = run_flow(e, SEED_POLICY, np.random.default_rng(seed), BUDGET)
        res["RSIFlow (adaptive)"].append(e.final_report(p).mean); edits.append(len(meta.events))
        e = Evaluator(t); st = run_self_hpo(e, SEED_POLICY, np.random.default_rng(seed), BUDGET, mode="joint")
        res["Self-HPO (joint)"].append(e.final_report(st.policy, wrapper=st.harness).mean); guards.append(len(st.harness.guards))
    rows.append([tariff, ms(res["RSIFlow (adaptive)"]), ms(res["Self-HPO (joint)"]),
                 f"{np.mean(edits):.1f}", f"{np.mean(guards):.1f}"])
    data[tariff] = dict(res, rsiflow_graph_edits=float(np.mean(edits)), self_hpo_guards=float(np.mean(guards)))
table(rows, ["tariff", "RSIFlow", "Self-HPO", "RSIFlow graph edits / run", "Self-HPO guards kept / run"])

section("3. When is each preferable?")
table([
    ["RSIFlow", "the bottleneck shifts over time and different bottlenecks need different actions",
     "the bottleneck is stable (as on most GridWise tariffs) -> routing reduces to round-robin plus exploration cost"],
    ["Self-HPO", "failures are fixable by wrapping the policy (a missing guard, a skewed curriculum)",
     "few validation days -> guards overfit; you must ship and maintain the wrapper too"],
], ["method", "preferable when", "weak when"])
save("10.12", data, t0)
