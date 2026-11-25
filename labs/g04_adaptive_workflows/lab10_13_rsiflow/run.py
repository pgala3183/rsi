"""Lab 10.13 - Simulate the RSIFlow meta-agent (arXiv:2610.21791).

The improvement workflow is a graph:  diagnose -> {tune | extend | restructure | prune} -> evaluate.
A static workflow visits branches round-robin. The RSIFlow meta-agent names the
current bottleneck (largest oracle-relative symptom), routes to the branch with
the best upper-confidence credit for that bottleneck, and edits the graph itself:
it prunes branches that never pay off and grows a composite branch when stuck.

Run:  python labs/g04_adaptive_workflows/lab10_13_rsiflow/run.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.rsiflow import BRANCHES, run_flow  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.13", "RSIFlow: an evolving meta-agent routes the workflow", "arXiv:2610.21791")
BUDGET = 48 * 80
task = make_task("tou", seed=0)

section("1. The workflow graph")
print("  diagnose -> " + " | ".join(f"{b} {list(w)}" for b, w in BRANCHES.items()) + " -> evaluate -> accept?")

section("2. One adaptive run (TOU, seed 0): how routing and the graph evolve")
ev = Evaluator(task)
pol, sc, meta, route = run_flow(ev, SEED_POLICY, np.random.default_rng(0), BUDGET)
for step, bott, br, g in route[:25]:
    print(f"  step {step:2d}: bottleneck={bott:18s} -> {br:12s} gain={g:+.4f}")
print(f"  ... {len(route)} steps")
print("  graph edits by the meta-agent: " + ("; ".join(meta.events) or "none"))
print("\n  learned credit table (visits / mean gain x100) per (bottleneck, branch):")
rows = [[k[0], k[1], v[0], f"{v[1] / max(v[0], 1) * 100:.2f}"] for k, v in sorted(meta.stats.items()) if v[0]]
table(rows, ["bottleneck", "branch", "visits", "mean gain x100"])
print(f"\n  routing mix: {dict(Counter(b for _, _, b, _ in route))}")
print(f"  held-out {ev.final_report(pol)}")

section("3. Adaptive meta-agent vs static workflow (8 seeds, same budget)")
rows, data = [], {}
for tariff in ("tou", "critical_peak", "wholesale", "solar_glut"):
    t = make_task(tariff, 0)
    a, s = [], []
    for seed in seeds(8):
        e = Evaluator(t); p, *_ = run_flow(e, SEED_POLICY, np.random.default_rng(seed), BUDGET); a.append(e.final_report(p).mean)
        e = Evaluator(t); p, *_ = run_flow(e, SEED_POLICY, np.random.default_rng(seed), BUDGET, adaptive=False); s.append(e.final_report(p).mean)
    d = np.array(a) - np.array(s)
    rows.append([tariff, ms(s), ms(a), f"{d.mean():+.3f} ({int((d > 1e-9).sum())}/{len(d)})"])
    data[tariff] = dict(static=s, adaptive=a)
table(rows, ["tariff", "static workflow", "RSIFlow meta-agent", "adaptive - static (wins)"])
bott = Counter(b for _, b, _, _ in route)
top_b, top_n = bott.most_common(1)[0]
gains = np.mean([np.mean(np.array(d["adaptive"]) - np.array(d["static"])) for d in data.values()])
print(f"\n  mean adaptive - static over tariffs: {gains:+.3f}.")
print(f"  Why so little? In the example run the bottleneck was '{top_b}' in {top_n}/{len(route)} steps: the")
print("  symptom signal barely changes, so there is little to adapt TO, and UCB still pays to explore")
print("  every branch for that one bottleneck. RSIFlow's premise -- bottlenecks that EVOLVE -- is weak on")
print("  this task; the static recipe already spreads effort across the useful actions.")
save("10.13", {"route": route, "meta_events": meta.events, "comparison": data}, t0)
