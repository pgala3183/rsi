"""Lab 10.17 - Analyse the new RSI metrics, each measured on our own runs.

  RSI-Index (RSIGym)                : fraction of remaining gap closed      <- lab 10.15
  steering-split solve rate (Budget): proxy for held-out during search      <- measured here
  Recursive Learnability (R2Orch)   : does choosing tasks by p(1-p) help?   <- measured here
  swarm diversity (SwarmRSI)        : rule-set distance between agents      <- measured here

Run:  python labs/g05_environments_metrics/lab10_17_metrics/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, load, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.edits import propose  # noqa: E402
from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator, score_batch  # noqa: E402
from gridwise.methods.budget_rsi import search  # noqa: E402
from gridwise.methods.swarm_rsi import OP_PRIOR, run_swarm  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.17", "Analyse the October 2026 RSI metrics on real runs")
task = make_task("tou", seed=0)
out = {}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


section("1. RSI-Index (from lab 10.15)")
try:
    r = load("10.15")
    print("  " + ", ".join(f"{m}: {v:.3f}" for m, v in r["rsi_index"].items()))
    out["rsi_index"] = r["rsi_index"]
except FileNotFoundError:
    print("  run lab 10.15 first")

section("2. Steering-split solve rate: is the search proxy faithful to held-out?")
st_mean, st_rate, ho, gaps = [], [], [], []
for seed in seeds(6):
    ev = Evaluator(task)
    res = search(ev, SEED_POLICY, "hgm", np.random.default_rng(seed), archive_size=30)
    for nd in res.nodes:
        st_mean.append(nd.score.mean); st_rate.append(nd.score.rate); ho.append(ev.final_report(nd.policy).mean)
    gaps.append(res.best.score.mean - ev.final_report(res.best.policy).mean)
print(f"  {len(ho)} evaluated nodes from 6 HGM searches")
print(f"  Spearman(steering capture, held-out capture)    = {spearman(st_mean, ho):.3f}")
print(f"  Spearman(steering solve RATE, held-out capture) = {spearman(st_rate, ho):.3f}")
print(f"  selected node: steering minus held-out = {ms(gaps)}  (positive = winner's-curse optimism)")
out["steering_proxy"] = dict(spearman_mean=spearman(st_mean, ho), spearman_rate=spearman(st_rate, ho), optimism=gaps)

section("3. Recursive Learnability: train on the days with the most diverse outcomes?")
rng = np.random.default_rng(1)
pop = [SEED_POLICY]
for _ in range(30):
    c, _ = propose(pop[int(rng.integers(len(pop)))], rng); pop.append(c)
pool = task.steering
S = np.array([score_batch(p, pool).per_day >= 0.8 for p in pop], float)
p = S.mean(0); learn = p * (1 - p)
choices = {"learnable (max p(1-p))": np.argsort(-learn)[:12],
           "hardest (lowest p)": np.argsort(p)[:12],
           "easiest (highest p)": np.argsort(-p)[:12]}
res = {}
for name, idx in choices.items():
    hs = []
    for seed in seeds(8):
        ev = Evaluator(task)
        pol, cur = SEED_POLICY, ev.score(SEED_POLICY, "steering", idx=idx)
        g = np.random.default_rng(seed)
        while ev.day_sims < 12 * 80:
            ch, _ = propose(pol, g, op_weights=OP_PRIOR)
            s = ev.score(ch, "steering", idx=idx)
            if s.mean > cur.mean + 1e-4:
                pol, cur = ch, s
        hs.append(ev.final_report(pol).mean)
    res[name] = hs
rand = []
for seed in seeds(8):
    idx = np.random.default_rng(100 + seed).choice(pool.n, 12, replace=False)
    ev = Evaluator(task); pol, cur = SEED_POLICY, ev.score(SEED_POLICY, "steering", idx=idx)
    g = np.random.default_rng(seed)
    while ev.day_sims < 12 * 80:
        ch, _ = propose(pol, g, op_weights=OP_PRIOR)
        s = ev.score(ch, "steering", idx=idx)
        if s.mean > cur.mean + 1e-4:
            pol, cur = ch, s
    rand.append(ev.final_report(pol).mean)
res["random 12 days"] = rand
table([[k, ms(v)] for k, v in res.items()], ["12 training days chosen by", "held-out capture after 80 evaluations"])
print("\n  Learnability picks days whose outcomes DIFFER across the population -- a narrow slice of day types.")
print("  Optimising on that slice alone transfers worse than a random sample: learnability is a good signal")
print("  for what to ADD to a curriculum, not a replacement for a representative one.")
out["learnability"] = res

section("4. Swarm diversity: does it predict whether recombination still works?")
curves, recs = [], []
for seed in seeds(4):
    ev = Evaluator(task)
    r = run_swarm(ev, SEED_POLICY, np.random.default_rng(seed), 48 * 150)
    curves.append([c[2] for c in r.curve])
    recs.append(sum("recombined" in e for e in r.events))
L = min(len(c) for c in curves)
mean_curve = np.mean([c[:L] for c in curves], axis=0)
print("  mean diversity by generation: " + " ".join(f"{x:.2f}" for x in mean_curve))
print(f"  successful recombinations per run: {recs}")
out["diversity"] = dict(curve=mean_curve.tolist(), recombinations=recs)

section("5. Summary: what each metric solves, when it is useful, its limits (with our numbers)")
table([
    ["RSI-Index", "compares improvement across benchmarks of different difficulty",
     "reporting one number over a suite", "near-solved benchmarks dominate; hides absolute gains; 'ideal' must exist"],
    ["steering-split solve rate", "cheap proxy used for selection during search",
     f"when it ranks like held-out (Spearman {out['steering_proxy']['spearman_mean']:.2f} here)",
     f"the selected node is optimistic by {np.mean(gaps):+.3f}; solve counts are coarser than means"],
    ["Recursive Learnability", "chooses practice tasks that separate good from bad solvers",
     "self-play curricula without labels",
     f"here a learnable-only curriculum scored {np.mean(res['learnable (max p(1-p))']):.3f} vs random "
     f"{np.mean(res['random 12 days']):.3f}: it is unrepresentative of the test distribution"],
    ["swarm diversity", "detects premature convergence of a swarm",
     "deciding when to inject new hypotheses", "syntactic distance can hide behavioural sameness"],
], ["metric", "problem it solves", "useful when", "limitation"])
save("10.17", out, t0)
