"""Lab 10.16 - Compare environments: RSIGym-style EaaS vs terminal access vs self-play.

Part 1 exercises our EaaS layer (gridwise/eaas.py): sandbox, permissions, budget.
Part 2 runs RSIGym's three improvement tracks (data / harness / joint) THROUGH the
       metered service layer.
Part 3 drives the same environment the way a terminal agent would (Terminal-RSI):
       write a file, run a CLI, parse JSON.
Part 4 sketches an R2Orch-style self-play environment: a task proposer chooses
       which days to practise on by "learnability".

Run:  python labs/g05_environments_metrics/lab10_16_environments/run.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import ROOT, banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.eaas import BudgetExceeded, EaaS, MeteredEvaluator, SandboxError  # noqa: E402
from gridwise.edits import propose  # noqa: E402
from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import BoundaryViolation  # noqa: E402
from gridwise.methods.self_hpo import run_self_hpo  # noqa: E402
from gridwise.policy import EXPERT_POLICY, SEED_POLICY  # noqa: E402

t0 = banner("10.16", "Environments for RSI: EaaS, terminal, self-play", "RSIGym arXiv:2610.10310, Terminal-RSI, R2Orch")
task = make_task("tou", seed=0)
out = {}

section("1a. Sandbox: the agent submits SOURCE TEXT; only the rule grammar is accepted")
svc = EaaS(task, role="agent", budget=48 * 10)
subs = {
    "expert policy": EXPERT_POLICY.source(),
    "import attempt": "def policy(o):\n    import os\n    return IDLE\n",
    "attribute escape": "def policy(o):\n    if o.__class__ >= 0:\n        return IDLE\n    return IDLE\n",
    "loop": "def policy(o):\n    while True:\n        pass\n",
    "unknown action": "def policy(o):\n    return SELL_TO_NEIGHBOUR\n",
}
rows = []
for name, src in subs.items():
    try:
        r = svc.evaluate(src)
        rows.append([name, "accepted", f"capture {r['capture']:.3f}"])
    except SandboxError as e:
        rows.append([name, "REJECTED", str(e)])
table(rows, ["submission", "sandbox", "result"])
out["sandbox"] = rows

section("1b. Permissions: agent vs auditor")
rows = []
for role in ("agent", "auditor"):
    s = EaaS(task, role=role)
    for split in ("steering", "heldout"):
        try:
            r = s.evaluate(EXPERT_POLICY.source(), split)
            rows.append([role, split, f"allowed (capture {r['capture']:.3f})"])
        except BoundaryViolation as e:
            rows.append([role, split, "DENIED"])
table(rows, ["role", "split", "evaluate()"])

section("1c. Budget control: an improver is stopped by the meter, not by good manners")
svc = EaaS(task, role="agent", budget=48 * 12)
pol, best = SEED_POLICY, svc.evaluate(SEED_POLICY.source())["capture"]
rng = np.random.default_rng(0)
n = 0
try:
    while True:
        child, _ = propose(pol, rng)
        c = svc.evaluate(child.source())["capture"]; n += 1
        if c > best:
            pol, best = child, c
except BudgetExceeded as e:
    print(f"  after {n} evaluations: BudgetExceeded: {e}")
print(f"  best steering capture reached within budget: {best:.3f}; service calls {svc.calls}")

section("2. RSIGym's three tracks through the metered service layer (6 seeds, 2880 day-sims)")
rows, tracks = [], {}
for tariff in ("tou", "critical_peak", "wholesale"):
    t = make_task(tariff, 0)
    res = {"data track": [], "harness track": [], "joint track": []}
    for seed in seeds(6):
        for name, mode in (("data track", "data_only"), ("harness track", "harness_only"), ("joint track", "joint")):
            ev = MeteredEvaluator(t, budget=48 * 60)          # the ENVIRONMENT enforces the budget
            st = run_self_hpo(ev, SEED_POLICY, np.random.default_rng(seed), 10 ** 9, mode=mode)
            res[name].append(ev.final_report(st.policy, wrapper=st.harness).mean)
    tracks[tariff] = res
    for k, v in res.items():
        rows.append([tariff, k, ms(v)])
table(rows, ["tariff", "track", "held-out capture"])
best = {t: max(r, key=lambda k: np.mean(r[k])) for t, r in tracks.items()}
print("\n  best track per tariff: " + ", ".join(f"{t}: {k}" for t, k in best.items()))
print("  RSIGym reports the joint track beating data-only and harness-only; check whether that holds here.")
out["tracks"] = tracks

section("3. Terminal-style access (Terminal-RSI): write a file, run the CLI, parse JSON")
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "candidate_policy.py"
    f.write_text(EXPERT_POLICY.source())
    for split in ("steering", "heldout"):
        cmd = [sys.executable, "-m", "gridwise.cli", "evaluate", str(f), "--split", split]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        print(f"  $ python -m gridwise.cli evaluate candidate_policy.py --split {split}")
        print(f"    exit={p.returncode} {p.stdout.strip()}")

section("4. Self-play environment (R2Orch-style): the proposer picks days by learnability")
rng = np.random.default_rng(1)
pop = [SEED_POLICY]
for _ in range(15):
    c, _ = propose(pop[int(rng.integers(len(pop)))], rng); pop.append(c)
from gridwise.evaluate import score_batch  # noqa: E402
S = np.array([score_batch(p, task.fit).per_day >= 0.8 for p in pop], float)     # policy x day solved
p_solve = S.mean(0)
learn = p_solve * (1 - p_solve)                                                  # outcome variance
pick = np.argsort(-learn)[:5]
print(f"  population of {len(pop)} programs on the {task.fit.n} fit days; per-day solve rate:")
print("    " + " ".join(f"{x:.2f}" for x in p_solve))
print(f"  learnability p(1-p) is highest for days {pick.tolist()} (rates {p_solve[pick].round(2).tolist()});")
print("  days every program solves (p=1) or none solves (p=0) teach nothing -> the proposer skips them.")
out["learnability"] = dict(p_solve=p_solve.tolist(), picked=pick.tolist())

section("5. When is each environment appropriate?")
table([
    ["RSIGym (EaaS)", "typed services + shared budget + permissions", "fair, metered comparisons; many improvers",
     "the agent cannot do anything the services do not expose"],
    ["Terminal-RSI", "shell + files + CLI", "agents that write and run code freely", "needs an OS sandbox; budgets and splits are by convention"],
    ["R2Orch self-play", "a task proposer + the solver population", "curriculum discovery without labels",
     "learnability rewards can drift toward easy-but-noisy tasks"],
], ["environment", "interface", "good for", "limitation"])
save("10.16", out, t0)
