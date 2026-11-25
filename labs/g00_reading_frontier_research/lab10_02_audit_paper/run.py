"""Lab 10.02 - Audit a paper (and audit our own adaptation with the same template).

The audit template asks: what changed, what feedback, what evaluation protocol,
what cost, what is untested. Before trusting ANY number this repo produces, we
check that the GridWise harness actually enforces the protocol the papers rely on.

Run:  python labs/g00_reading_frontier_research/lab10_02_audit_paper/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, save, section, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import SOLVED_AT, TARIFFS, capture, make_task, simulate  # noqa: E402
from gridwise.evaluate import BoundaryViolation, Evaluator  # noqa: E402
from gridwise.edits import propose  # noqa: E402
from gridwise.methods.budget_rsi import calibrate, search  # noqa: E402
from gridwise.policy import EXPERT_POLICY, IDLE_POLICY, SEED_POLICY, check_source_matches  # noqa: E402

t0 = banner("10.02", "Audit the evaluation protocol before trusting any result")
checks = []


def check(name, ok, detail):
    checks.append([name, "PASS" if ok else "FAIL", detail])


task = make_task("tou", seed=0)

section("1. Partitions: 16 fit / 48 steering / 100 held-out days, disjoint")
keys = {s: {(int(d), round(float(task.split(s).price[i].sum()), 6)) for i, d in enumerate(task.split(s).doy)}
        for s in ("fit", "steering", "heldout")}
sizes = {s: task.split(s).n for s in keys}
overlap = len(keys["fit"] & keys["steering"]) + len(keys["fit"] & keys["heldout"]) + len(keys["steering"] & keys["heldout"])
print(f"  sizes {sizes}, overlapping days {overlap}")
check("split sizes match the paper (16/48/100)", sizes == {"fit": 16, "steering": 48, "heldout": 100}, str(sizes))
check("splits are disjoint", overlap == 0, f"{overlap} shared days")

section("2. The held-out boundary is enforced, not just promised")
ev = Evaluator(task)
try:
    ev.score(SEED_POLICY, "heldout", purpose="sneaky improver")
    denied = False
except BoundaryViolation as e:
    denied = True
    print(f"  BoundaryViolation: {e}")
cal = calibrate(ev, SEED_POLICY, np.random.default_rng(0))
res = search(ev, SEED_POLICY, "readout", np.random.default_rng(1), readout=cal.readout)
touched = [a for a in ev.access_log if a[1] == "heldout" and a[0] == "ok"]
print(f"  a full Budget RSI run made {len(ev.access_log)} split accesses; held-out reads during search: {len(touched)}")
check("improver cannot read held-out", denied, "BoundaryViolation raised")
check("search never touched held-out", len(touched) == 0, f"{len(touched)} reads")

section("3. The cost ledger counts what was simulated")
print(f"  ledger after one Budget RSI run: {dict(ev.ledger)}")
expected_fit = len({c.key for c in cal.children} | {p.key for p in cal.parents}) * 16
check("fit-split cost = 16 x distinct programs labelled", ev.ledger["fit"] == expected_fit,
      f"{ev.ledger['fit']} vs {expected_fit}")
check("steering cost = 48 x archive nodes", ev.ledger["steering"] == 48 * len(res.nodes),
      f"{ev.ledger['steering']} vs {48 * len(res.nodes)}")

section("4. Determinism: same seed -> same result")
def one(seed):
    e = Evaluator(task); c = calibrate(e, SEED_POLICY, np.random.default_rng(seed))
    r = search(e, SEED_POLICY, "readout", np.random.default_rng(seed + 100), readout=c.readout)
    return r.best.policy.key, e.final_report(r.best.policy).mean
a, b = one(3), one(3)
check("re-running a seed reproduces the program and score", a == b, f"score {a[1]:.4f} vs {b[1]:.4f}")

section("5. The readable program IS the evaluated program")
rng = np.random.default_rng(0)
pols = [EXPERT_POLICY, SEED_POLICY]
p = SEED_POLICY
for _ in range(30):
    p, _ = propose(p, rng); pols.append(p)
bad = sum(check_source_matches(q, task.steering, n_checks=100, seed=i) for i, q in enumerate(pols))
print(f"  exec'd rendered source vs vectorised interpreter on {len(pols)} programs x 100 states: {bad} mismatches")
check("source == interpreter", bad == 0, f"{bad} mismatches")

section("6. Is the oracle really an upper bound? (per day, every tariff)")
from gridwise.env import SOC_STEP  # noqa: E402
rows = []
worst_ratio, beaten = 0.0, 0
for tariff in TARIFFS:
    b = make_task(tariff, 0).heldout
    day_peak = np.maximum(b.price, b.export).max(axis=1)
    for name, pol in (("idle", IDLE_POLICY), ("seed", SEED_POLICY), ("expert", EXPERT_POLICY)):
        excess = b.cost_oracle - simulate(b, pol.act).cost        # > 0: the policy beat the "optimum"
        beaten += int((excess > 1e-9).sum())
        worst_ratio = max(worst_ratio, float((excess / (SOC_STEP * day_peak)).max()))
    rows.append([tariff, float(np.mean(b.cost_none)), float(np.mean(b.cost_oracle)),
                 float(np.mean(capture(b, simulate(b, SEED_POLICY.act).cost) >= SOLVED_AT))])
table(rows, ["tariff", "no-battery $/day", "oracle $/day", "seed solved frac"])
print(f"  policy-days where a continuous policy beat the DP oracle: {beaten} of {3 * 100 * len(TARIFFS)}")
print(f"  worst excess, in units of (one {SOC_STEP} kWh SoC step x that day's peak price): {worst_ratio:.2f}")
print("  FINDING: the oracle is optimal on a 0.25 kWh grid, not in continuous space. A policy can beat it")
print("  by about one grid step of energy at the day's best price. capture() therefore clips at 1.0, and")
print("  the 'ideal' in RSI-Index is 'oracle-level', not a proven optimum.")
check("oracle slack <= 1.5 SoC steps at the day's peak price", worst_ratio <= 1.5, f"worst {worst_ratio:.2f} steps")

section("Audit summary")
table(checks, ["check", "result", "detail"])
assert all(c[1] == "PASS" for c in checks), "protocol audit failed"
save("10.02", {"checks": checks}, t0)
