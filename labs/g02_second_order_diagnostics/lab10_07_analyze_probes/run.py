"""Lab 10.07 - Analyse the diagnostic probes.

For every anomaly SO-RSI raised (8 seeds x 2 tariffs) we record: which probe ran,
what it concluded, whether the structural edit it licensed survived evaluation,
and what it cost. We also run the SAME detector post-hoc on Naive RSI's traces
to see how much thrashing the first-order loop does when nobody is watching.

Run:  python labs/g02_second_order_diagnostics/lab10_07_analyze_probes/run.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import banner, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.env import make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.so_rsi import (FirstOrderImprover, SecondOrderImprover, detect,  # noqa: E402
                                     run_improver, thrash_stats)
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.07", "Analyse diagnostic probes", "arXiv:2610.05701")
BUDGET = 48 * 100
PROBE_OF = {"sweep": "recurrence", "reorder": "recurrence", "reroute": "recurrence", "split": "opposing", "guard": "mismatch"}

triggers = Counter(); simultaneous = 0; findings = defaultdict(Counter)
interventions = defaultdict(lambda: [0, 0, 0.0])        # anomaly -> [tried, accepted, total gain]
probe_cost = Counter(); total_cost = 0
naive_stats = []
for tariff in ("tou", "critical_peak"):
    task = make_task(tariff, 0)
    for seed in seeds(8):
        ev = Evaluator(task, charge_repeats=True)
        imp, _ = run_improver(SecondOrderImprover, ev, SEED_POLICY, seed, BUDGET)
        first = [a for a in imp.trace if a.kind in ("create", "tweak")]
        for j in range(1, len(first) + 1):
            found = detect(first[:j])
            triggers.update(found)
            simultaneous += len(found) >= 2
        for e in imp.memory.entries:
            f = e["finding"]
            key = ("NOT a lever" if "NOT a lever" in f else "IS a lever" if "IS a lever" in f else
                   "no lever found" if "no lever" in f else "shadowed" if "shadowed" in f else
                   "energy-limited" if "energy-limited" in f else "conflicting constraints" if "CONFLICTING" in f else
                   "no conflict" if "no conflict" in f else "blocks absorbing" if "blocks absorbing" in f else
                   "grid charge displaced solar" if "free solar" in f else "energy spent early" if "spent before" in f else
                   "inconclusive" if "inconclusive" in f else "other")
            findings[e["anomaly"]][f"{e['probe']}: {key}"] += 1
        for a in imp.trace:
            if a.kind in PROBE_OF:
                st = interventions[PROBE_OF[a.kind]]
                st[0] += 1; st[1] += a.accepted; st[2] += max(a.d_score, 0)
        for k, v in ev.by_purpose.items():
            probe_cost["probes" if k.startswith("probe") else "first-order loop"] += v
        total_cost += ev.day_sims
        evn = Evaluator(task, charge_repeats=True)
        impn, _ = run_improver(FirstOrderImprover, evn, SEED_POLICY, seed, BUDGET)
        naive_stats.append(thrash_stats(impn.trace))

section("1. How often each anomaly fired (SO-RSI's own first-order attempts)")
table([[k, v] for k, v in triggers.most_common()] + [["two or more at once", simultaneous]], ["anomaly", "count"])

section("2. What each probe concluded")
rows = [[an, k, v] for an, c in findings.items() for k, v in c.most_common()]
table(rows, ["anomaly", "probe: finding", "count"])
inconclusive = sum(v for c in findings.values() for k, v in c.items() if "inconclusive" in k or "no lever" in k or "no conflict" in k)
total_f = sum(sum(c.values()) for c in findings.values())
print(f"\n  inconclusive / negative findings: {inconclusive}/{total_f} ({inconclusive / max(total_f, 1):.0%}).")
print("  Negative findings are still recorded: they ban an exhausted patch, which is what stops recurrence.")

section("3. Did the structural edit each probe recommended survive evaluation?")
table([[an, v[0], v[1], f"{v[1] / max(v[0], 1):.0%}", f"{v[2]:.3f}"] for an, v in interventions.items()],
      ["anomaly", "interventions tried", "accepted", "acceptance", "total capture gained"])

section("4. Recommendation table (the lab's expected output)")
table([
    ["recurrence", "sweep the threshold; if no value helps, replay the trace (lever probe)",
     "stops retrying a no-op; reveals shadowed rules / empty battery", "costs 6-10 evaluations"],
    ["opposing", "evaluate +step and -step per day; split days on day-level context",
     "turns a tug-of-war into two context-specific rules", "only helps if the days really disagree"],
    ["mismatch", "attribute the cost change to hours; check SoC / exports there",
     "adds the missing guard (e.g. no grid charge on sunny days)", "inconclusive when the cost moves diffusely"],
], ["anomaly", "probe", "why it addresses the root cause", "limitation"])

section("5. The same detector run post-hoc on Naive RSI's traces (nobody acts on it)")
keys = ("recurrence", "opposing", "mismatch", "longest_zero_progress", "attempts", "accepted")
table([[k, f"{np.mean([s[k] for s in naive_stats]):.1f}"] for k in keys], ["naive trace statistic", "mean per run"])
print(f"\n  share of SO-RSI's day-sims spent on probes: {probe_cost['probes'] / max(total_cost, 1):.0%} "
      f"(the rest went to the first-order loop)")
save("10.07", {"triggers": dict(triggers), "simultaneous": simultaneous, "probe_share": probe_cost['probes'] / max(total_cost, 1),
               "findings": {k: dict(v) for k, v in findings.items()},
               "interventions": dict(interventions), "naive_stats": naive_stats}, t0)
