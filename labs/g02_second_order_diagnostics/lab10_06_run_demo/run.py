"""Lab 10.06 - Run SO-RSI (arXiv:2610.05701) on GridWise.

Naive RSI (first-order) and SO-RSI (second-order) run the SAME symptom->patch
loop with the SAME evaluation budget. SO-RSI additionally watches its own trace
for recurrence / opposing edits / expectation mismatch and pays for probes out
of that budget. Unlike the course demo, nothing about success is hard-coded:
every number comes from simulating the programs the improvers write.

Run:  python labs/g02_second_order_diagnostics/lab10_06_run_demo/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs._lab import RESULTS, banner, ms, save, section, seeds, table  # noqa: E402

import numpy as np  # noqa: E402

from gridwise.diagnose import symptoms  # noqa: E402
from gridwise.env import TARIFFS, make_task  # noqa: E402
from gridwise.evaluate import Evaluator  # noqa: E402
from gridwise.methods.so_rsi import FirstOrderImprover, SecondOrderImprover, run_improver, thrash_stats  # noqa: E402
from gridwise.policy import SEED_POLICY  # noqa: E402

t0 = banner("10.06", "Run SO-RSI: diagnose before patching", "arXiv:2610.05701")
BUDGET = 48 * 100          # 100 steering evaluations' worth of day-simulations, same for both
task = make_task("tou", seed=0)

section("1. What the first-order loop sees: oracle-relative symptoms of the starting program")
ev = Evaluator(task, charge_repeats=True)
s, sim = ev.trace(SEED_POLICY)
print(f"  seed steering {s}")
print(f"  {symptoms(task.steering, sim)}   ($-weighted mismatch vs the oracle, summed over 48 days)")

section("2. Naive RSI, first 15 attempts (TOU, seed 0)")
imp_n, fin_n = run_improver(FirstOrderImprover, Evaluator(task, charge_repeats=True), SEED_POLICY, 0, BUDGET, verbose=True)
print("\n".join("  " + l for l in imp_n.log[:15]))
print(f"  ... {len(imp_n.trace)} attempts, {sum(a.accepted for a in imp_n.trace)} accepted")

section("3. SO-RSI, same seed and budget (full trace)")
ev_s = Evaluator(task, charge_repeats=True)
imp_s, fin_s = run_improver(SecondOrderImprover, ev_s, SEED_POLICY, 0, BUDGET, verbose=True)
print("\n".join("  " + l for l in imp_s.log))
print("\n  final program:")
print("    " + fin_s.source().replace("\n", "\n    "))
imp_s.memory.save(RESULTS / "so_rsi_inquiry_memory_tou.json")
print(f"  inquiry memory saved: results/so_rsi_inquiry_memory_tou.json ({len(imp_s.memory.entries)} entries)")

section("4. Matched-budget comparison, 8 seeds x 5 tariffs (held-out capture on 100 unseen days)")
rows, data = [], {}
for tariff in TARIFFS:
    t = make_task(tariff, seed=0)
    res = {}
    for cls in (FirstOrderImprover, SecondOrderImprover):
        H, S = [], []
        for seed in seeds(8):
            e = Evaluator(t, charge_repeats=True)
            imp, fin = run_improver(cls, e, SEED_POLICY, seed, BUDGET)
            H.append(e.final_report(fin).mean); S.append(thrash_stats(imp.trace))
        res[cls.name] = (H, S)
    root = Evaluator(t).final_report(SEED_POLICY).mean
    (hn, sn), (hs, ss) = res["naive"], res["so_rsi"]
    diff = np.array(hs) - np.array(hn)
    rows.append([tariff, f"{root:.3f}", ms(hn), ms(hs), f"{diff.mean():+.3f}",
                 f"{np.mean([x['recurrence'] for x in sn]):.0f} / {np.mean([x['recurrence'] for x in ss]):.0f}",
                 f"{np.mean([x['longest_zero_progress'] for x in sn]):.0f} / {np.mean([x['longest_zero_progress'] for x in ss]):.0f}"])
    data[tariff] = dict(root=root, naive=hn, so_rsi=hs, naive_thrash=sn, so_rsi_thrash=ss)
table(rows, ["tariff", "root", "Naive RSI", "SO-RSI", "SO-RSI - naive", "recurrences n/so", "longest zero-progress n/so"])
gains = [np.mean(np.array(d["so_rsi"]) - np.array(d["naive"])) for d in data.values()]
print(f"\n  mean gain over tariffs: {np.mean(gains):+.3f}; SO-RSI better on {sum(g > 0.005 for g in gains)}/5, "
      f"worse on {sum(g < -0.005 for g in gains)}/5")
print("  Paper (Lean 4 / Verus): +21.8 and +25.8 points held-out pass rate under matched 24-hour budgets.")
print("  Here: large gains where failures are STRUCTURAL (a patch is a no-op until another rule exists),")
print("  small losses where cheap first-order patches already work and probes just spend budget.")
save("10.06", {"comparison": data, "example_trace": imp_s.log, "final_program": fin_s.source()}, t0)
