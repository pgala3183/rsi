# Group 02 · Second-Order RSI: Diagnostic Inquiry (labs 10.06–10.08)

**Paper:** *Second-Order Problem Solving for Recursive Self-Improvement in Formal Verification* (SO-RSI), arXiv:2610.05701
**Code:** [`gridwise/methods/so_rsi.py`](../../gridwise/methods/so_rsi.py) · symptoms in [`gridwise/diagnose.py`](../../gridwise/diagnose.py)

**Question:** should an improver ask *why* its patches fail before patching again?

```
           FIRST-ORDER LOOP (both arms)                         SECOND-ORDER LAYER (SO-RSI only)
 ┌──────────────────────────────────────────────┐   after every attempt, read the improver's OWN trace
 │ symptoms vs oracle  ──►  pick a symptom       │   ┌──────────────┬──────────────────────────────────┐
 │        ▲                    │                 │   │ recurrence   │ sweep probe -> lever probe        │
 │        │            symptom -> (slot, dir)    │──►│ opposing     │ split probe (days disagree?)      │
 │        │            patch table               │   │ mismatch     │ attribution probe (where did $ go)│
 │        │                    ▼                 │   └──────┬───────┴──────────────────────────────────┘
 │ accept if steering ▲   evaluate patch         │          ▼  findings -> INQUIRY MEMORY
 └──────────────────────────────────────────────┘   bans exhausted levers, learns guards, shrinks steps,
                  ▲                                  re-maps symptoms to the slot that really causes them
                  └──────── memory edits the patch table & proposer (= improving the improver) ────────┘
```

| Lab | What you run | Finding |
|-----|--------------|---------|
| [10.06 · Run the demo](lab10_06_run_demo/README.md) | full annotated trace, 5 tariffs × 8 seeds at matched budget | +0.122 TOU, +0.083 critical-peak, tie VPP, small losses where patches already work; recurrences 60 → 6 |
| [10.07 · Analyse probes](lab10_07_analyze_probes/README.md) | probe-by-probe outcomes and cost | 78% of findings negative; mismatch → guard delivers almost all of the gain; probes are 64% of the budget |
| [10.08 · Apply to debugging](lab10_08_apply_debugging/README.md) | a failing test, a 5-attempt debugging trace | the anomalies point away from the policy, to an off-by-one-hour pipeline bug |

**Bottom line:** second-order diagnosis reliably suppresses thrashing. It raises the final score when failures are *structural* (a patch is a no-op until something else changes). When cheap patches already work, it costs budget.

Next: [Group 03 · Scaling](../g03_scaling_coordination/README.md)
