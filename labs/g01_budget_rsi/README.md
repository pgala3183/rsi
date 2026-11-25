# Group 01 · Budget RSI: Scoring Before Running (labs 10.03–10.05)

**Paper:** *Recursive Self-Improvement on a Budget: Scoring Code Edits Before They Run*, arXiv:2610.29121
**Code:** [`gridwise/methods/budget_rsi.py`](../../gridwise/methods/budget_rsi.py) · proposer and diff features in [`gridwise/edits.py`](../../gridwise/edits.py)

**Question:** can we pick which self-edit to evaluate *before* running it?

```
 calibration (once per run)                        search (archive of 4 nodes)
 ───────────────────────────                       ─────────────────────────────────────────────
 96 random edits ──► simulate on 16 fit days        each archive node proposes 4 edits
        │                                                     │
        ▼                                                     ▼  score every proposal WITHOUT running it
 X = [parent rate, 8 diff coordinates]              readout(parent rate, diff(proposal))
 y = child's fit capture                                      │
        │                                                     ▼
        └──► ridge fit ──► FROZEN linear readout ───► evaluate ONLY the top proposal on 48 steering days
                                                              │
                                                              ▼
                                          return the earliest node with the max steering solve count
```

| Lab | What you run | Finding |
|-----|--------------|---------|
| [10.03 · Run the demo](lab10_03_run_demo/README.md) | calibration, one scored diff, one search, 12-run comparison, cost | readout 0.817 > HGM 0.787 > parent-rate 0.770 > root 0.712: the paper's ordering |
| [10.04 · Analyse the mechanism](lab10_04_analyze_mechanism/README.md) | coefficients, fresh-edit ranking, ablations, training size | the diff coordinates carry the signal; parent rate barely matters here |
| [10.05 · Extend](lab10_05_extend/README.md) | domain diff features, cross-tariff transfer, matched total cost | +0.030 from domain features; transfers; at matched total cost plain HGM is as good |

**Bottom line:** the mechanism reproduces where the paper ran it (a tiny archive, a fixed search budget). Its cost advantage depends on amortising calibration.

Next: [Group 02 · SO-RSI](../g02_second_order_diagnostics/README.md)
