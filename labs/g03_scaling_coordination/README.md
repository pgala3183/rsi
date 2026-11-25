# Group 03 · Scaling RSI: Coordination & Swarms (labs 10.09–10.11)

**Papers:** AReaL-RSI (arXiv:2610.18083) · **SwarmRSI** (arXiv:2610.20431)
**Code:** [`gridwise/methods/areal_rsi.py`](../../gridwise/methods/areal_rsi.py) · [`gridwise/methods/swarm_rsi.py`](../../gridwise/methods/swarm_rsi.py)

**Question:** how should several improving agents share work and experience?

```
 AReaL-RSI (leader-follower)                          SwarmRSI (peer swarm)
 ───────────────────────────                          ─────────────────────────────────────────────
 leader reads CONDITIONAL MEMORY                      5 agents = (hypothesis, implementation)
   unresolved problems x orientation priority           1 mutation   : edits drawn from own hypothesis
        │ assigns orientations                                         + shared feedback board
        ▼                                               2 recombine  : most COMPLEMENTARY pair (good on
 researcher A          researcher B                                    different days) -> merged / gated child
 (peak / arbitrage /   (solar / events ...)             3 evolve     : evidence re-weights hypotheses;
  ...) short search     short search                                   stagnant agent restarts from a mutated
        │ results marked supported / unresolved                        copy of the leader's hypothesis
        ▼                                               archive of successful edits = EXPERIENCE for the
 evidence-based review re-weights orientations          next campaign
```

| Lab | What you run | Finding |
|-----|--------------|---------|
| [10.09 · Compare](lab10_09_compare/README.md) | both approaches + 2 baselines, one budget, 3 tariffs | all four within seed noise on 2 tariffs; simple baselines win on wholesale |
| [10.10 · Leader-follower](lab10_10_leader_follower/README.md) | campaign log, conditional memory, sharing vs independent | sharing beats independent on 3/3 tariffs; the conditional filter is task-dependent |
| [10.11 · Swarm](lab10_11_swarm/README.md) | events + diversity curve, ablations, budget curve, experience transfer with a control | recombination & feedback matter on TOU; the experience gain is mostly program transfer |

Next: [Group 04 · Adaptive workflows](../g04_adaptive_workflows/README.md)
