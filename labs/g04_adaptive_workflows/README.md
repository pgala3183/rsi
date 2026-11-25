# Group 04 · Adaptive Workflows: RSIFlow & Self-HPO (labs 10.12–10.14)

**Papers:** RSIFlow (arXiv:2610.21791) · Self-HPO (arXiv:2610.22791)
**Code:** [`gridwise/methods/rsiflow.py`](../../gridwise/methods/rsiflow.py) · [`gridwise/methods/self_hpo.py`](../../gridwise/methods/self_hpo.py)

**Question:** can the improvement *workflow* adapt to the current bottleneck?

```
 RSIFlow: route the next improvement action           Self-HPO: alternate what surrounds the policy and the policy
 ──────────────────────────────────────────           ───────────────────────────────────────────────────────────
 diagnose -> bottleneck (largest symptom)             H-step: the policy analyses ITS failures on validation days
      │ meta-agent: UCB credit per                            -> proposes runtime guards for its top symptoms
      │ (bottleneck, branch)                                  -> its hardest training days become the curriculum
      ▼                                               P-step: hill-climb the policy on the curriculum,
 tune | extend | restructure | prune | (extend+tune)          running INSIDE the harness
      │ evaluate, credit                              deployed system = policy + harness (guards)
      ▼
 meta-agent edits the graph: prune dead branches,
 grow a composite branch when stalled
```

| Lab | What you run | Finding |
|-----|--------------|---------|
| [10.12 · Compare](lab10_12_compare/README.md) | RSIFlow vs Self-HPO at one budget | Self-HPO ahead on TOU, RSIFlow ahead on wholesale, ties elsewhere |
| [10.13 · RSIFlow](lab10_13_rsiflow/README.md) | routing log, credit table, graph edits, adaptive vs static | no gain here (−0.004 on average): the bottleneck barely changes |
| [10.14 · Self-HPO](lab10_14_self_hpo/README.md) | H/P alternation log, learned guards, synergy | synergy holds on 1/4 tariffs (TOU, +0.032) |

Next: [Group 05 · Environments & metrics](../g05_environments_metrics/README.md)
