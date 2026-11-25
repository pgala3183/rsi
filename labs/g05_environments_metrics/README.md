# Group 05 · Environments & Metrics: RSIGym & More (labs 10.15–10.17)

**Papers:** RSIGym (arXiv:2610.10310) · Terminal-RSI (arXiv:2610.22192) · R2Orch (arXiv:2610.22740) · plus the metrics of Budget RSI and SwarmRSI
**Code:** [`gridwise/eaas.py`](../../gridwise/eaas.py) (Everything-as-a-Service layer) · [`gridwise/cli.py`](../../gridwise/cli.py) (terminal access)

**Question:** what environment and what metric make RSI results comparable?

| Lab | What you run | Finding |
|-----|--------------|---------|
| [10.15 · RSI-Index](lab10_15_rsi_index/README.md) | the three main methods on 5 tariff benchmarks at one budget | SwarmRSI 0.732 > SO-RSI 0.607 > Budget RSI 0.524. The course formula has a sign error |
| [10.16 · Environments](lab10_16_environments/README.md) | sandbox, permissions and budget meter; RSIGym's 3 tracks; terminal CLI; self-play learnability | the environment, not the agent, enforces budgets and splits; joint track best on 1/3 tariffs |
| [10.17 · Metrics](lab10_17_metrics/README.md) | each metric computed on real runs | steering proxy Spearman 0.94 but +0.037 optimistic; a learnability-only curriculum underperforms random days |

Next: back to the [repository overview](../../README.md), or on to Theme 11 (capstones) of the course.
