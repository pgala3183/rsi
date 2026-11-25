# 10.16 · Compare Environments: RSIGym (EaaS) vs Terminal-RSI vs R2Orch

**Purpose:** build an RSIGym-style environment, then drive the same task the way each environment style would.
**Run:** `python labs/g05_environments_metrics/lab10_16_environments/run.py` (≈18 s) · **Evidence:** [`results/logs/lab10_16.txt`](../../../results/logs/lab10_16.txt)

## Walkthrough of `gridwise/eaas.py`

* **`parse_policy()` is the sandbox.** An agent submits policy *source text*, and only the rule grammar is accepted: `def policy(o):`, `if o.<feature> (>=|<=) <number> [and ...]:`, `return <ACTION>`. It is a whitelist parser, not `exec`, so imports, dunder access, loops and unknown actions are rejected before anything runs.
* **`EaaS` services:** `sandbox`, `evaluate`, `rollout` (one day's hourly trace), `oracle_gap`.
* **Roles:** `agent` sees fit and steering only and is metered; `auditor` may read held-out.
* **`MeteredEvaluator`** raises `BudgetExceeded` *before* simulating past its budget. The environment, not the improver's good manners, enforces cost.

## What we observed

**1. Sandbox**

| submission | sandbox | result |
|---|---|---|
| expert policy | accepted | capture 0.932 |
| `import os` | REJECTED | statement not allowed |
| `o.__class__ >= 0` | REJECTED | condition not allowed |
| `while True:` | REJECTED | statement not allowed |
| `return SELL_TO_NEIGHBOUR` | REJECTED | unknown action |

**Permissions:** the agent is DENIED held-out; the auditor gets 0.924. **Budget:** an improver with 576 day-sims is stopped after 16 evaluations by `BudgetExceeded`.

**2. RSIGym's three tracks, run through the metered layer** (6 seeds, 2,880 day-sims). These reuse Self-HPO's components: the data track may change only the training curriculum, the harness track only runtime guards, and the joint track both.

| tariff | data track | harness track | joint track |
|---|---|---|---|
| tou | 0.879 | 0.712 | **0.920** |
| critical_peak | **0.946** | 0.843 | 0.935 |
| wholesale | **0.795** | 0.633 | 0.709 |

RSIGym reports joint > data-only and harness-only. Here that holds on TOU only.

**3. Terminal-style access** (Terminal-RSI): the agent writes `candidate_policy.py` and runs
`python -m gridwise.cli evaluate candidate_policy.py --split steering` → `{"capture": 0.932, ...}`. The same command with `--split heldout` gets exit code 3 and `BoundaryViolation`. The terminal interface goes through the same services.

**4. Self-play environment** (R2Orch-style): a population of 16 programs, with each fit day's solve rate *p*. The task proposer picks days by **learnability p(1−p)**, skipping days that everyone or no one solves. Lab 10.17 tests whether that choice is good.

## When is each environment appropriate?

| environment | interface | good for | limitation |
|---|---|---|---|
| RSIGym (EaaS) | typed services + shared budget + permissions | fair, metered comparisons across many improvers | the agent can do only what the services expose |
| Terminal-RSI | shell + files + CLI | agents that write and run code freely | needs an OS-level sandbox; budgets and splits hold by convention unless a service enforces them (as here) |
| R2Orch self-play | task proposer + solver population | curriculum discovery without labels | learnability depends on the population and can favour unrepresentative tasks (lab 10.17) |

Next: [10.17 · Analyse metrics](../lab10_17_metrics/README.md)
