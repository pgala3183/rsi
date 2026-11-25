# 10.11 · Simulate Swarm Evolution (SwarmRSI)

**Purpose:** see SwarmRSI's three mechanisms run, then test each one, and the paper's experience claim, with ablations and a control.
**Run:** `python labs/g03_scaling_coordination/lab10_11_swarm/run.py` (≈50 s) · **Evidence:** [`results/logs/lab10_11.txt`](../../../results/logs/lab10_11.txt)

## Walkthrough of `swarm_rsi.py`

* **Agents** start with five research directions (`ORIENTATIONS`): `peak_manager`, `solar_saver`, `arbitrageur`, `event_hunter`, `generalist`. A hypothesis is a weight vector over rule templates (`tmpl_w`) and edit operators (`op_w`); an implementation is the agent's policy program.
* **1. Feedback-guided mutation.** `Agent.sample_templates()` draws templates by `own weight × (1 + λ · board[template])`. The **board** is shared: any agent's success with a template raises every agent's chance of trying it. Each agent's own weights are multiplied up after success and down after failure.
* **2. Complementary recombination.** For each pair, complementarity = mean(max of the two per-day scores) − max of their means, i.e. how much an oracle switch between them would gain. The most complementary pair is crossed three ways: `merge(A,B)`, `merge(B,A)`, and **`gate(A,B,f,m)`** (A's rules on days with `day_surplus_fc ≤ m` or `day_peak ≤ m`, B's rules otherwise, with the split found by `best_gate`). A child that beats both parents replaces the weakest agent.
* **3. Recursive evolution.** An agent stagnant for 3 generations gets a mutated copy of the leader's *hypothesis* but keeps its own *implementation*, which preserves behavioural diversity.
* **Experience.** `archive` holds successful edits plus the final best program; `run_swarm(experience=...)` warm-starts the board and seeds one agent with the retained program.

## What we observed

**A. Ablations (10 seeds, 7,200 day-sims)**

| variant | TOU | critical_peak | wholesale |
|---|---|---|---|
| SwarmRSI (full) | **0.937** | 0.945 | 0.757 |
| no recombination | 0.915 | 0.945 | 0.759 |
| no feedback board | 0.911 | 0.956 | 0.758 |
| no hypothesis evolution | 0.933 | 0.949 | 0.764 |
| single agent, same budget | 0.932 | 0.950 | **0.796** |

On TOU, where agents specialise on different day types, recombination (+0.022) and the feedback board (+0.026) matter. On critical-peak and wholesale they don't, and on wholesale a single agent is clearly better.

**B. Budget curve (swarm − single):** TOU +0.029 at 1,440 day-sims, +0.001 at 2,880, +0.005 at 7,200; critical-peak +0.007, +0.007, −0.004. **The swarm helps most when the budget is small.**

**C. Experience accumulation** (campaign 1 = TOU with 7,200 day-sims; campaign 2 = 1,440 day-sims):

| campaign 2 | swarm, fresh | swarm + experience | control: single agent + retained program |
|---|---|---|---|
| TOU, new days | 0.928 | 0.948 | 0.950 |
| critical_peak (new tariff) | 0.907 | 0.929 | 0.933 |
| solar_glut (new tariff) | 0.964 | **0.966** | 0.941 |

Experience improves later campaigns, as the paper claims. **The control shows why:** on two of three targets, a single agent handed only the retained *program* does as well. The swarm's accumulated *hypotheses* add value beyond the program only on solar_glut, where the retained TOU program transfers poorly and the swarm needs diversity to adapt.

## Quiz-style questions

* **Is "swarm diversity" good for its own sake?** No. Lab 10.17 shows it stays around 0.9 for the whole run, yet recombination succeeds only 2–6 times per run. Diversity is a precondition for recombination, not a guarantee of it.
* **What would refute the experience claim?** The control matching the swarm on *every* target. It didn't, but it came close on two.

Next: [Group 04 · Adaptive Workflows](../../g04_adaptive_workflows/README.md)
