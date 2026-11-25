# 10.14 · Simulate Self-HPO Alternation

**Purpose:** alternate harness optimisation (H-step) with policy optimisation (P-step), with the policy proposing its own harness.
**Run:** `python labs/g04_adaptive_workflows/lab10_14_self_hpo/run.py` (≈28 s) · **Evidence:** [`results/logs/lab10_14.txt`](../../../results/logs/lab10_14.txt)

## Walkthrough of `self_hpo.py`

* **`Harness`** = `curriculum` (which steering days the P-step trains on) + `guards` (runtime action filters). It is called as `wrapper(obs, actions)` inside `simulate()`, so the deployed system is policy *plus* harness.
* **`Guard`**: "if these conditions hold and the policy chose one of these actions, replace it". `guard_candidates(symptom)` maps each failure symptom to guards a policy would want. For example, `excess_grid_charge` → `no_grid_charge_if_sunny>=v` (`day_surplus_fc ≥ v`: block charge actions).
* **`h_step()`:** simulate the current policy inside the current harness on **validation** days (steering days 32–47, never trained on). Take its top-2 symptoms, try their guards, and keep any that raise validation capture. Then **hard-task identification**: the policy's 12 worst training days plus 8 random ones become the next curriculum.
* **`p_step()`:** 6 proposals, each scored on the curriculum inside the harness; keep improvements.
* **Arms:** `policy_only` (fixed harness), `harness_only` (the seed policy never changes), `joint` (alternate). Plus `data_only` (H-step may change the curriculum only), used by lab 10.16.

## What we observed

TOU, joint run: the first H-step finds no useful guard (validation 0.847). After one P-step, the second H-step diagnoses `excess_grid_charge` and adds `no_grid_charge_if_sunny>=6.0` (validation 0.947). Later P-steps learn cheap-hour and cloudy-day charging inside that guard. Final held-out: **0.920**.

| tariff | root | policy-only | harness-only | joint | synergy = joint gain − (P gain + H gain) |
|---|---|---|---|---|---|
| tou | 0.712 | 0.884 | 0.712 | **0.916** | **+0.032** |
| critical_peak | 0.790 | **0.952** | 0.843 | 0.936 | −0.069 |
| wholesale | 0.501 | **0.780** | 0.633 | 0.714 | −0.198 |
| vpp_events | 0.841 | 0.962 | 0.962 | 0.961 | −0.121 |

## Interpretation

* **The synergy claim holds on TOU only.** There, a guard alone does nothing for the seed policy (harness-only +0.000), but it lets the P-step learn aggressive charging rules safely. That's the paper's "harness targets the policy's failure modes".
* **Elsewhere the additive baseline double-counts.** On VPP, one guard alone fixes the main failure (+0.121) and so does policy-only, so "joint − (P + H)" is strongly negative even though joint is as good as either. Synergy defined additively is a poor metric when both arms find *the same* fix.
* **Overfitting:** on wholesale, joint keeps guards chosen on 16 validation days and loses 0.066 to policy-only on held-out.

## Quiz-style question

**Why must the H-step use days the P-step never trains on?** Otherwise a guard is rewarded for patching the curriculum's quirks. Even with separate validation days, 16 of them were too few on wholesale.

Next: [Group 05 · Environments & metrics](../../g05_environments_metrics/README.md)
