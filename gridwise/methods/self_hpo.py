"""Self-HPO (arXiv:2610.22791): alternate harness optimisation and policy
optimisation, with the policy proposing its own harness. Adapted to GridWise.

Harness = what surrounds the policy at training and run time:
  * curriculum : which training days the P-step optimises on
  * guards     : runtime action filters (a safety/heuristic wrapper around the policy)
Policy  = the rule program.

H-step : the current policy analyses its failures on validation days and
         proposes harness candidates (guards targeted at its symptoms; a
         curriculum of its hardest training days). The best candidate by
         validation score is kept.
P-step : hill-climb the policy on the harness's curriculum, running INSIDE the harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..diagnose import symptoms
from ..eaas import BudgetExceeded
from ..edits import propose
from ..env import ACTION_ID
from ..evaluate import Evaluator
from ..policy import Policy

DISCHARGING = [ACTION_ID[a] for a in ("cover_load", "discharge_max", "self_consume")]
CHARGING = [ACTION_ID[a] for a in ("charge_max", "charge_half")]
OPW = {"add_rule": 3.0, "tweak_threshold": 2.0, "add_condition": 1.0, "drop_condition": 0.5, "delete_rule": 0.5,
       "swap_rules": 0.5, "flip_op": 0.2, "change_action": 0.5, "change_default": 0.3}


@dataclass(frozen=True)
class Guard:
    name: str
    conds: Tuple[Tuple[str, str, float], ...]
    forbid: Tuple[int, ...]      # action ids that are overridden (empty = all)
    replace: int

    def apply(self, obs, a):
        m = np.ones(obs.n, bool)
        for f, op, t in self.conds:
            v = obs[f]
            m &= (v >= t) if op == ">=" else (v <= t)
        if self.forbid:
            m &= np.isin(a, self.forbid)
        return np.where(m, self.replace, a)


@dataclass(frozen=True)
class Harness:
    curriculum: Tuple[int, ...]
    guards: Tuple[Guard, ...] = ()

    @property
    def key(self):
        return ("H", self.guards)          # the wrapper part (curriculum only affects training)

    def __call__(self, obs, a):
        for g in self.guards:
            a = g.apply(obs, a)
        return a

    def describe(self):
        gs = ", ".join(g.name for g in self.guards) or "none"
        return f"curriculum={len(self.curriculum)} days, guards=[{gs}]"


def guard_candidates(symptom: str):
    """The policy's own failure analysis -> runtime guards it would like to have."""
    S = ACTION_ID["self_consume"]; A = ACTION_ID["absorb_surplus"]
    if symptom == "early_discharge":
        return [Guard(f"hold_if_peak>={v}", (("peak_ahead", ">=", v), ("price_rank", "<=", 0.6)), tuple(DISCHARGING), A)
                for v in (0.35, 0.5, 0.8)]
    if symptom == "wasted_solar":
        return [Guard(f"absorb_if_surplus>={v}", (("surplus_fc", ">=", v),), (ACTION_ID["idle"], ACTION_ID["cover_load"]), A)
                for v in (0.0, 0.5, 1.0)]
    if symptom == "excess_grid_charge":
        return [Guard(f"no_grid_charge_if_sunny>={v}", (("day_surplus_fc", ">=", v),), tuple(CHARGING), S)
                for v in (6.0, 12.0, 20.0)]
    if symptom == "missed_grid_charge":
        return [Guard(f"charge_if_cloudy<={v}", (("day_surplus_fc", "<=", v), ("price_rank", "<=", 0.15), ("soc", "<=", 0.85)),
                      (), ACTION_ID["charge_max"]) for v in (4.0, 8.0, 14.0)]
    if symptom == "uncovered_peak":
        return [Guard(f"cover_if_rank>={v}", (("price_rank", ">=", v),), (ACTION_ID["idle"], A), ACTION_ID["cover_load"])
                for v in (0.7, 0.8, 0.9)] + \
               [Guard(f"dump_if_export>={v}", (("export_price", ">=", v),), (), ACTION_ID["discharge_max"]) for v in (0.4, 0.6)]
    return []


@dataclass
class HPOState:
    policy: Policy
    harness: Harness
    log: list = field(default_factory=list)


def p_step(ev: Evaluator, st: HPOState, rng, n: int):
    cur = ev.score(st.policy, "steering", idx=st.harness.curriculum, wrapper=st.harness, purpose="P-step")
    for _ in range(n):
        child, _ = propose(st.policy, rng, op_weights=OPW)
        s = ev.score(child, "steering", idx=st.harness.curriculum, wrapper=st.harness, purpose="P-step")
        if s.mean > cur.mean + 1e-4:
            st.policy, cur = child, s
    st.log.append(f"P-step: policy now {len(st.policy.rules)} rules, curriculum capture {cur.mean:.3f}")


def h_step(ev: Evaluator, st: HPOState, train_pool, val_idx, rng, guards: bool = True):
    val = ev.task.steering.subset(np.asarray(val_idx))
    from ..env import simulate
    sim = simulate(val, st.policy.act, record=True, wrapper=st.harness)
    ev.ledger["steering"] += val.n                       # the failure analysis is paid for
    sym = symptoms(val, sim)
    ranked = sorted(sym.total, key=sym.total.get, reverse=True)[:2]
    best_h, best_s = st.harness, ev.score(st.policy, "steering", idx=val_idx, wrapper=st.harness, purpose="H-step")
    for symp in (ranked if guards else []):
        for g in guard_candidates(symp):
            if any(x.name.split("_if_")[0] == g.name.split("_if_")[0] for x in best_h.guards):
                continue
            cand = Harness(best_h.curriculum, best_h.guards + (g,))
            s = ev.score(st.policy, "steering", idx=val_idx, wrapper=cand, purpose="H-step")
            if s.mean > best_s.mean + 1e-4:
                best_h, best_s = cand, s
    # hard-task identification: the policy's worst training days become the curriculum core
    tr = ev.score(st.policy, "steering", idx=train_pool, wrapper=best_h, purpose="H-step")
    hard = [train_pool[i] for i in np.argsort(tr.per_day)[:12]]
    rest = [d for d in train_pool if d not in hard]
    extra = list(rng.choice(rest, size=min(8, len(rest)), replace=False))
    st.harness = Harness(tuple(sorted(int(x) for x in hard + extra)), best_h.guards)
    st.log.append(f"H-step: symptoms {ranked} -> {st.harness.describe()} (validation {best_s.mean:.3f})")


def run_self_hpo(ev: Evaluator, root: Policy, rng, budget: int, mode: str = "joint", p_iters: int = 6):
    """mode: 'joint' (alternate H/P), 'policy_only' (fixed harness), 'harness_only' (fixed policy),
    'data_only' (H-step may only change the curriculum -- RSIGym's data track)."""
    n = ev.task.steering.n
    train_pool = list(range(0, 32)); val_idx = list(range(32, n))
    st = HPOState(root, Harness(tuple(train_pool[:16])))
    rounds = 0
    while ev.day_sims < budget:
        rounds += 1
        before = ev.day_sims
        try:
            if mode in ("joint", "harness_only", "data_only"):
                h_step(ev, st, train_pool, val_idx, rng, guards=mode != "data_only")
            if mode in ("joint", "policy_only", "data_only"):
                p_step(ev, st, rng, p_iters)
        except BudgetExceeded:            # a metered environment stopped us: keep the best state so far
            st.log.append(f"stopped by the budget meter after {ev.day_sims} day-sims")
            break
        if ev.day_sims == before or rounds > 200:
            break
    return st
