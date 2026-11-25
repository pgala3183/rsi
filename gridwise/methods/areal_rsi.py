"""AReaL-RSI (arXiv:2610.18083) leader-follower coordination, adapted to GridWise.

Mechanism (course summary): a leader agent picks research orientations for
specialist researchers; a CONDITIONAL memory separates supported findings
from unresolved problems; after each campaign an evidence-based review revises
orientation priorities. No model weights change -- the team organisation does.

Here
  orientation  = a family of rule templates aimed at one failure symptom
  researcher   = a short hill-climb restricted to its orientation's templates
  memory entry = {orientation, problem, status: supported | unresolved, evidence}
  leader       = ranks orientations by (unresolved $-symptom) x (review priority)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..diagnose import symptoms
from ..edits import propose, soft_templates
from ..evaluate import Evaluator
from ..policy import Policy

ORIENT = {
    "peak": dict(templates=("peak_cover", "hold_for_peak", "peak_dump"), symptoms=("uncovered_peak", "early_discharge")),
    "solar": dict(templates=("store_surplus", "hold_for_peak"), symptoms=("wasted_solar",)),
    "arbitrage": dict(templates=("cheap_charge", "cloudy_charge", "need_charge", "cheap_now"),
                      symptoms=("missed_grid_charge", "excess_grid_charge")),
    "events": dict(templates=("spike_dump", "peak_dump"), symptoms=("uncovered_peak",)),
}
OPW = {"add_rule": 3.0, "tweak_threshold": 2.0, "add_condition": 1.0, "drop_condition": 0.5, "delete_rule": 0.5,
       "swap_rules": 0.5, "flip_op": 0.2, "change_action": 0.5, "change_default": 0.3}


@dataclass
class ConditionalMemory:
    entries: List[dict] = field(default_factory=list)

    def store(self, campaign, orientation, problem, resolved, evidence, policy=None):
        self.entries.append(dict(campaign=campaign, orientation=orientation, problem=problem,
                                 status="supported" if resolved else "unresolved", evidence=evidence,
                                 policy=policy))

    def supported(self):
        return [e for e in self.entries if e["status"] == "supported"]

    def unresolved(self):
        solved = {e["problem"] for e in self.supported()}
        return [e for e in self.entries if e["status"] == "unresolved" and e["problem"] not in solved]


def investigate(ev: Evaluator, start: Policy, orientation: str, rng, steps: int):
    pol, sc = start, ev.score(start, purpose=f"researcher:{orientation}")
    for _ in range(steps):
        child, _ = propose(pol, rng, op_weights=OPW, templates=soft_templates(ORIENT[orientation]["templates"]))
        s = ev.score(child, purpose=f"researcher:{orientation}")
        if s.mean > sc.mean + 1e-4:
            pol, sc = child, s
    return pol, sc


def run_team(ev: Evaluator, root: Policy, rng, budget: int, steps: int = 8, n_researchers: int = 2,
             conditional: bool = True, verbose: bool = False):
    """Leader + researchers sharing memory.

    conditional=True : only SUPPORTED findings (validated improvements) are shared and
                       built on; unresolved problems are tracked separately.
    conditional=False: every researcher's latest result is shared as if it were a finding
                       (no resolved/unresolved distinction) -- the ablation.
    """
    mem = ConditionalMemory()
    priority = {o: 1.0 for o in ORIENT}
    best, best_s = root, ev.score(root)
    log, campaign = [], 0
    while ev.day_sims < budget:
        campaign += 1
        s, sim = ev.trace(best, purpose="leader")
        sym = symptoms(ev.task.steering, sim).total
        need = {o: priority[o] * sum(sym[k] for k in ORIENT[o]["symptoms"]) + 1e-6 for o in ORIENT}
        chosen = sorted(need, key=need.get, reverse=True)[:n_researchers]
        results = []
        shared = best
        for o in chosen:
            start = best if conditional else shared         # conditional: build on supported findings only
            pol, sc = investigate(ev, start, o, rng, steps)
            if not conditional:
                shared = pol                                 # unconditional: pass on whatever came back
            gain = sc.mean - best_s.mean
            prob = max(ORIENT[o]["symptoms"], key=lambda k: sym[k])
            mem.store(campaign, o, prob, gain > 1e-3, round(gain, 4), pol if gain > 1e-3 else None)
            results.append((sc.mean, o, pol, gain))
            if ev.day_sims >= budget:
                break
        # evidence-based review: integrate, then revise priorities
        for score, o, pol, gain in sorted(results, key=lambda r: -r[0]):
            if conditional:
                if score > best_s.mean + 1e-4:
                    best, best_s = pol, ev.score(pol)
            priority[o] = min(priority[o] * 1.2, 3.0) if gain > 1e-3 else priority[o] * 0.5
        if not conditional:
            best, best_s = shared, ev.score(shared)          # the team continues from the last shared result
        line = (f"campaign {campaign}: leader chose {chosen} | gains "
                + ", ".join(f"{o}={g:+.3f}" for _, o, _, g in results) + f" | best={best_s.mean:.3f}"
                + f" | unresolved={sorted({e['problem'] for e in mem.unresolved()})}")
        log.append(line)
    return best, best_s, mem, log


def run_independent(ev: Evaluator, root: Policy, rng, budget: int, n_researchers: int = 2, steps: int = 8):
    """Baseline: researchers explore independently (all templates), no sharing; best one wins."""
    pols = [(root, ev.score(root)) for _ in range(n_researchers)]
    while ev.day_sims < budget:
        for k in range(n_researchers):
            pol, sc = pols[k]
            for _ in range(steps):
                child, _ = propose(pol, rng, op_weights=OPW)
                s = ev.score(child, purpose="independent")
                if s.mean > sc.mean + 1e-4:
                    pol, sc = child, s
                if ev.day_sims >= budget:
                    break
            pols[k] = (pol, sc)
    return max(pols, key=lambda x: x[1].mean)
