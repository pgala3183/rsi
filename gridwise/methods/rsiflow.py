"""RSIFlow (arXiv:2610.21791): an evolving meta-agent that re-routes the
self-improvement workflow toward the current bottleneck, adapted to GridWise.

Workflow graph:   diagnose -> {tune | extend | restructure | prune} -> evaluate -> accept?
* static workflow : visit the branches round-robin (a fixed recipe)
* RSIFlow         : the meta-agent (1) names the bottleneck (largest symptom),
                    (2) routes to the branch with the best upper-confidence
                    credit for THAT bottleneck, (3) evolves the graph itself:
                    prunes branches that never pay off and, when progress
                    stalls, grows a composite branch (extend-then-tune).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..diagnose import symptoms
from ..edits import propose, soft_templates
from ..evaluate import Evaluator
from ..policy import Policy
from .areal_rsi import ORIENT

BRANCHES = {
    "tune": {"tweak_threshold": 1.0},
    "extend": {"add_rule": 1.0},
    "restructure": {"add_condition": 1.0, "swap_rules": 1.0, "flip_op": 0.3},
    "prune": {"delete_rule": 1.0, "drop_condition": 1.0},
}
SYMPTOM_TEMPLATES = {}
for _o in ORIENT:
    for _s in ORIENT[_o]["symptoms"]:
        SYMPTOM_TEMPLATES.setdefault(_s, [])
        SYMPTOM_TEMPLATES[_s] += list(ORIENT[_o]["templates"])


class MetaAgent:
    def __init__(self, branches, c: float = 0.6):
        self.active = list(branches)
        self.stats = defaultdict(lambda: [0, 0.0])      # (bottleneck, branch) -> [visits, total gain]
        self.c = c
        self.events: List[str] = []

    def route(self, bottleneck: str, rng) -> str:
        total = sum(self.stats[(bottleneck, b)][0] for b in self.active) + 1
        def ucb(b):
            n, g = self.stats[(bottleneck, b)]
            return float("inf") if n == 0 else g / n * 100 + self.c * math.sqrt(math.log(total) / n)
        vals = {b: ucb(b) for b in self.active}
        top = max(vals.values())
        return str(rng.choice([b for b, v in vals.items() if v == top]))

    def credit(self, bottleneck, branch, gain):
        st = self.stats[(bottleneck, branch)]
        st[0] += 1; st[1] += max(gain, 0.0)

    def evolve(self, step: int, stalled: int):
        # prune a branch that has been tried often and never helped any bottleneck
        for b in list(self.active):
            n = sum(v[0] for k, v in self.stats.items() if k[1] == b)
            g = sum(v[1] for k, v in self.stats.items() if k[1] == b)
            if n >= 12 and g == 0 and len(self.active) > 2:
                self.active.remove(b)
                self.events.append(f"step {step}: pruned branch '{b}' ({n} visits, no gain)")
        if stalled >= 10 and "extend+tune" not in self.active:
            self.active.append("extend+tune")
            self.events.append(f"step {step}: progress stalled -> grew composite branch 'extend+tune'")


def _branch_edit(pol: Policy, branch: str, bottleneck: str, rng):
    if branch == "extend":
        return propose(pol, rng, op_weights=BRANCHES["extend"], templates=soft_templates(SYMPTOM_TEMPLATES[bottleneck]))[0]
    return propose(pol, rng, op_weights=BRANCHES[branch])[0]


def run_flow(ev: Evaluator, root: Policy, rng, budget: int, adaptive: bool = True, per_step: int = 2):
    pol, sc = root, ev.score(root)
    meta = MetaAgent(BRANCHES)
    order = list(BRANCHES)
    step = stalled = 0
    route_log = []
    while ev.day_sims < budget:
        step += 1
        _, sim = ev.trace(pol, purpose="diagnose")
        bottleneck = symptoms(ev.task.steering, sim).top()
        branch = meta.route(bottleneck, rng) if adaptive else order[step % len(order)]
        cands = []
        if branch == "extend+tune":
            base = _branch_edit(pol, "extend", bottleneck, rng)
            cands = [base] + [_branch_edit(base, "tune", bottleneck, rng) for _ in range(per_step)]
        else:
            tmpl = soft_templates(SYMPTOM_TEMPLATES[bottleneck]) if adaptive else None
            for _ in range(per_step):
                if branch == "extend":
                    cands.append(propose(pol, rng, op_weights=BRANCHES["extend"], templates=tmpl)[0])
                else:
                    cands.append(_branch_edit(pol, branch, bottleneck, rng))
        scored = [(ev.score(c, purpose=f"branch:{branch}"), c) for c in cands]
        s, c = max(scored, key=lambda x: x[0].mean)
        gain = s.mean - sc.mean
        meta.credit(bottleneck, branch, gain)
        if gain > 1e-4:
            pol, sc, stalled = c, s, 0
        else:
            stalled += 1
        route_log.append((step, bottleneck, branch, round(gain, 4)))
        if adaptive:
            meta.evolve(step, stalled)
    return pol, sc, meta, route_log
