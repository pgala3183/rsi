"""SwarmRSI (arXiv:2610.20431), adapted to GridWise.

Paper mechanism (from the abstract):
  * Feedback-Guided Swarm Mutation   -- agents refine hypotheses and
    implementations using feedback from OTHER agents.
  * Complementary Swarm Recombination -- reuse retained hypotheses and program
    components across agents.
  * Recursive Swarm Evolution         -- both mechanisms in one feedback loop;
    accumulated research experience improves later experiments.

Here each agent is (hypothesis, implementation):
  hypothesis     = weights over rule templates (what it believes matters) and
                   over edit operators (how it likes to experiment)
  implementation = its current battery policy program
Each generation:
  1. mutation      : every agent proposes edits drawn from its hypothesis, mixed
                     with the swarm's shared feedback board; keeps the best if it improves
  2. recombination : the most COMPLEMENTARY pair (good on different days) is
                     crossed by merging their rules; the child replaces the weakest agent
  3. evolution     : hypotheses are re-weighted by evidence; a stagnant agent
                     restarts from a mutated copy of the best agent's hypothesis
  The retained archive of successful edits is the swarm's "experience".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..edits import OPERATORS, TEMPLATES, propose
from ..evaluate import Evaluator, Score
from ..policy import Policy

OP_PRIOR = {o: 1.0 for o in OPERATORS}
OP_PRIOR.update({"add_rule": 3.0, "tweak_threshold": 2.0})   # shared by the swarm and the single-agent baseline

ORIENTATIONS = {   # initial research directions of the five agents
    "peak_manager": ("peak_cover", "hold_for_peak", "peak_dump"),
    "solar_saver": ("store_surplus", "hold_for_peak"),
    "arbitrageur": ("cheap_charge", "cloudy_charge", "need_charge", "cheap_now"),
    "event_hunter": ("spike_dump", "peak_dump"),
    "generalist": tuple(TEMPLATES),
}


@dataclass
class Agent:
    name: str
    tmpl_w: Dict[str, float]
    op_w: Dict[str, float]
    policy: Policy
    score: Score
    stagnant: int = 0
    history: List[float] = field(default_factory=list)

    def sample_templates(self, rng, board: Dict[str, float], lam: float, k: int = 3):
        names = list(self.tmpl_w)
        w = np.array([self.tmpl_w[n] * (1 + lam * board.get(n, 0.0)) for n in names])
        return list(rng.choice(names, size=min(k, len(names)), replace=False, p=w / w.sum()))


def _norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


def merge(a: Policy, b: Policy, a_first: bool = True, max_rules: int = 8) -> Policy:
    """Complementary recombination: keep both parents' rules (deduplicated)."""
    first, second = (a, b) if a_first else (b, a)
    rules, seen = [], set()
    for r in first.rules + second.rules:
        if r not in seen:
            rules.append(r); seen.add(r)
    return Policy(tuple(rules[:max_rules]), first.default)


def gate(a: Policy, b: Policy, feature: str, m: float, max_rules: int = 8) -> Optional[Policy]:
    """Context-gated recombination: A's rules on days with feature <= m, B's on the others."""
    from ..policy import Cond, Rule, snap
    lo, hi = Cond(feature, "<=", snap(feature, m)), Cond(feature, ">=", snap(feature, m))
    ga = [Rule(r.conds + (lo,), r.action) for r in a.rules if len(r.conds) < 3]
    gb = [Rule(r.conds + (hi,), r.action) for r in b.rules if len(r.conds) < 3]
    rules = ga + gb
    if b.default != a.default:
        rules.append(Rule((hi,), b.default))
    if not rules or len(rules) > max_rules:
        return None
    return Policy(tuple(rules), a.default)


def best_gate(a_days: np.ndarray, b_days: np.ndarray, ctx: Dict[str, np.ndarray]):
    """Find the day-context split where A wins below and B wins above."""
    best = None
    for f, x in ctx.items():
        for q in (0.25, 0.4, 0.5, 0.6, 0.75):
            m = float(np.quantile(x, q)); lo = x <= m
            gain = np.where(lo, a_days, b_days).mean()
            if best is None or gain > best[0]:
                best = (gain, f, m)
    return best


def rule_jaccard(a: Policy, b: Policy) -> float:
    A, B = set(a.rules), set(b.rules)
    if not A and not B:
        return 0.0
    return 1.0 - len(A & B) / len(A | B)


def diversity(agents: List[Agent]) -> float:
    d = [rule_jaccard(x.policy, y.policy) for i, x in enumerate(agents) for y in agents[i + 1:]]
    return float(np.mean(d)) if d else 0.0


@dataclass
class SwarmResult:
    best: Agent
    agents: List[Agent]
    curve: List[tuple]            # (day_sims, best steering capture, diversity)
    archive: List[dict]           # retained successful edits = experience
    events: List[str]


def run_swarm(ev: Evaluator, root: Policy, rng: np.random.Generator, budget: int,
              n_agents: int = 5, edits_per_agent: int = 3, feedback: bool = True,
              recombination: bool = True, evolution: bool = True, lam: float = 2.0,
              experience: Optional[List[dict]] = None, verbose: bool = False) -> SwarmResult:
    names = list(ORIENTATIONS)[:n_agents]
    base = ev.score(root)
    agents = []
    for nm in names:
        tw = {t: (1.0 if t in ORIENTATIONS[nm] else 0.15) for t in TEMPLATES}
        agents.append(Agent(nm, _norm(tw), dict(OP_PRIOR), root, base))
    board: Dict[str, float] = {}               # shared feedback: template -> success evidence
    archive: List[dict] = list(experience or [])
    events: List[str] = []
    for e in archive:                          # warm start from retained experience
        if e.get("template"):
            board[e["template"]] = board.get(e["template"], 0.0) + 0.5
    donors = [e["policy"] for e in archive if e.get("policy") is not None]
    if donors:                                 # retained program components join the swarm
        p0 = donors[-1]
        agents[-1].policy, agents[-1].score = p0, ev.score(p0)
        events.append(f"experience: {agents[-1].name} starts from a retained program ({agents[-1].score.mean:.3f})")
    curve = [(ev.day_sims, base.mean, 0.0)]
    from ..diagnose import day_context
    ctx = day_context(ev.task.steering)
    gen = 0
    while ev.day_sims < budget:
        gen += 1
        # 1) feedback-guided mutation --------------------------------------
        for ag in agents:
            if ev.day_sims >= budget:
                break
            tmpls = ag.sample_templates(rng, board if feedback else {}, lam)
            best = None
            for j in range(edits_per_agent):
                child, info = propose(ag.policy, rng, op_weights=ag.op_w, templates=[tmpls[j % len(tmpls)]])
                s = ev.score(child)
                if best is None or s.mean > best[1].mean:
                    best = (child, s, info)
                tname = info.detail if info.operator == "add_rule" else None
                gain = s.mean - ag.score.mean
                # evidence for the agent's own hypothesis
                ag.op_w[info.operator] *= 1.15 if gain > 1e-4 else 0.95
                if tname in ag.tmpl_w:
                    ag.tmpl_w[tname] *= 1.3 if gain > 1e-4 else 0.9
                if feedback and tname and gain > 1e-4:
                    board[tname] = board.get(tname, 0.0) + gain * 10
            ag.tmpl_w = _norm(ag.tmpl_w)
            child, s, info = best
            if s.mean > ag.score.mean + 1e-4:
                archive.append({"agent": ag.name, "template": info.detail if info.operator == "add_rule" else info.operator,
                                "operator": info.operator, "gain": s.mean - ag.score.mean, "gen": gen})
                ag.policy, ag.score, ag.stagnant = child, s, 0
            else:
                ag.stagnant += 1
        # 2) complementary recombination ----------------------------------
        if recombination and ev.day_sims < budget:
            best_pair = None
            for i in range(len(agents)):
                for k in range(i + 1, len(agents)):
                    a, b = agents[i], agents[k]
                    comp = np.maximum(a.score.per_day, b.score.per_day).mean() - max(a.score.mean, b.score.mean)
                    if a.policy.key != b.policy.key and (best_pair is None or comp > best_pair[0]):
                        best_pair = (comp, i, k)
            if best_pair and best_pair[0] > 0.005:
                comp, i, k = best_pair
                a, b = agents[i], agents[k]
                kids = [merge(a.policy, b.policy, True), merge(a.policy, b.policy, False)]
                g1 = best_gate(a.score.per_day, b.score.per_day, ctx)
                g2 = best_gate(b.score.per_day, a.score.per_day, ctx)
                for (gain, f, m), (x, y) in ((g1, (a, b)), (g2, (b, a))):
                    child = gate(x.policy, y.policy, f, m)
                    if child is not None:
                        kids.append(child)
                existing = {x.policy.key for x in agents}
                scored = [(ev.score(c), c) for c in kids if c.key not in existing]
                if scored:
                    s, c = max(scored, key=lambda x: x[0].mean)
                    if s.mean > max(a.score.mean, b.score.mean) + 1e-4:
                        weakest = min(agents, key=lambda x: x.score.mean)
                        msg = (f"gen {gen}: recombined {a.name} x {b.name} (complementarity {comp:.3f}) -> "
                               f"{s.mean:.3f}, replaces {weakest.name} ({weakest.score.mean:.3f})")
                        events.append(msg)
                        weakest.policy, weakest.score, weakest.stagnant = c, s, 0
                        weakest.tmpl_w = _norm({t: (a.tmpl_w[t] + b.tmpl_w[t]) / 2 for t in TEMPLATES})
                        archive.append({"agent": weakest.name, "template": "recombination", "operator": "merge",
                                        "gain": s.mean - max(a.score.mean, b.score.mean), "gen": gen})
        # 3) recursive evolution of hypotheses ----------------------------
        if evolution:
            leader = max(agents, key=lambda x: x.score.mean)
            for ag in agents:
                if ag is not leader and ag.stagnant >= 3:
                    noise = {t: leader.tmpl_w[t] * float(rng.lognormal(0, 0.5)) for t in TEMPLATES}
                    ag.tmpl_w = _norm(noise)       # new hypothesis; the implementation is kept (diversity)
                    ag.op_w = {o: float(np.sqrt(ag.op_w[o] * leader.op_w[o])) for o in OPERATORS}
                    ag.stagnant = 0
                    events.append(f"gen {gen}: {ag.name} stagnated -> adopts a mutated copy of {leader.name}'s hypothesis")
        for ag in agents:
            ag.history.append(ag.score.mean)
        curve.append((ev.day_sims, max(a.score.mean for a in agents), diversity(agents)))
    best = max(agents, key=lambda x: x.score.mean)
    archive.append({"agent": best.name, "template": None, "operator": "retained_program",
                    "gain": 0.0, "gen": gen, "policy": best.policy})
    return SwarmResult(best, agents, curve, archive, events)


def run_single(ev: Evaluator, root: Policy, rng: np.random.Generator, budget: int, edits_per_step: int = 3):
    """Single-agent hill climber with the same proposer and budget (baseline)."""
    pol, sc = root, ev.score(root)
    while ev.day_sims < budget:
        best = None
        for _ in range(edits_per_step):
            child, _ = propose(pol, rng, op_weights=OP_PRIOR)
            s = ev.score(child)
            if best is None or s.mean > best[1].mean:
                best = (child, s)
        if best[1].mean > sc.mean + 1e-4:
            pol, sc = best
    return pol, sc
