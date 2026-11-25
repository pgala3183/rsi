"""SO-RSI: second-order diagnostic inquiry (arXiv:2610.05701), adapted to GridWise.

Paper mechanism (from the abstract):
  First-order RSI patches surface parameters in response to failure symptoms
  and thrashes. SO-RSI passively monitors execution traces for three structural
  anomalies -- recurrence, opposing edits, expectation mismatch -- runs
  lightweight diagnostic probes, and keeps a persistent inquiry memory across
  rounds so that edits address mechanisms, not symptoms.

Here
  * `FirstOrderImprover` ("Naive RSI") reads the oracle-relative symptoms of the
    incumbent policy and applies the matching patch from a fixed
    symptom -> (slot, direction) table. It keeps a patch iff the steering score rises.
  * `SecondOrderImprover` ("SO-RSI") runs the same loop, but after every attempt
    it checks its own trace for the three anomalies. Each anomaly triggers a
    probe, paid for out of the SAME evaluation budget:
        recurrence -> sweep + lever probe : is this threshold even the binding lever?
        opposing   -> split probe         : do different days want opposite moves?
        mismatch   -> attribution probe   : where did the money actually go?
    Findings go to `InquiryMemory`, which edits the improver's own workflow:
    it re-maps symptoms to the slots that actually cause them, bans exhausted
    levers, remembers guards, and shrinks step sizes after oscillation.
    Those are changes to the *improvement procedure*, not to the policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..diagnose import SYMPTOMS, day_context, symptoms
from ..env import CAP, Obs
from ..evaluate import Evaluator
from ..policy import C, FEATURES, Cond, Policy, R, Rule, snap

# ------------------------------------------------------------------ slots
# A "slot" is a role a rule plays in the controller; first-order patches target slots.
SLOTS = {
    "spike": dict(actions=("discharge_max",), feature="export_price", op=">=", priority=0,
                  template=lambda: R("discharge_max", C("export_price", ">=", 0.60))),
    "absorb": dict(actions=("absorb_surplus",), feature="surplus_fc", op=">=", priority=1,
                   template=lambda: R("absorb_surplus", C("surplus_fc", ">=", 0.0))),
    "peak": dict(actions=("cover_load",), feature="price_rank", op=">=", priority=2,
                 template=lambda: R("cover_load", C("price_rank", ">=", 0.80))),
    "charge": dict(actions=("charge_max", "charge_half"), feature="price_rank", op="<=", priority=3,
                   template=lambda: R("charge_max", C("price_rank", "<=", 0.15), C("soc", "<=", 0.90))),
    "hold": dict(actions=("idle",), feature="peak_ahead", op=">=", priority=4,
                 template=lambda: R("idle", C("peak_ahead", ">=", 0.40), C("price_rank", "<=", 0.60),
                                   C("surplus_fc", "<=", 0.0))),
}
# first-order symptom -> candidate (slot, direction) patches; direction moves the threshold
PATCHES = {
    "uncovered_peak": [("peak", -1), ("spike", -1)],
    "early_discharge": [("hold", -1)],
    "missed_grid_charge": [("charge", +1)],
    "excess_grid_charge": [("charge", -1)],
    "wasted_solar": [("absorb", -1), ("hold", +1)],
}
SWEEP = {"export_price": [0.2, 0.35, 0.5, 0.7, 0.9, 1.2], "price_rank": [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
         "peak_ahead": [0.2, 0.35, 0.5, 0.7, 0.9, 1.2], "surplus_fc": [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]}
GUARD_CANDIDATES = {
    ("peak_ahead", "<="): [0.3, 0.45, 0.6, 0.9],
    ("surplus_left_fc", "<="): [4.0, 8.0, 12.0, 16.0],
    ("soc", ">="): [0.1, 0.25, 0.4, 0.55],
}


def slot_of(rule: Rule) -> Optional[str]:
    for name, s in SLOTS.items():
        if rule.action in s["actions"] and any(c.feature == s["feature"] and c.op == s["op"] for c in rule.conds):
            return name
    return None


def find_slot(p: Policy, slot: str) -> Optional[int]:
    return next((i for i, r in enumerate(p.rules) if slot_of(r) == slot), None)


def insert_pos(p: Policy, slot: str) -> int:
    pr = SLOTS[slot]["priority"]
    for i, r in enumerate(p.rules):
        s = slot_of(r)
        if s is not None and SLOTS[s]["priority"] > pr:
            return i
    return len(p.rules)


def with_slot_rule(p: Policy, slot: str, rule: Optional[Rule] = None) -> Tuple[Policy, int]:
    """Return p with the slot's rule present (created from its template if absent)."""
    i = find_slot(p, slot)
    if i is not None:
        return p, i
    rules = list(p.rules)
    i = insert_pos(p, slot)
    rules.insert(i, rule or SLOTS[slot]["template"]())
    return p.with_rules(rules), i


def _with_rule(p: Policy, i: int, rule: Rule) -> Policy:
    rules = list(p.rules); rules[i] = rule
    return p.with_rules(rules)


def _move_thr(rule: Rule, feature: str, op: str, delta: float) -> Rule:
    return Rule(tuple(Cond(c.feature, c.op, snap(c.feature, c.thr + delta)) if (c.feature, c.op) == (feature, op) else c
                      for c in rule.conds), rule.action)


def _set_thr(rule: Rule, feature: str, op: str, value: float) -> Rule:
    return Rule(tuple(Cond(c.feature, c.op, snap(c.feature, value)) if (c.feature, c.op) == (feature, op) else c
                      for c in rule.conds), rule.action)


def _add_guard(rule: Rule, feature: str, op: str, thr: float) -> Rule:
    conds = tuple(c for c in rule.conds if (c.feature, c.op) != (feature, op)) + (Cond(feature, op, snap(feature, thr)),)
    return Rule(conds[:3], rule.action)


# ------------------------------------------------------------------ trace
@dataclass
class Attempt:
    i: int
    symptom: str
    slot: str
    direction: int
    kind: str                 # create / tweak (first-order) | sweep / split / guard / reorder / reroute (probes)
    accepted: bool
    d_score: float
    d_symptom: float
    day_sims: int
    note: str = ""


@dataclass
class InquiryMemory:
    """Persistent causal evidence + the workflow edits it licenses."""
    entries: List[dict] = field(default_factory=list)
    banned: Dict[str, int] = field(default_factory=dict)        # slot -> attempt index until which tweaks are banned
    guards: Dict[str, list] = field(default_factory=dict)       # slot -> learned guard [feature, op, thr]
    fine_steps: Dict[str, int] = field(default_factory=dict)    # slot -> max step after oscillation
    remap: Dict[str, list] = field(default_factory=dict)        # symptom -> [slot, direction] (causal re-routing)

    def record(self, **kw):
        self.entries.append(kw)

    def to_json(self) -> dict:
        return {"entries": self.entries, "guards": self.guards, "fine_steps": self.fine_steps, "remap": self.remap}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=float)

    @classmethod
    def load(cls, path) -> "InquiryMemory":
        with open(path) as f:
            d = json.load(f)
        # bans refer to attempt indices of one run, so they do not transfer; causal findings do
        return cls(d["entries"], {}, d["guards"], d["fine_steps"], d["remap"])


# --------------------------------------------------------------- improvers
class FirstOrderImprover:
    """Naive RSI: symptom -> patch -> keep if the steering score improves."""
    name = "naive"

    def __init__(self, ev: Evaluator, root: Policy, rng: np.random.Generator, budget: int,
                 memory: Optional[InquiryMemory] = None, verbose: bool = False, max_attempts: int = 400):
        self.ev, self.rng, self.budget, self.verbose = ev, rng, budget, verbose
        self.memory = memory or InquiryMemory()
        self.b = ev.task.steering
        self.inc = root
        self.inc_s, self.inc_sim = ev.trace(root)
        self.trace: List[Attempt] = []
        self.history = [(ev.day_sims, self.inc_s.mean)]
        self.log: List[str] = []
        self.max_attempts = max_attempts

    def pick_symptom(self, sym) -> str:
        if self.rng.random() < 0.5:
            return sym.top()
        w = np.array([max(sym.total[k], 1e-9) for k in SYMPTOMS])
        return SYMPTOMS[int(self.rng.choice(len(SYMPTOMS), p=w / w.sum()))]

    def patch_options(self, symptom: str):
        return list(PATCHES[symptom])

    def patch(self, slot: str, direction: int) -> Tuple[Policy, str]:
        if find_slot(self.inc, slot) is None:
            rule = SLOTS[slot]["template"]()
            g = self.memory.guards.get(slot)
            if g:
                rule = _add_guard(rule, *g)
            p, _ = with_slot_rule(self.inc, slot, rule)
            return p, "create"
        i = find_slot(self.inc, slot)
        s = SLOTS[slot]
        k = int(self.rng.choice((1, 2, 4)[: self.memory.fine_steps.get(slot, 3)]))
        rule = _move_thr(self.inc.rules[i], s["feature"], s["op"], direction * k * FEATURES[s["feature"]][2])
        return _with_rule(self.inc, i, rule), "tweak"

    def run(self) -> Policy:
        stall = 0
        while self.ev.day_sims < self.budget and stall < 50 and len(self.trace) < self.max_attempts:
            sym = symptoms(self.b, self.inc_sim)
            symptom = self.pick_symptom(sym)
            options = self.patch_options(symptom)
            if not options:
                stall += 1
                continue
            slot, direction = options[int(self.rng.integers(len(options)))]
            child, kind = self.patch(slot, direction)
            if child.key == self.inc.key:
                stall += 1
                continue
            stall = 0
            c_s, c_sim = self.ev.trace(child)
            d_score = c_s.mean - self.inc_s.mean
            d_sym = symptoms(self.b, c_sim).total[symptom] - sym.total[symptom]
            att = Attempt(len(self.trace), symptom, slot, direction, kind, d_score > 1e-4, d_score, d_sym,
                          self.ev.day_sims)
            self.trace.append(att)
            self._say(f"[{att.i:3d}] {symptom:18s} -> {kind:6s} {slot:9s} dir={direction:+d}  "
                      f"{'ACCEPT' if att.accepted else 'reject'} dScore={d_score:+.4f} dSymptom={d_sym:+.1f}")
            if att.accepted:
                self._accept(child, c_s, c_sim)
            self.after_attempt(att, child, c_sim)
        return self.inc

    def _accept(self, child, s, sim):
        self.inc, self.inc_s, self.inc_sim = child, s, sim
        self.history.append((self.ev.day_sims, s.mean))

    def _say(self, msg):
        if self.verbose:
            self.log.append(msg)

    def after_attempt(self, att: Attempt, child: Policy, child_sim):
        pass


# ------------------------------------------------------------ anomalies
def detect(trace: List[Attempt], window: int = 8) -> List[str]:
    """Anomalies of the LAST attempt, judged only against the attempts BEFORE it."""
    if not trace:
        return []
    cur, hist = trace[-1], trace[:-1][-window:]
    if cur.kind not in ("create", "tweak"):
        return []
    found = []
    same_fail = [a for a in hist if a.slot == cur.slot and a.direction == cur.direction and not a.accepted
                 and a.kind in ("create", "tweak")]
    if not cur.accepted and len(same_fail) >= 2:
        found.append("recurrence")
    if any(a.slot == cur.slot and a.direction == -cur.direction and a.kind == "tweak" for a in hist[-4:]) \
            and cur.kind == "tweak":
        found.append("opposing")
    # expectation: the targeted symptom shrinks AND the score rises
    if not cur.accepted and abs(cur.d_symptom) > 1e-6 and cur.d_score < -1e-4:
        found.append("mismatch")
    return found


def thrash_stats(trace: List[Attempt]) -> Dict[str, int]:
    """Post-hoc anomaly counts for any improver's trace (compares thrashing)."""
    counts = {"recurrence": 0, "opposing": 0, "mismatch": 0}
    first = [a for a in trace if a.kind in ("create", "tweak")]
    for j in range(1, len(first) + 1):
        for a in detect(first[:j]):
            counts[a] += 1
    longest = cur = 0
    for a in trace:
        cur = 0 if a.accepted else cur + 1
        longest = max(longest, cur)
    counts["longest_zero_progress"] = longest
    counts["attempts"] = len(trace)
    counts["accepted"] = sum(a.accepted for a in trace)
    return counts


class SecondOrderImprover(FirstOrderImprover):
    """SO-RSI: the same first-order loop + anomaly monitor + probes + inquiry memory."""
    name = "so_rsi"
    COOLDOWN = 8

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cool: Dict[Tuple[str, str], int] = {}
        self.probes_run = {"sweep": 0, "split": 0, "attribution": 0}

    # memory edits the first-order workflow ---------------------------------
    def patch_options(self, symptom):
        if symptom in self.memory.remap:
            slot, d = self.memory.remap[symptom]
            opts = [(slot, int(d))]
        else:
            opts = list(PATCHES[symptom])
        return [(s, d) for s, d in opts if self.memory.banned.get(s, -1) < len(self.trace)]

    def after_attempt(self, att, child, child_sim):
        for kind in detect(self.trace):
            key = (kind, att.slot)
            if self.cool.get(key, -99) > att.i - self.COOLDOWN or self.ev.day_sims >= self.budget:
                continue
            self.cool[key] = att.i
            self._say(f"      !! anomaly={kind} on slot={att.slot}")
            if kind == "recurrence":
                self.sweep_probe(att)
            elif kind == "opposing":
                self.split_probe(att)
            else:
                self.attribution_probe(att, child, child_sim)

    def _try(self, cand: Policy, kind: str, att: Attempt, note: str) -> bool:
        s, sim = self.ev.trace(cand, purpose=f"probe:{kind}")
        d = s.mean - self.inc_s.mean
        ok = d > 1e-4
        self.trace.append(Attempt(len(self.trace), att.symptom, att.slot, 0, kind, ok, d, 0.0, self.ev.day_sims, note))
        if ok:
            self._accept(cand, s, sim)
        self._say(f"      -> {kind}: {note}  dScore={d:+.4f} {'ACCEPT' if ok else 'reject'}")
        return ok

    # recurrence --------------------------------------------------------------
    def sweep_probe(self, att):
        """Recurrence: is this threshold the binding lever? Sweep it across its range."""
        self.probes_run["sweep"] += 1
        base, i = with_slot_rule(self.inc, att.slot)
        s = SLOTS[att.slot]
        scores = []
        for v in SWEEP[s["feature"]]:
            cand = _with_rule(base, i, _set_thr(base.rules[i], s["feature"], s["op"], v))
            scores.append((self.ev.score(cand, purpose="probe:sweep").mean, v, cand))
        spread = max(x[0] for x in scores) - min(x[0] for x in scores)
        best = max(scores, key=lambda x: x[0])
        binding = best[0] > self.inc_s.mean + 0.002          # a lever must move the score meaningfully
        # the 1-D axis has now been explored: stop small tweaks on it for a while
        self.memory.banned[att.slot] = len(self.trace) + 2 * self.COOLDOWN
        if binding:
            finding = f"{att.slot}.{s['feature']} IS a lever: best threshold {best[1]} gains {best[0] - self.inc_s.mean:+.3f} (spread {spread:.3f})"
            self.memory.record(anomaly="recurrence", slot=att.slot, probe="sweep", finding=finding, at=att.i)
            self._say(f"      ?? sweep probe: {finding}")
            self._try(best[2], "sweep", att, f"jump {s['feature']} threshold to {best[1]}")
            return
        self._say(f"      ?? sweep probe: {att.slot}.{s['feature']} is NOT a lever (no threshold beats the incumbent;"
                  f" spread={spread:.3f})")
        self.lever_probe(att, base, i)

    def lever_probe(self, att, base: Policy, i: int):
        """Why is the threshold not a lever? Replay the trace: is the rule shadowed, or out of energy?"""
        _, sim = self.ev.trace(base, purpose="probe:lever")
        b, rule = self.b, base.rules[i]
        shadow = wanted = empty = 0
        for t in range(24):
            obs = Obs(b, t, sim.soc[:, t])
            cond = np.ones(b.n, bool)
            for c in rule.conds:
                v = obs[c.feature]
                cond &= (v >= c.thr) if c.op == ">=" else (v <= c.thr)
            fired = base.which(obs)
            wanted += cond.sum()
            shadow += (cond & (fired < i)).sum()
            empty += (cond & (fired == i) & (sim.soc[:, t] < 0.05 * CAP)).sum()
        wanted = max(wanted, 1)
        frac_shadow, frac_empty = shadow / wanted, empty / wanted
        if frac_shadow > 0.3:
            cause = f"shadowed: {frac_shadow:.0%} of the hours it should fire, an earlier rule fires instead"
            self.memory.record(anomaly="recurrence", slot=att.slot, probe="lever", finding=cause, at=att.i)
            self._say(f"      ?? lever probe: {cause}")
            rules = list(base.rules); r = rules.pop(i); rules.insert(max(i - 1, 0), r)
            self._try(self.inc.with_rules(rules), "reorder", att, f"move {att.slot} rule above the rule shadowing it")
        elif frac_empty > 0.3 and att.symptom in ("uncovered_peak",):
            cause = (f"energy-limited: the battery is empty in {frac_empty:.0%} of the hours the rule fires -> "
                     f"the cause of '{att.symptom}' is upstream (energy spent too early), re-route to the HOLD slot")
            self.memory.record(anomaly="recurrence", slot=att.slot, probe="lever", finding=cause, at=att.i)
            self.memory.remap[att.symptom] = ["hold", -1]          # second-order edit of the patch table
            self._say(f"      ?? lever probe: {cause}")
            hold_base, h = with_slot_rule(self.inc, "hold")
            best = max(((self.ev.score(c, purpose="probe:lever").mean, v, c) for v, c in
                        ((v, _with_rule(hold_base, h, _set_thr(hold_base.rules[h], "peak_ahead", ">=", v)))
                         for v in (0.3, 0.45, 0.6, 0.9))), key=lambda x: x[0])
            self._try(best[2], "reroute", att, f"hold energy while peak_ahead >= {best[1]}")
        else:
            cause = f"no lever found (shadowed {frac_shadow:.0%}, empty {frac_empty:.0%}): ban this patch"
            self.memory.record(anomaly="recurrence", slot=att.slot, probe="lever", finding=cause, at=att.i)
            self._say(f"      ?? lever probe: {cause}")

    # opposing ----------------------------------------------------------------
    def split_probe(self, att):
        """Opposing edits: do some days want the threshold up and others down?"""
        self.probes_run["split"] += 1
        i = find_slot(self.inc, att.slot)
        if i is None:
            return
        s = SLOTS[att.slot]
        step = 2 * FEATURES[s["feature"]][2]
        base = self.inc_s.per_day
        up = _with_rule(self.inc, i, _move_thr(self.inc.rules[i], s["feature"], s["op"], +step))
        dn = _with_rule(self.inc, i, _move_thr(self.inc.rules[i], s["feature"], s["op"], -step))
        d_up = self.ev.score(up, purpose="probe:split").per_day - base
        d_dn = self.ev.score(dn, purpose="probe:split").per_day - base
        single = max(d_up.sum(), d_dn.sum(), 0.0)
        best = None
        for f, x in day_context(self.b).items():
            for q in (0.33, 0.5, 0.67):
                m = float(np.quantile(x, q))
                lo = x <= m
                sums = {+1: (d_up[lo].sum(), d_up[~lo].sum()), -1: (d_dn[lo].sum(), d_dn[~lo].sum())}
                dl = max((+1, -1), key=lambda d: sums[d][0]); dh = max((+1, -1), key=lambda d: sums[d][1])
                gain = max(sums[dl][0], 0) + max(sums[dh][1], 0)
                if best is None or gain > best[0]:
                    best = (gain, f, m, dl, dh)
        gain, f, m, dl, dh = best
        conflict = dl != dh and gain > single + 0.02
        finding = (f"{att.slot}: +step helps {int((d_up > 1e-3).sum())} days, -step helps {int((d_dn > 1e-3).sum())} days;"
                   f" best context split {f}<={snap(f, m)} wants {dl:+d} below / {dh:+d} above; "
                   f"split gain={gain:.3f} vs single={single:.3f}" + (" -> CONFLICTING CONSTRAINTS" if conflict else " -> no conflict"))
        self.memory.record(anomaly="opposing", slot=att.slot, probe="split", finding=finding, at=att.i)
        self.memory.fine_steps[att.slot] = 1          # oscillation => smaller steps from now on
        self._say(f"      ?? split probe: {finding}")
        if conflict and len(self.inc.rules) < 8:
            r = self.inc.rules[i]
            lo_rule = _add_guard(_move_thr(r, s["feature"], s["op"], dl * step), f, "<=", m)
            hi_rule = _move_thr(r, s["feature"], s["op"], dh * step)
            rules = list(self.inc.rules); rules[i:i + 1] = [lo_rule, hi_rule]
            self._try(self.inc.with_rules(rules), "split", att, f"split {att.slot} rule on {f}<={snap(f, m)}")

    # mismatch ----------------------------------------------------------------
    def attribution_probe(self, att, child, child_sim):
        """Expectation mismatch: where did the rejected patch actually lose money?"""
        self.probes_run["attribution"] += 1
        inc_sim, b = self.inc_sim, self.b
        dh = (child_sim.hourly_cost - inc_sim.hourly_cost).sum(axis=0)
        h = int(np.argmax(dh))
        surplus = np.maximum(b.solar - b.load, 0)
        exported_surplus = lambda sim: (np.maximum(-(b.load - b.solar + sim.power), 0) * (surplus > 0)).sum()
        blocked = exported_surplus(child_sim) - exported_surplus(inc_sim)
        full_waste = (((child_sim.soc[:, 1:] > 0.95 * CAP) & (surplus > 0)).sum()
                      - ((inc_sim.soc[:, 1:] > 0.95 * CAP) & (surplus > 0)).sum())
        soc_drop = (inc_sim.soc[:, h] - child_sim.soc[:, h]).mean() / CAP
        fix = None
        if blocked > 1.0 and att.slot == "hold":
            cause = f"the patch made the battery idle while solar was exported ({blocked:.0f} kWh more export): it blocks absorbing"
            fix = "absorb_above"
        elif full_waste > 0 and att.slot == "charge":
            cause = "grid charging filled the battery before free solar arrived"
            fix = ("surplus_left_fc", "<=")
        elif soc_drop > 0.05:
            cause = f"battery {soc_drop:.0%} emptier at hour {h}: energy spent before a higher-value hour"
            fix = ("peak_ahead", "<=") if att.slot in ("peak", "spike") else ("soc", ">=")
        else:
            cause = "inconclusive"
        finding = f"cost rose most at hour {h} (+${dh[h]:.2f} over {b.n} days): {cause}"
        self.memory.record(anomaly="mismatch", slot=att.slot, probe="attribution", finding=finding, at=att.i)
        self._say(f"      ?? attribution probe: {finding}")
        if fix is None or len(child.rules) >= 8:
            return
        i = find_slot(child, att.slot)
        if i is None:
            return
        if fix == "absorb_above":
            rules = list(child.rules); rules.insert(i, SLOTS["absorb"]["template"]())
            if self._try(child.with_rules(rules), "guard", att, "keep the patch, but absorb surplus above it"):
                self.memory.guards.setdefault("hold", ["surplus_fc", "<=", 0.0])
            return
        best = None
        for v in GUARD_CANDIDATES[fix]:
            cand = _with_rule(child, i, _add_guard(child.rules[i], fix[0], fix[1], v))
            sc = self.ev.score(cand, purpose="probe:attribution").mean
            if best is None or sc > best[0]:
                best = (sc, v, cand)
        if self._try(best[2], "guard", att, f"keep the patch, add guard {fix[0]} {fix[1]} {best[1]}"):
            self.memory.guards[att.slot] = [fix[0], fix[1], best[1]]


def run_improver(cls, ev: Evaluator, root: Policy, seed: int, budget: int,
                 memory: Optional[InquiryMemory] = None, verbose: bool = False):
    imp = cls(ev, root, np.random.default_rng(seed), budget, memory=memory, verbose=verbose)
    final = imp.run()
    return imp, final
