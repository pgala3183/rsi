"""The proposer: programmatic edits to a policy program, plus diff features.

In the papers an LLM proposes scaffold/workflow edits. Here a seeded edit
generator plays that role so that every run is reproducible on a laptop and
costs nothing. Each edit returns an `EditInfo` describing *what* was changed,
which the second-order (SO-RSI) improver reads from its trace.

`diff_features()` computes eight coordinates from the textual unified diff of
the rendered source -- without running anything -- mirroring Budget RSI's
"eight coordinates of the proposed diff".
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .env import ACTIONS
from .policy import FEATURES, FEATURE_NAMES, OPS, Cond, Policy, Rule, snap

MAX_RULES = 8
MAX_CONDS = 3
OPERATORS = ("tweak_threshold", "flip_op", "add_condition", "drop_condition", "add_rule",
             "delete_rule", "swap_rules", "change_action", "change_default")


@dataclass
class EditInfo:
    operator: str
    rule: Optional[int] = None          # index of the rule touched (None = default / whole program)
    feature: Optional[str] = None       # feature whose threshold/condition was touched
    direction: int = 0                  # +1 / -1 for threshold moves
    action: Optional[str] = None        # action of the touched rule (after the edit)
    detail: str = ""

    @property
    def target(self) -> Tuple:
        return (self.operator, self.rule, self.feature)


def random_cond(rng: np.random.Generator, features: Sequence[str] = FEATURE_NAMES) -> Cond:
    f = str(rng.choice(list(features)))
    lo, hi, step = FEATURES[f]
    thr = snap(f, rng.uniform(lo, hi))
    return Cond(f, str(rng.choice(OPS)), thr)


# Domain-informed rule templates: what a coding agent that has read the task
# brief would plausibly write. Thresholds are still random, so many proposals
# are harmful -- selection still matters.
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


TEMPLATES = {
    "spike_dump": lambda g: Rule((Cond("export_price", ">=", snap("export_price", _u(g, 0.3, 1.0))),), "discharge_max"),
    "peak_cover": lambda g: Rule((Cond("price_rank", ">=", snap("price_rank", _u(g, 0.5, 0.95))),), "cover_load"),
    "peak_dump": lambda g: Rule((Cond("price_rank", ">=", snap("price_rank", _u(g, 0.75, 1.0))),
                                 Cond("soc", ">=", snap("soc", _u(g, 0.0, 0.4)))), "discharge_max"),
    "cheap_charge": lambda g: Rule((Cond("price_rank", "<=", snap("price_rank", _u(g, 0.05, 0.4))),
                                    Cond("soc", "<=", snap("soc", _u(g, 0.5, 0.95)))),
                                   str(g.choice(["charge_max", "charge_half"]))),
    "cloudy_charge": lambda g: Rule((Cond("price_rank", "<=", snap("price_rank", _u(g, 0.05, 0.4))),
                                     Cond("surplus_left_fc", "<=", snap("surplus_left_fc", _u(g, 2, 20)))), "charge_max"),
    "store_surplus": lambda g: Rule((Cond("surplus_fc", ">=", snap("surplus_fc", _u(g, -0.5, 2.0))),), "absorb_surplus"),
    "hold_for_peak": lambda g: Rule((Cond("peak_ahead", ">=", snap("peak_ahead", _u(g, 0.3, 1.0))),
                                     Cond("price_rank", "<=", snap("price_rank", _u(g, 0.4, 0.8)))), "idle"),
    "need_charge": lambda g: Rule((Cond("peak_need_fc", ">=", snap("peak_need_fc", _u(g, 2, 10))),
                                   Cond("price_rank", "<=", snap("price_rank", _u(g, 0.1, 0.4)))), "charge_half"),
    "cheap_now": lambda g: Rule((Cond("price", "<=", snap("price", _u(g, 0.0, 0.15))),), "charge_max"),
}
GUARDS = {  # plausible extra conditions for add_condition
    "soc": lambda g, a: Cond("soc", "<=" if "charge" in a or a == "absorb_surplus" else ">=",
                             snap("soc", _u(g, 0.5, 0.95) if "charge" in a or a == "absorb_surplus" else _u(g, 0.05, 0.4))),
    "peak_ahead": lambda g, a: Cond("peak_ahead", "<=" if a in ("discharge_max", "cover_load") else ">=",
                                    snap("peak_ahead", _u(g, 0.2, 1.0))),
    "surplus_left_fc": lambda g, a: Cond("surplus_left_fc", "<=", snap("surplus_left_fc", _u(g, 2, 20))),
    "price_rank": lambda g, a: Cond("price_rank", ">=" if a in ("discharge_max", "cover_load") else "<=",
                                    snap("price_rank", _u(g, 0.5, 0.95) if a in ("discharge_max", "cover_load") else _u(g, 0.05, 0.4))),
    "hour": lambda g, a: Cond("hour", str(g.choice(OPS)), float(g.integers(0, 24))),
}
SENSIBLE_DEFAULTS = ("self_consume", "absorb_surplus", "idle", "cover_load")


def soft_templates(preferred: Sequence[str], weight: int = 3) -> list:
    """A research orientation as a soft preference: preferred templates are
    `weight` times as likely, but every template stays reachable."""
    return [t for t in preferred for _ in range(weight)] + list(TEMPLATES)


def _replace_rule(p: Policy, i: int, rule: Rule) -> Policy:
    rules = list(p.rules); rules[i] = rule
    return p.with_rules(rules)


def propose(parent: Policy, rng: np.random.Generator,
            op_weights: Optional[Dict[str, float]] = None,
            features: Sequence[str] = FEATURE_NAMES,
            actions: Sequence[str] = ACTIONS,
            max_tries: int = 50, mode: str = "informed",
            templates: Optional[Sequence[str]] = None) -> Tuple[Policy, EditInfo]:
    """Return a child program that differs textually from `parent`.

    mode="informed": new rules come from TEMPLATES and new conditions from GUARDS
    (an LLM-like prior). mode="uniform": everything is uniformly random (ablation).
    `templates` restricts which rule templates may be used (research orientations).
    """
    ops = list(OPERATORS)
    w = np.array([(op_weights or {}).get(o, 1.0) for o in ops], dtype=float)
    for _ in range(max_tries):
        op = ops[int(rng.choice(len(ops), p=w / w.sum()))]
        out = _apply(parent, op, rng, features, actions, mode, templates)
        if out is not None and out[0].key != parent.key:
            return out
    # fall back: always-possible edit
    return _apply(parent, "add_rule", rng, features, actions, mode, templates) or (parent, EditInfo("noop"))


def _apply(p: Policy, op: str, rng, features, actions, mode="informed", templates=None):
    informed = mode == "informed"
    nr = len(p.rules)
    if op == "tweak_threshold" and nr:
        i = int(rng.integers(nr)); r = p.rules[i]
        j = int(rng.integers(len(r.conds))); c = r.conds[j]
        step = FEATURES[c.feature][2]
        d = int(rng.choice([-1, 1])); k = int(rng.integers(1, 4))
        new = Cond(c.feature, c.op, snap(c.feature, c.thr + d * k * step))
        conds = list(r.conds); conds[j] = new
        return _replace_rule(p, i, Rule(tuple(conds), r.action)), EditInfo(op, i, c.feature, d, r.action, f"{c.thr}->{new.thr}")
    if op == "flip_op" and nr:
        i = int(rng.integers(nr)); r = p.rules[i]
        j = int(rng.integers(len(r.conds))); c = r.conds[j]
        conds = list(r.conds); conds[j] = Cond(c.feature, "<=" if c.op == ">=" else ">=", c.thr)
        return _replace_rule(p, i, Rule(tuple(conds), r.action)), EditInfo(op, i, c.feature, 0, r.action)
    if op == "add_condition" and nr:
        i = int(rng.integers(nr)); r = p.rules[i]
        if len(r.conds) >= MAX_CONDS:
            return None
        if informed:
            g = [f for f in GUARDS if f in features] or list(GUARDS)
            c = GUARDS[str(rng.choice(g))](rng, r.action)
        else:
            c = random_cond(rng, features)
        if any(x.feature == c.feature and x.op == c.op for x in r.conds):
            return None
        return _replace_rule(p, i, Rule(r.conds + (c,), r.action)), EditInfo(op, i, c.feature, 0, r.action)
    if op == "drop_condition" and nr:
        i = int(rng.integers(nr)); r = p.rules[i]
        if len(r.conds) <= 1:
            return None
        j = int(rng.integers(len(r.conds)))
        conds = r.conds[:j] + r.conds[j + 1:]
        return _replace_rule(p, i, Rule(conds, r.action)), EditInfo(op, i, r.conds[j].feature, 0, r.action)
    if op == "add_rule":
        if nr >= MAX_RULES:
            return None
        if informed:
            names = list(templates or TEMPLATES)
            tname = str(rng.choice(names))
            rule = TEMPLATES[tname](rng)
            conds, a = rule.conds, rule.action
        else:
            tname = "random"
            k = int(rng.integers(1, 3))
            conds = tuple(random_cond(rng, features) for _ in range(k))
            a = str(rng.choice(list(actions)))
        pos = int(rng.integers(nr + 1))
        rules = list(p.rules); rules.insert(pos, Rule(conds, a))
        return p.with_rules(rules), EditInfo(op, pos, conds[0].feature, 0, a, tname)
    if op == "delete_rule" and nr:
        i = int(rng.integers(nr))
        rules = list(p.rules); gone = rules.pop(i)
        return p.with_rules(rules), EditInfo(op, i, gone.conds[0].feature, 0, gone.action)
    if op == "swap_rules" and nr >= 2:
        i = int(rng.integers(nr - 1))
        rules = list(p.rules); rules[i], rules[i + 1] = rules[i + 1], rules[i]
        return p.with_rules(rules), EditInfo(op, i, None, 0, rules[i].action)
    if op == "change_action" and nr:
        i = int(rng.integers(nr)); r = p.rules[i]
        a = str(rng.choice([x for x in actions if x != r.action] or list(ACTIONS)))
        return _replace_rule(p, i, Rule(r.conds, a)), EditInfo(op, i, r.conds[0].feature, 0, a)
    if op == "change_default":
        pool = SENSIBLE_DEFAULTS if informed else actions
        a = str(rng.choice([x for x in pool if x != p.default] or list(ACTIONS)))
        return Policy(p.rules, a), EditInfo(op, None, None, 0, a)
    return None


# ----------------------------------------------------------------- diffing
_NUM = re.compile(r"-?\d+\.\d+|-?\d+")
DIFF_FEATURE_NAMES = ("lines_added", "lines_deleted", "rules_added", "rules_deleted",
                      "conds_delta", "constants_changed", "returns_changed", "reordered")


def unified_diff(parent: Policy, child: Policy) -> str:
    return "".join(difflib.unified_diff(parent.source().splitlines(True), child.source().splitlines(True),
                                        "parent.py", "child.py", n=1))


def _diff_lines(parent: Policy, child: Policy):
    plus, minus = [], []
    for line in difflib.unified_diff(parent.source().splitlines(), child.source().splitlines(), n=0, lineterm=""):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            plus.append(line[1:].strip())
        elif line.startswith("-"):
            minus.append(line[1:].strip())
    return plus, minus


def diff_features(parent: Policy, child: Policy) -> np.ndarray:
    """Eight coordinates computed from the textual diff alone (no execution)."""
    plus, minus = _diff_lines(parent, child)
    n_if = lambda ls: sum(l.startswith("if ") for l in ls)
    n_cmp = lambda ls: sum(l.count(">=") + l.count("<=") for l in ls)
    nums_p = Counter(n for l in plus for n in _NUM.findall(l))
    nums_m = Counter(n for l in minus for n in _NUM.findall(l))
    constants_changed = sum((nums_p - nums_m).values())
    returns = sum(l.startswith("return") for l in plus)
    reordered = float(bool(plus) and Counter(plus) == Counter(minus))
    return np.array([len(plus), len(minus), n_if(plus), n_if(minus), n_cmp(plus) - n_cmp(minus),
                     constants_changed, returns, reordered], dtype=float)


# -------------------------------------------------------- domain features
DOMAIN_FEATURE_NAMES = ("adds_discharge_logic", "adds_charge_logic", "touches_price_logic",
                        "touches_soc_guard", "touches_solar_logic", "edit_priority")


def domain_features(parent: Policy, child: Policy) -> np.ndarray:
    """Extra, task-aware coordinates for lab 10.05 (still computed from the diff text only)."""
    plus, minus = _diff_lines(parent, child)
    text_p = " ".join(plus)
    text = " ".join(plus + minus)
    first = parent.source().splitlines()
    try:
        pos = next(i for i, (a, b) in enumerate(zip(first, child.source().splitlines())) if a != b)
    except StopIteration:
        pos = len(first)
    return np.array([
        float("DISCHARGE" in text_p or "COVER_LOAD" in text_p),
        float("CHARGE_MAX" in text_p or "CHARGE_HALF" in text_p or "ABSORB" in text_p),
        float(any(k in text for k in ("o.price", "o.peak_ahead", "o.export_price"))),
        float("o.soc" in text),
        float("surplus" in text),
        1.0 - pos / max(len(first), 1),      # 1 = edit near the top (highest priority rule)
    ])
