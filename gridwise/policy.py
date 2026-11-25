"""Policies are small *programs*: an ordered list of if-rules plus a default.

The program has two faithful views:

* `Policy.source()`  -> readable Python (what an agent edits and what diffs are taken of)
* `Policy.act(obs)`  -> a vectorised interpreter used by the fast simulator

`check_source_matches()` executes the rendered Python on scalar observations
and confirms it agrees with the vectorised interpreter, so the readable code
really is the code being evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

from .env import ACTIONS, ACTION_ID

# feature -> (low, high, mutation step). Thresholds are rounded to the step.
FEATURES: Dict[str, Tuple[float, float, float]] = {
    "hour": (0.0, 23.0, 1.0),
    "price": (0.0, 1.6, 0.02),
    "export_price": (0.0, 1.5, 0.05),
    "price_rank": (0.0, 1.0, 0.05),
    "peak_ahead": (0.0, 2.0, 0.05),
    "soc": (0.0, 1.0, 0.05),
    "surplus_fc": (-4.0, 6.0, 0.25),
    "surplus_left_fc": (0.0, 40.0, 1.0),
    "peak_need_fc": (0.0, 16.0, 0.5),
    "day_surplus_fc": (0.0, 60.0, 1.0),
    "day_peak": (0.0, 2.5, 0.05),
}
FEATURE_NAMES = tuple(FEATURES)
OPS = (">=", "<=")


def snap(feature: str, value: float) -> float:
    lo, hi, step = FEATURES[feature]
    v = min(max(value, lo), hi)
    return round(round(v / step) * step, 4)


def fmt(feature: str, value: float) -> str:
    return f"{value:.0f}" if feature == "hour" else f"{value:.2f}"


@dataclass(frozen=True)
class Cond:
    feature: str
    op: str
    thr: float

    def code(self) -> str:
        return f"o.{self.feature} {self.op} {fmt(self.feature, self.thr)}"


@dataclass(frozen=True)
class Rule:
    conds: Tuple[Cond, ...]
    action: str


@dataclass(frozen=True)
class Policy:
    rules: Tuple[Rule, ...]
    default: str = "idle"

    # ------------------------------------------------------------------ views
    def source(self, name: str = "policy") -> str:
        lines = [f"def {name}(o):"]
        for r in self.rules:
            cond = " and ".join(c.code() for c in r.conds)
            lines.append(f"    if {cond}:")
            lines.append(f"        return {r.action.upper()}")
        lines.append(f"    return {self.default.upper()}")
        return "\n".join(lines) + "\n"

    @property
    def key(self) -> str:
        return self.source()

    def act(self, obs) -> np.ndarray:
        out = np.full(obs.n, ACTION_ID[self.default])
        undecided = np.ones(obs.n, dtype=bool)
        cache = {}
        for r in self.rules:
            m = undecided.copy()
            for c in r.conds:
                v = cache.get(c.feature)
                if v is None:
                    v = cache[c.feature] = obs[c.feature]
                m &= (v >= c.thr) if c.op == ">=" else (v <= c.thr)
            out[m] = ACTION_ID[r.action]
            undecided &= ~m
        return out

    def which(self, obs) -> np.ndarray:
        """Index of the rule that fires for each day (len(rules) = the default)."""
        out = np.full(obs.n, len(self.rules))
        undecided = np.ones(obs.n, dtype=bool)
        for k, r in enumerate(self.rules):
            m = undecided.copy()
            for c in r.conds:
                v = obs[c.feature]
                m &= (v >= c.thr) if c.op == ">=" else (v <= c.thr)
            out[m] = k
            undecided &= ~m
        return out

    # -------------------------------------------------------------- utilities
    def n_conds(self) -> int:
        return sum(len(r.conds) for r in self.rules)

    def with_rules(self, rules) -> "Policy":
        return replace(self, rules=tuple(rules))

    def compile(self):
        """Exec the rendered source in a namespace with only action constants."""
        ns = {a.upper(): ACTION_ID[a] for a in ACTIONS}
        ns["__builtins__"] = {}
        exec(compile(self.source(), "<policy>", "exec"), ns)
        return ns["policy"]


def C(feature: str, op: str, thr: float) -> Cond:
    return Cond(feature, op, snap(feature, thr))


def R(action: str, *conds: Cond) -> Rule:
    return Rule(tuple(conds), action)


# ------------------------------------------------------------- named policies
SEED_POLICY = Policy(rules=(), default="self_consume")
"""What most home batteries ship with: store surplus solar, cover deficits."""

IDLE_POLICY = Policy(rules=(), default="idle")

EXPERT_POLICY = Policy(rules=(
    R("discharge_max", C("export_price", ">=", 0.50)),                            # dump into spikes / events
    R("cover_load", C("price_rank", ">=", 0.75)),                                  # cover the peak
    R("charge_max", C("price_rank", "<=", 0.25), C("surplus_left_fc", "<=", 8), C("soc", "<=", 0.85)),
    R("absorb_surplus", C("surplus_fc", ">=", 0.0)),
    R("idle", C("peak_ahead", ">=", 0.35)),                                        # hold for the peak
), default="self_consume")
"""A hand-written reference: a person's best guess, not an optimum."""


class _ScalarObs:
    def __init__(self, d):
        self.__dict__.update(d)


def check_source_matches(policy: Policy, b, n_checks: int = 200, seed: int = 0) -> int:
    """Execute the rendered source on scalar observations; count disagreements."""
    from .env import Obs, CAP
    fn = policy.compile()
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_checks):
        d = int(rng.integers(b.n)); t = int(rng.integers(24))
        soc = np.full(b.n, rng.uniform(0, CAP))
        obs = Obs(b, t, soc)
        vec = int(policy.act(obs)[d])
        scalar = fn(_ScalarObs({f: (float(obs[f]) if f == "hour" else float(obs[f][d])) for f in FEATURE_NAMES}))
        bad += int(vec != scalar)
    return bad
