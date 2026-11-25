"""An RSIGym-style "Everything as a Service" layer for GridWise (arXiv:2610.10310).

An improving agent never touches the simulator directly. It calls services:

  sandbox(source)              parse policy SOURCE TEXT against a whitelist grammar
  evaluate(source, split, ...) score a program (metered, permission-checked)
  rollout(source, split, day)  hourly trace of one day (for diagnosis)
  oracle_gap(split)            how much room there is (aggregate only)

with shared budget and permission controls:
  * role "agent"   : fit + steering only, hard day-sim budget
  * role "auditor" : may read held-out, unmetered

The sandbox is a parser, not `exec`: anything outside the rule grammar
(imports, attribute access other than o.<feature>, loops, calls) is rejected.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import numpy as np

from .env import ACTIONS, Task, simulate
from .evaluate import BoundaryViolation, Evaluator
from .policy import FEATURES, Cond, Policy, Rule


class SandboxError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


_COND = re.compile(r"^o\.([a-z_]+)\s*(>=|<=)\s*(-?\d+(?:\.\d+)?)$")
_ACT = {a.upper(): a for a in ACTIONS}


def parse_policy(source: str) -> Policy:
    lines = [l.rstrip() for l in source.strip("\n").splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines or lines[0].strip() != "def policy(o):":
        raise SandboxError("program must start with 'def policy(o):'")
    body, rules, i = lines[1:], [], 0
    while i < len(body):
        line = body[i].strip()
        if line.startswith("if ") and line.endswith(":"):
            conds = []
            for part in line[3:-1].split(" and "):
                m = _COND.match(part.strip())
                if not m or m.group(1) not in FEATURES:
                    raise SandboxError(f"condition not allowed: {part.strip()!r}")
                conds.append(Cond(m.group(1), m.group(2), float(m.group(3))))
            if i + 1 >= len(body) or not body[i + 1].strip().startswith("return "):
                raise SandboxError("each 'if' must be followed by 'return ACTION'")
            act = body[i + 1].strip()[7:]
            if act not in _ACT:
                raise SandboxError(f"unknown action {act!r}")
            if len(conds) > 3 or len(rules) >= 8:
                raise SandboxError("too many conditions or rules")
            rules.append(Rule(tuple(conds), _ACT[act])); i += 2
        elif line.startswith("return ") and i == len(body) - 1:
            act = line[7:]
            if act not in _ACT:
                raise SandboxError(f"unknown action {act!r}")
            return Policy(tuple(rules), _ACT[act])
        else:
            raise SandboxError(f"statement not allowed: {line!r}")
    raise SandboxError("program must end with a default 'return ACTION'")


class MeteredEvaluator(Evaluator):
    """An Evaluator that refuses to run past its budget."""

    def __init__(self, task: Task, budget: int, protected=("heldout",), **kw):
        super().__init__(task, protected=protected, **kw)
        self.budget = budget

    def _check(self, n):
        if self.day_sims + n > self.budget:
            raise BudgetExceeded(f"budget of {self.budget} day-sims exhausted ({self.day_sims} used)")

    def score(self, policy, split="steering", idx=None, wrapper=None, purpose="search"):
        n = self.task.split(split).n if idx is None else len(idx)
        if split not in self.protected:
            key = (policy.key, split, None if idx is None else tuple(int(i) for i in idx), getattr(wrapper, "key", None))
            if key not in self._cache or self.charge_repeats:
                self._check(n)
        return super().score(policy, split, idx, wrapper, purpose)

    def trace(self, policy, split="steering", purpose="diagnosis"):
        if split not in self.protected and (policy.key, split, None, None) not in self._cache:
            self._check(self.task.split(split).n)
        return super().trace(policy, split, purpose)


class EaaS:
    ROLES = {"agent": {"splits": {"fit", "steering"}, "metered": True},
             "auditor": {"splits": {"fit", "steering", "heldout"}, "metered": False}}

    def __init__(self, task: Task, role: str = "agent", budget: int = 4800):
        if role not in self.ROLES:
            raise ValueError(role)
        self.role, self.task = role, task
        prot = tuple(s for s in ("fit", "steering", "heldout") if s not in self.ROLES[role]["splits"])
        self.ev = (MeteredEvaluator(task, budget, protected=prot) if self.ROLES[role]["metered"]
                   else Evaluator(task, protected=prot))
        self.calls: Dict[str, int] = {}

    def _call(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def sandbox(self, source: str) -> Policy:
        self._call("sandbox")
        return parse_policy(source)

    def evaluate(self, source: str, split: str = "steering", idx: Optional[Sequence[int]] = None, wrapper=None) -> dict:
        self._call("evaluate")
        s = self.ev.score(self.sandbox(source), split, idx, wrapper, purpose=f"eaas:{self.role}")
        return {"capture": s.mean, "solved": s.solved, "n": s.n}

    def rollout(self, source: str, split: str, day: int) -> dict:
        self._call("rollout")
        if split not in self.ROLES[self.role]["splits"]:
            raise BoundaryViolation(f"role {self.role!r} may not roll out on {split!r}")
        if self.ROLES[self.role]["metered"]:
            self.ev._check(1)
        self.ev.ledger[split] += 1
        b = self.task.split(split).subset([day])
        sim = simulate(b, self.sandbox(source).act, record=True)
        return {"actions": [ACTIONS[a] for a in sim.actions[0]], "soc": sim.soc[0].round(2).tolist(),
                "cost": float(sim.cost[0])}

    def oracle_gap(self, split: str = "steering") -> float:
        self._call("oracle_gap")
        if split not in self.ROLES[self.role]["splits"]:
            raise BoundaryViolation(f"role {self.role!r} may not query {split!r}")
        b = self.task.split(split)
        return float(np.mean(b.cost_none - b.cost_oracle))

    @property
    def used(self) -> int:
        return self.ev.day_sims
