"""Evaluation with an honest cost ledger and a frozen held-out boundary.

* Every simulated day an improver pays for is counted in `Evaluator.ledger`
  ("day-sims", the analogue of benchmark-instance evaluations in the papers).
  Re-scoring a program already scored on the same days is free (cached), as it
  would be for a real system that stores results.
* The held-out split is *protected*: an improver that asks for it gets a
  `BoundaryViolation`. Only `final_report()` -- called by the lab, after the
  search has returned -- may read it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .env import SOLVED_AT, DayBatch, Task, capture, simulate
from .policy import Policy


class BoundaryViolation(RuntimeError):
    pass


@dataclass
class Score:
    mean: float            # mean fraction of oracle savings captured
    solved: int            # days with capture >= SOLVED_AT
    n: int
    per_day: np.ndarray

    @property
    def rate(self) -> float:
        return self.solved / self.n

    def __repr__(self):
        return f"Score(capture={self.mean:.3f}, solved={self.solved}/{self.n})"


def score_batch(policy: Policy, b: DayBatch, wrapper=None) -> Score:
    res = simulate(b, policy.act, wrapper=wrapper)
    cap = capture(b, res.cost)
    return Score(float(cap.mean()), int((cap >= SOLVED_AT).sum()), b.n, cap)


class Evaluator:
    def __init__(self, task: Task, protected: Sequence[str] = ("heldout",), charge_repeats: bool = False):
        """`charge_repeats=True` bills re-evaluations of an already-seen program
        (an improver with no memory of its past attempts would re-run them)."""
        self.task = task
        self.charge_repeats = charge_repeats
        self.protected = set(protected)
        self.ledger: Counter = Counter()
        self.by_purpose: Counter = Counter()     # day-sims billed per purpose ("search", "probe:sweep", ...)
        self.access_log = []
        self._cache = {}
        self._traces = {}

    def score(self, policy: Policy, split: str = "steering", idx: Optional[Sequence[int]] = None,
              wrapper=None, purpose: str = "search") -> Score:
        if split in self.protected:
            self.access_log.append(("DENIED", split, purpose))
            raise BoundaryViolation(f"{purpose!r} tried to read the protected {split!r} split")
        b = self.task.split(split)
        key_idx = None if idx is None else tuple(int(i) for i in idx)
        key = (policy.key, split, key_idx, getattr(wrapper, "key", None))
        hit = self._cache.get(key)
        if hit is not None:
            if self.charge_repeats:
                self.ledger[split] += hit.n
                self.by_purpose[purpose] += hit.n
            return hit
        sub = b if idx is None else b.subset(np.asarray(idx))
        s = score_batch(policy, sub, wrapper)
        self._cache[key] = s
        self.ledger[split] += sub.n
        self.by_purpose[purpose] += sub.n
        self.access_log.append(("ok", split, purpose))
        return s

    def trace(self, policy: Policy, split: str = "steering", purpose: str = "diagnosis"):
        """Score + full hourly record (for diagnosis). Same price as `score`."""
        if split in self.protected:
            self.access_log.append(("DENIED", split, purpose))
            raise BoundaryViolation(f"{purpose!r} tried to read the protected {split!r} split")
        key = (policy.key, split, None, None)
        hit = self._traces.get(key)
        if hit is not None:
            if self.charge_repeats:
                self.ledger[split] += hit[0].n
                self.by_purpose[purpose] += hit[0].n
            return hit
        b = self.task.split(split)
        sim = simulate(b, policy.act, record=True)
        cap = capture(b, sim.cost)
        s = Score(float(cap.mean()), int((cap >= SOLVED_AT).sum()), b.n, cap)
        if key not in self._cache:
            self.ledger[split] += b.n
            self.by_purpose[purpose] += b.n
            self._cache[key] = s
            self.access_log.append(("ok", split, purpose))
        self._traces[key] = (self._cache[key], sim)
        return self._traces[key]

    @property
    def day_sims(self) -> int:
        return sum(v for k, v in self.ledger.items() if k != "heldout_report")

    def final_report(self, policy: Policy, wrapper=None) -> Score:
        """Called by the experimenter after search; never by an improver."""
        self.ledger["heldout_report"] += self.task.heldout.n
        self.access_log.append(("report", "heldout", "final_report"))
        return score_batch(policy, self.task.heldout, wrapper)


def mean_sd(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0
