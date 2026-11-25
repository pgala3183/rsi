"""Failure symptoms: *where* does a policy lose money relative to the oracle?

Compares the policy's battery power with the oracle's, hour by hour, and puts
each mismatch into one of five symptom categories, weighted by the price at
that hour (a dollar-scale proxy for regret). First-order improvers patch the
largest symptom; second-order improvers ask why patches keep failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .env import CAP, DayBatch, SimResult

SYMPTOMS = ("uncovered_peak", "early_discharge", "missed_grid_charge", "excess_grid_charge", "wasted_solar")


@dataclass
class Symptoms:
    total: Dict[str, float]          # $-weighted mismatch per category, summed over days
    per_day: Dict[str, np.ndarray]   # (n,) per category

    def top(self) -> str:
        return max(self.total, key=self.total.get)

    def __repr__(self):
        return "Symptoms(" + ", ".join(f"{k}={v:.1f}" for k, v in sorted(self.total.items(), key=lambda kv: -kv[1])) + ")"


def symptoms(b: DayBatch, sim: SimResult) -> Symptoms:
    p_pi, p_or = sim.power, b.oracle_p
    d = p_or - p_pi
    surplus = (b.solar - b.load) > 0
    rank = b.feats["price_rank"]
    w = np.abs(d) * np.maximum(np.abs(b.price), b.export)
    masks = {
        "uncovered_peak": (d < -0.1) & (p_or < 0) & (rank >= 0.7),
        "early_discharge": (d > 0.1) & (p_pi < 0) & (rank < 0.7),
        "missed_grid_charge": (d > 0.1) & ~surplus & (p_or > 0) & (p_pi >= 0),
        "excess_grid_charge": (d < -0.1) & ~surplus & (p_pi > 0),
        "wasted_solar": (d > 0.1) & surplus & (p_or > 0),
    }
    per_day = {k: (w * m).sum(axis=1) for k, m in masks.items()}
    return Symptoms({k: float(v.sum()) for k, v in per_day.items()}, per_day)


def day_context(b: DayBatch) -> Dict[str, np.ndarray]:
    """Day-level context features (constant through a day, readable by rules)."""
    return {
        "day_surplus_fc": b.feats["day_surplus_fc"][:, 0],     # forecast solar surplus today, kWh
        "day_peak": b.feats["day_peak"][:, 0],                 # highest value of a kWh today, $/kWh
    }


def soc_at(sim: SimResult, hour: int) -> np.ndarray:
    return sim.soc[:, hour] / CAP
