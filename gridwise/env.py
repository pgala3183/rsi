"""GridWise task: hourly dispatch of a home solar + battery system.

Every "instance" is one day (24 hourly decisions). A controller (the policy)
sees day-ahead prices, noisy solar/load forecasts and the battery state of
charge, and chooses an action each hour. We simulate the day with the *actual*
solar and load, and charge the household for imports / credit it for exports.

Why this is a good RSI testbed
------------------------------
* There is an exact upper bound: a dynamic-programming oracle with perfect
  foresight gives the cheapest possible day. So every day has a normalised
  score in (-inf, 1]: the fraction of the oracle's savings the policy captured.
* The best behaviour is *conditional* (grid-charge at night only when solar
  will not fill the battery; hold charge for the evening peak; dump energy
  into price spikes), so simple threshold patches thrash and structural edits
  are needed -- exactly the situation the October 2026 RSI papers study.
* Five tariffs act as five benchmarks (for RSIGym's RSI-Index).

Everything is NumPy, vectorised across days, and seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

HOURS = 24
CAP = 13.5            # usable battery capacity, kWh
PMAX = 5.0            # max charge / discharge power at the inverter, kW
ETA_C = 0.95          # charge efficiency
ETA_D = 0.95          # discharge efficiency
SOC0 = 4.0            # kWh in the battery at midnight
TERMINAL_VALUE = 0.12  # $/kWh credited for energy still stored at midnight
SOC_STEP = 0.25       # oracle DP resolution, kWh
SOLVED_AT = 0.80      # a day is "solved" if the policy captures >= 80% of oracle savings

TARIFFS = ("tou", "vpp_events", "critical_peak", "wholesale", "solar_glut")

ACTIONS = (
    "idle",            # battery does nothing
    "charge_max",      # charge at full power (from solar and/or grid)
    "discharge_max",   # discharge at full power (excess is exported)
    "absorb_surplus",  # charge only with measured solar surplus
    "cover_load",      # discharge only to cover the measured deficit
    "self_consume",    # absorb surplus if any, else cover deficit
    "charge_half",     # charge at half power
)
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


def _gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


# --------------------------------------------------------------------------
# Day generation
# --------------------------------------------------------------------------
def _prices(tariff: str, rng: np.random.Generator, n: int, summer: np.ndarray):
    """Return (import_price, export_price, spike_flag) arrays of shape (n, 24)."""
    t = np.arange(HOURS)[None, :]
    spike = np.zeros(n, dtype=bool)
    if tariff == "tou":
        base = np.select([t < 6, t < 16, t < 21], [0.13, 0.22, 0.48], 0.26).astype(float)
        day_mult = rng.lognormal(0.0, 0.06, size=(n, 1))
        peak_mult = rng.uniform(0.85, 1.25, size=(n, 1))
        price = base * day_mult
        peak = (t >= 16) & (t < 21)
        price = np.where(peak, price * peak_mult, price)
        spike = rng.random(n) < (0.04 + 0.12 * summer)
        spike_mult = rng.uniform(2.5, 4.0, size=(n, 1))
        in_spike = spike[:, None] & (t >= 17) & (t < 21)
        price = np.where(in_spike, price * spike_mult, price)
        export = np.where(in_spike, 0.6 * price, 0.05)
    elif tariff == "vpp_events":
        # flat retail price, but a virtual-power-plant operator pays a large
        # export credit during announced evening events on ~30% of days
        price = np.full((n, HOURS), 0.30) * rng.lognormal(0.0, 0.03, size=(n, 1))
        spike = rng.random(n) < 0.30
        in_event = spike[:, None] & (t >= 17) & (t < 20)
        export = np.where(in_event, rng.uniform(0.6, 1.0, size=(n, 1)), 0.06)
    elif tariff == "critical_peak":
        base = np.select([t < 6, t < 16, t < 21], [0.14, 0.24, 0.42], 0.28).astype(float)
        price = base * rng.lognormal(0.0, 0.05, size=(n, 1))
        spike = rng.random(n) < 0.15
        in_cpp = spike[:, None] & (t >= 16) & (t < 21)
        price = np.where(in_cpp, 1.60, price)
        export = np.full((n, HOURS), 0.05)
    elif tariff == "wholesale":
        duck = (0.10 + 0.08 * _gauss(t, 7.5, 1.5) + 0.32 * _gauss(t, 19.0, 1.8)
                - 0.12 * _gauss(t, 12.5, 2.2))
        level = rng.lognormal(0.0, 0.15, size=(n, 1))
        dip = (0.6 + 0.8 * summer)[:, None]
        price = duck * level - 0.04 * dip * _gauss(t, 12.5, 2.2)
        # AR(1) hourly noise
        eps = np.zeros((n, HOURS))
        z = rng.normal(0, 0.025, size=(n, HOURS))
        for h in range(HOURS):
            eps[:, h] = (0.7 * eps[:, h - 1] if h else 0) + z[:, h]
        price = np.maximum(price + eps, -0.05)
        spike = rng.random(n) < 0.06
        in_spike = spike[:, None] & (t >= 18) & (t < 21)
        price = np.where(in_spike, price + rng.uniform(0.6, 1.5, size=(n, 1)), price)
        export = 0.9 * price
    elif tariff == "solar_glut":
        base = np.select([t < 6, t < 10, t < 16, t < 22], [0.20, 0.30, 0.06, 0.55], 0.22).astype(float)
        price = base * rng.lognormal(0.0, 0.05, size=(n, 1))
        export = np.where((t >= 10) & (t < 16), 0.01, 0.04) + 0 * price
    else:
        raise ValueError(f"unknown tariff {tariff!r}; choose from {TARIFFS}")
    return price, export, spike


def _weather(rng: np.random.Generator, n: int):
    doy = rng.integers(0, 365, size=n)
    summer = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365.0)   # 1 in June, 0 in Dec
    u = rng.random(n)
    cloud = np.where(u < 0.5, rng.uniform(0.9, 1.0, n),
                     np.where(u < 0.8, rng.uniform(0.45, 0.8, n), rng.uniform(0.08, 0.3, n)))
    cloudy = cloud < 0.5
    t = np.arange(HOURS)[None, :] + 0.5
    sunrise = (6.8 - 1.6 * summer)[:, None]
    sunset = (16.8 + 3.2 * summer)[:, None]
    phase = np.clip((t - sunrise) / (sunset - sunrise), 0, 1)
    peak_kw = (7.0 * (0.55 + 0.45 * summer) * cloud)[:, None]
    solar = peak_kw * np.sin(np.pi * phase) ** 1.2
    solar = np.clip(solar * (1 + 0.10 * rng.normal(size=solar.shape)), 0, None)

    tt = np.arange(HOURS)[None, :]
    load = 0.45 + 1.0 * _gauss(tt, 7.5, 1.3) + 2.2 * _gauss(tt, 19.5, 2.0)
    heat = ((summer > 0.6) & ~cloudy).astype(float)[:, None]
    cold = (summer < 0.3).astype(float)[:, None]
    load = load + 1.8 * heat * _gauss(tt, 16, 3.0) + 0.9 * cold * (_gauss(tt, 6, 2) + _gauss(tt, 20, 2))
    load = load * rng.normal(1.0, 0.10, size=(n, 1))
    load = np.clip(load * (1 + 0.08 * rng.normal(size=load.shape)), 0.2, None)

    sbias = np.clip(rng.normal(1.0, 0.15, size=(n, 1)), 0.5, 1.5)
    solar_fc = np.clip(solar * sbias * (1 + 0.10 * rng.normal(size=solar.shape)), 0, None)
    load_fc = np.clip(load * rng.normal(1.0, 0.08, size=(n, 1)) * (1 + 0.05 * rng.normal(size=load.shape)), 0.1, None)
    return doy, summer, cloudy, solar, load, solar_fc, load_fc


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class DayBatch:
    """A set of days. All arrays have the day as their first axis."""
    tariff: str
    doy: np.ndarray
    price: np.ndarray
    export: np.ndarray
    solar: np.ndarray
    load: np.ndarray
    solar_fc: np.ndarray
    load_fc: np.ndarray
    tags: Dict[str, np.ndarray]
    feats: Dict[str, np.ndarray] = field(default_factory=dict)   # static features (n, 24)
    cost_none: Optional[np.ndarray] = None      # cost with the battery idle
    cost_oracle: Optional[np.ndarray] = None    # perfect-foresight optimum
    oracle_p: Optional[np.ndarray] = None       # oracle battery power per hour (n, 24)

    @property
    def n(self) -> int:
        return len(self.doy)

    def subset(self, idx) -> "DayBatch":
        idx = np.asarray(idx)
        pick = lambda a: None if a is None else a[idx]
        return DayBatch(
            tariff=self.tariff, doy=self.doy[idx], price=self.price[idx], export=self.export[idx],
            solar=self.solar[idx], load=self.load[idx], solar_fc=self.solar_fc[idx], load_fc=self.load_fc[idx],
            tags={k: v[idx] for k, v in self.tags.items()},
            feats={k: v[idx] for k, v in self.feats.items()},
            cost_none=pick(self.cost_none), cost_oracle=pick(self.cost_oracle), oracle_p=pick(self.oracle_p),
        )


def _static_features(b: DayBatch) -> Dict[str, np.ndarray]:
    """Features a controller may read, computed only from day-ahead information."""
    n = b.n
    price = b.price
    # the value of a stored kWh at hour h: avoid an import, or earn an export
    value = np.maximum(b.price, b.export)
    rank = (value[:, None, :] < value[:, :, None]).sum(axis=2) / (HOURS - 1.0)
    peak_ahead = np.zeros((n, HOURS))
    for h in range(HOURS - 1):
        peak_ahead[:, h] = value[:, h + 1:].max(axis=1)
    surplus_fc = b.solar_fc - b.load_fc
    surplus_left = np.cumsum(np.maximum(surplus_fc, 0)[:, ::-1], axis=1)[:, ::-1]
    deficit = np.maximum(-surplus_fc, 0)
    q75 = np.quantile(value, 0.75, axis=1, keepdims=True)
    high = value >= q75
    need = np.zeros((n, HOURS))
    for h in range(HOURS):
        need[:, h] = (deficit[:, h + 1:] * high[:, h + 1:]).sum(axis=1)
    return {
        "hour": np.broadcast_to(np.arange(HOURS, dtype=float), (n, HOURS)).copy(),
        "price": price,
        "export_price": b.export.astype(float).copy(),
        "price_rank": rank,
        "peak_ahead": peak_ahead,
        "surplus_fc": surplus_fc,
        "surplus_left_fc": surplus_left,
        "peak_need_fc": need,
        # day-level context (constant through the day; known day-ahead)
        "day_surplus_fc": np.repeat(surplus_left[:, :1], HOURS, axis=1),
        "day_peak": np.repeat(value.max(axis=1, keepdims=True), HOURS, axis=1),
    }


def make_days(n: int, tariff: str = "tou", seed: int = 0) -> DayBatch:
    """Generate `n` days for a tariff, with features, no-battery cost and oracle cost."""
    rng = np.random.default_rng([seed, TARIFFS.index(tariff) if tariff in TARIFFS else 99])
    doy, summer, cloudy, solar, load, solar_fc, load_fc = _weather(rng, n)
    price, export, spike = _prices(tariff, rng, n, summer)
    b = DayBatch(tariff, doy, price, export, solar, load, solar_fc, load_fc,
                 tags={"summer": summer, "cloudy": cloudy, "spike": spike})
    b.feats = _static_features(b)
    b.cost_none = day_cost_idle(b)
    b.cost_oracle, b.oracle_p = oracle(b)
    return b


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def desired_power(action: np.ndarray, solar: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Map action ids to requested battery power (+ charge, - discharge), kW."""
    surplus = np.maximum(solar - load, 0.0)
    deficit = np.maximum(load - solar, 0.0)
    table = np.stack([
        np.zeros_like(solar),               # idle
        np.full_like(solar, PMAX),          # charge_max
        np.full_like(solar, -PMAX),         # discharge_max
        surplus,                            # absorb_surplus
        -deficit,                           # cover_load
        surplus - deficit,                  # self_consume
        np.full_like(solar, PMAX / 2),      # charge_half
    ])
    return table[action, np.arange(len(action))]


@dataclass
class SimResult:
    cost: np.ndarray                         # (n,) $ per day (after terminal credit)
    hourly_cost: Optional[np.ndarray] = None  # (n, 24)
    actions: Optional[np.ndarray] = None      # (n, 24) action ids
    power: Optional[np.ndarray] = None        # (n, 24) battery power at the bus
    soc: Optional[np.ndarray] = None          # (n, 25) kWh before each hour (+ end)


class Obs:
    """What the policy can see at hour t (vectorised over days)."""
    __slots__ = ("n", "t", "_b", "soc_kwh")

    def __init__(self, b: DayBatch, t: int, soc_kwh: np.ndarray):
        self.n, self.t, self._b, self.soc_kwh = b.n, t, b, soc_kwh

    def __getitem__(self, name: str):
        if name == "soc":
            return self.soc_kwh / CAP
        if name == "hour":
            return float(self.t)
        return self._b.feats[name][:, self.t]


def simulate(b: DayBatch, act_fn, record: bool = False, wrapper=None) -> SimResult:
    """Run a policy for all days in `b`. `act_fn(obs) -> int array of action ids`.

    `wrapper(obs, actions) -> actions` is an optional runtime harness component
    (used by the Self-HPO lab) that may override the policy's choice.
    """
    n = b.n
    rows = np.arange(n)
    soc = np.full(n, SOC0)
    cost = np.zeros(n)
    if record:
        hc = np.zeros((n, HOURS)); acts = np.zeros((n, HOURS), dtype=int)
        pw = np.zeros((n, HOURS)); socs = np.zeros((n, HOURS + 1)); socs[:, 0] = soc
    for t in range(HOURS):
        obs = Obs(b, t, soc)
        a = np.asarray(act_fn(obs), dtype=int)
        if a.ndim == 0:
            a = np.full(n, int(a))
        if wrapper is not None:
            a = np.asarray(wrapper(obs, a), dtype=int)
        solar, load = b.solar[:, t], b.load[:, t]
        p = desired_power(a, solar, load)
        p = np.clip(p, -np.minimum(PMAX, soc * ETA_D), np.minimum(PMAX, (CAP - soc) / ETA_C))
        soc = np.clip(soc + np.where(p > 0, p * ETA_C, p / ETA_D), 0.0, CAP)
        grid = load - solar + p
        c = b.price[:, t] * np.maximum(grid, 0) - b.export[:, t] * np.maximum(-grid, 0)
        cost += c
        if record:
            hc[:, t] = c; acts[:, t] = a; pw[:, t] = p; socs[:, t + 1] = soc
    cost = cost - TERMINAL_VALUE * soc
    if record:
        return SimResult(cost, hc, acts, pw, socs)
    return SimResult(cost)


def day_cost_idle(b: DayBatch) -> np.ndarray:
    grid = b.load - b.solar
    c = (b.price * np.maximum(grid, 0) - b.export * np.maximum(-grid, 0)).sum(axis=1)
    return c - TERMINAL_VALUE * SOC0


def oracle(b: DayBatch):
    """Perfect-foresight dynamic program over a discretised state of charge.

    Returns (cost per day, battery power per hour). Vectorised over days.
    """
    L = int(round(CAP / SOC_STEP)) + 1
    lv = np.arange(L) * SOC_STEP
    delta = lv[None, :] - lv[:, None]                     # (from i, to j) energy change in battery
    p_bus = np.where(delta > 0, delta / ETA_C, delta * ETA_D)
    feasible = np.abs(p_bus) <= PMAX + 1e-9
    n = b.n
    V = np.broadcast_to(-TERMINAL_VALUE * lv, (n, L)).copy()
    choice = np.zeros((HOURS, n, L), dtype=np.int16)
    for t in range(HOURS - 1, -1, -1):
        net = (b.load[:, t] - b.solar[:, t])[:, None, None]
        grid = net + p_bus[None]
        c = b.price[:, t, None, None] * np.maximum(grid, 0) - b.export[:, t, None, None] * np.maximum(-grid, 0)
        q = np.where(feasible[None], c + V[:, None, :], np.inf)
        choice[t] = q.argmin(axis=2)
        V = q.min(axis=2)
    i0 = int(round(SOC0 / SOC_STEP))
    cost = V[:, i0]
    # roll forward to recover the plan
    idx = np.full(n, i0)
    power = np.zeros((n, HOURS))
    rows = np.arange(n)
    for t in range(HOURS):
        j = choice[t, rows, idx]
        power[:, t] = p_bus[idx, j]
        idx = j
    return cost, power


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def capture(b: DayBatch, cost: np.ndarray) -> np.ndarray:
    """Fraction of the oracle's savings (vs. no battery) the policy captured, per day."""
    room = np.maximum(b.cost_none - b.cost_oracle, 0.05)
    return np.clip((b.cost_none - cost) / room, -1.0, 1.0)


@dataclass
class Task:
    """A frozen partition of one tariff's days (mirrors the Budget RSI paper's
    16 fit / 48 steering / 100 held-out split of HumanEval+'s 164 problems)."""
    tariff: str
    seed: int
    fit: DayBatch
    steering: DayBatch
    heldout: DayBatch

    def split(self, name: str) -> DayBatch:
        return {"fit": self.fit, "steering": self.steering, "heldout": self.heldout}[name]


def make_task(tariff: str = "tou", seed: int = 0, n_fit: int = 16, n_steer: int = 48, n_held: int = 100) -> Task:
    days = make_days(n_fit + n_steer + n_held, tariff=tariff, seed=seed)
    order = np.random.default_rng(seed + 1000).permutation(days.n)
    return Task(tariff, seed,
                fit=days.subset(order[:n_fit]),
                steering=days.subset(order[n_fit:n_fit + n_steer]),
                heldout=days.subset(order[n_fit + n_steer:]))
