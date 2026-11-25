"""Budget RSI (arXiv:2610.29121), adapted to GridWise.

Paper mechanism (from the abstract):
  1. Fit ONE linear readout on [parent's steering-split solve rate, 8 diff
     coordinates] from 96 edits labelled on a 16-instance fit split, then freeze it.
  2. During search, each archive node proposes edits; the readout scores every
     proposal *before it runs*; only the top-scored proposal is evaluated.
  3. Keep an archive of four nodes; return the earliest evaluated node whose
     steering-split solve count is maximal.
  Baselines: Huxley-Godel Machine (HGM) selection, parent-rate ablation, root,
  calibration-best edit.

Adaptation:
  * "scaffold edit" -> edit to the battery policy program (gridwise.edits)
  * "solve rate"    -> mean fraction of oracle savings captured (continuous);
                       "solve count" -> days with capture >= 0.80
  * 16 fit / 48 steering / 100 held-out days (same sizes as the paper's HumanEval+ split)
  * evaluations are deterministic simulations, so each edit is labelled once
    (the paper averages four stochastic LLM evaluations)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..edits import DIFF_FEATURE_NAMES, DOMAIN_FEATURE_NAMES, EditInfo, diff_features, domain_features, propose
from ..evaluate import Evaluator, Score
from ..policy import Policy

SELECTORS = ("readout", "parent_rate", "hgm", "random")


# ------------------------------------------------------------------ readout
class LinearReadout:
    """Ridge-regularised linear model on standardised inputs (pure NumPy)."""

    def __init__(self, names: Sequence[str], ridge: float = 1.0):
        self.names = list(names)
        self.ridge = ridge
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearReadout":
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.b = y.mean()
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - self.b))
        resid = y - self.predict(X)
        self.r2 = 1 - resid.var() / (y.var() + 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        return Z @ self.w + self.b

    def importance(self):
        """Standardised coefficients, largest |w| first."""
        return sorted(zip(self.names, self.w), key=lambda kv: -abs(kv[1]))


def featurize(parent_rate: float, parent: Policy, child: Policy, feature_set: str = "diff") -> np.ndarray:
    parts = [np.array([parent_rate])]
    if feature_set in ("diff", "diff+domain"):
        parts.append(diff_features(parent, child))
    if feature_set == "diff+domain":
        parts.append(domain_features(parent, child))
    return np.concatenate(parts)


def feature_names(feature_set: str = "diff"):
    names = ["parent_rate"]
    if feature_set in ("diff", "diff+domain"):
        names += list(DIFF_FEATURE_NAMES)
    if feature_set == "diff+domain":
        names += list(DOMAIN_FEATURE_NAMES)
    return names


# -------------------------------------------------------------- calibration
@dataclass
class Calibration:
    readout: LinearReadout
    X: np.ndarray
    y: np.ndarray
    parents: List[Policy]
    children: List[Policy]
    best_child: Policy          # "calibration-best edit" baseline
    best_child_fit: float


def calibrate(ev: Evaluator, root: Policy, rng: np.random.Generator, n_edits: int = 96,
              feature_set: str = "diff", proposer: Callable = propose) -> Calibration:
    """Label `n_edits` random edits on the 16-day fit split and fit the readout once."""
    root_fit = ev.score(root, "fit", purpose="calibration").mean
    pool = [root]
    X, y, parents, children = [], [], [], []
    for _ in range(n_edits):
        parent = pool[int(rng.integers(len(pool)))]
        child, _info = proposer(parent, rng)
        ps = ev.score(parent, "fit", purpose="calibration").mean
        cs = ev.score(child, "fit", purpose="calibration").mean
        X.append(featurize(ps, parent, child, feature_set)); y.append(cs)
        parents.append(parent); children.append(child)
        if cs >= root_fit - 0.15:          # keep plausible programs as future parents
            pool.append(child)
    X, y = np.array(X), np.array(y)
    readout = LinearReadout(feature_names(feature_set)).fit(X, y)
    k = int(np.argmax(y))
    return Calibration(readout, X, y, parents, children, children[k], float(y[k]))


# -------------------------------------------------------------------- search
@dataclass
class Node:
    policy: Policy
    score: Score                    # steering-split score
    parent: Optional[int]
    order: int
    edit: Optional[EditInfo] = None
    predicted: Optional[float] = None


@dataclass
class SearchResult:
    nodes: List[Node]
    best: Node
    proposals_scored: int
    log: List[str] = field(default_factory=list)


def search(ev: Evaluator, root: Policy, selector: str, rng: np.random.Generator,
           readout: Optional[LinearReadout] = None, feature_set: str = "diff",
           archive_size: int = 4, proposals_per_node: int = 4, proposer: Callable = propose) -> SearchResult:
    """Grow an archive of `archive_size` evaluated nodes, expanding ONE proposal per step."""
    assert selector in SELECTORS
    nodes = [Node(root, ev.score(root, "steering"), None, 0)]
    seen = {root.key}
    scored = 0
    log = []
    while len(nodes) < archive_size:
        pool = []
        for i, nd in enumerate(nodes):
            for _ in range(proposals_per_node):
                child, info = proposer(nd.policy, rng)
                if child.key not in seen:
                    pool.append((i, child, info))
        if not pool:
            break
        if selector == "readout":
            X = np.array([featurize(nodes[i].score.mean, nodes[i].policy, c, feature_set) for i, c, _ in pool])
            pred = np.round(readout.predict(X), 9)   # identical diffs tie exactly; break ties by order
            scored += len(pool)
            k = int(np.argmax(pred)); predicted = float(pred[k])
        elif selector == "parent_rate":
            rates = np.array([nodes[i].score.mean for i, _, _ in pool])
            k = int(np.flatnonzero(rates == rates.max())[0]); predicted = None
        elif selector == "hgm":
            # Thompson sampling on steering solve counts of evaluated nodes
            draws = [rng.beta(1 + nd.score.solved, 1 + nd.score.n - nd.score.solved) for nd in nodes]
            par = int(np.argmax(draws))
            mine = [j for j, (i, _, _) in enumerate(pool) if i == par] or list(range(len(pool)))
            k = mine[0]; predicted = None
        else:
            k = int(rng.integers(len(pool))); predicted = None
        i, child, info = pool[k]
        s = ev.score(child, "steering")
        nodes.append(Node(child, s, i, len(nodes), info, predicted))
        seen.add(child.key)
        log.append(f"expand node {len(nodes)-1} <- parent {i} via {info.operator:15s} "
                   f"pred={'-' if predicted is None else f'{predicted:.3f}'} steering={s.mean:.3f} solved={s.solved}/{s.n}")
    top = max(nd.score.solved for nd in nodes)
    best = next(nd for nd in nodes if nd.score.solved == top)   # earliest with maximal solve count
    return SearchResult(nodes, best, scored, log)
