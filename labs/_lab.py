"""Small helpers shared by every lab script (printing, saving evidence)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Lab scripts print ASCII only so they run in any Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAST = os.environ.get("GRIDWISE_FAST") == "1"   # fewer seeds for a quick smoke test


def seeds(n: int) -> range:
    return range(max(2, n // 4) if FAST else n)


def banner(lab: str, title: str, paper: str = ""):
    print("=" * 78)
    print(f"LAB {lab} | {title}")
    if paper:
        print(f"source: {paper}")
    print("=" * 78)
    return time.time()


def section(title: str):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def table(rows, headers):
    """Print a GitHub-markdown table (also readable in a terminal)."""
    rows = [[f"{c:.3f}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")


def ms(xs) -> str:
    import numpy as np
    xs = np.asarray(xs, float)
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    return f"{xs.mean():.3f} +/- {sd:.3f}"


def save(lab: str, data: dict, t0: float = None):
    if t0 is not None:
        data["runtime_seconds"] = round(time.time() - t0, 2)
    data["fast_mode"] = FAST
    path = RESULTS / f"lab{lab.replace('.', '_')}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    print(f"\n[evidence saved to {path.relative_to(ROOT)}]")
    if t0 is not None:
        print(f"[runtime {time.time() - t0:.1f}s]")


def load(lab: str) -> dict:
    path = RESULTS / f"lab{lab.replace('.', '_')}.json"
    with open(path) as f:
        return json.load(f)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
