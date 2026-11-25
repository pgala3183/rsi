"""Terminal-style access to GridWise (the way a terminal agent would use it).

    python -m gridwise.cli evaluate my_policy.py --tariff tou --split steering
    python -m gridwise.cli show-seed

The policy file is parsed by the same sandbox as the EaaS layer.
"""
from __future__ import annotations

import argparse
import json
import sys

from .eaas import EaaS, SandboxError
from .env import TARIFFS, make_task
from .policy import EXPERT_POLICY, SEED_POLICY


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gridwise")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="score a policy file")
    e.add_argument("file")
    e.add_argument("--tariff", default="tou", choices=TARIFFS)
    e.add_argument("--split", default="steering", choices=("fit", "steering", "heldout"))
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--role", default="agent", choices=("agent", "auditor"))
    sub.add_parser("show-seed", help="print the starting program")
    sub.add_parser("show-expert", help="print the hand-written reference program")
    a = ap.parse_args(argv)
    if a.cmd == "show-seed":
        print(SEED_POLICY.source(), end="")
        return 0
    if a.cmd == "show-expert":
        print(EXPERT_POLICY.source(), end="")
        return 0
    with open(a.file) as f:
        src = f.read()
    svc = EaaS(make_task(a.tariff, a.seed), role=a.role)
    try:
        out = svc.evaluate(src, a.split)
    except SandboxError as err:
        print(json.dumps({"error": f"sandbox: {err}"}))
        return 2
    except Exception as err:                      # permission / budget errors
        print(json.dumps({"error": f"{type(err).__name__}: {err}"}))
        return 3
    out.update(tariff=a.tariff, split=a.split, day_sims_used=svc.used)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
