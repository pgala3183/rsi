"""Run every Section 10 codelab in order and keep the console output as evidence.

    python run_all.py            # full run (~5 minutes on a laptop CPU)
    python run_all.py --fast     # fewer seeds, quick smoke test
    python run_all.py 10.06      # just one lab (or several: 10.03 10.04)

Logs go to results/logs/lab10_XX.txt, structured results to results/lab10_XX.json.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
LABS = sorted(ROOT.glob("labs/g0*/lab10_*/run.py"))


def lab_id(p: Path) -> str:
    return p.parent.name.split("_")[0].replace("lab", "") + "." + p.parent.name.split("_")[1]


def main(argv):
    fast = "--fast" in argv
    wanted = [a for a in argv if a[0].isdigit()]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if fast:
        env["GRIDWISE_FAST"] = "1"
    LOGS.mkdir(parents=True, exist_ok=True)
    summary = []
    for p in LABS:
        lid = lab_id(p)
        if wanted and lid not in wanted:
            continue
        t = time.time()
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        dt = time.time() - t
        log = LOGS / f"lab{lid.replace('.', '_')}.txt"
        log.write_text(r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else ""), encoding="utf-8")
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        summary.append((lid, p.parent.parent.name, status, dt))
        print(f"lab {lid:6s} {status:12s} {dt:6.1f}s  -> {log.relative_to(ROOT)}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
    failed = [s for s in summary if s[2] != "ok"]
    print(f"\n{len(summary) - len(failed)}/{len(summary)} labs ok, total {sum(s[3] for s in summary):.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
