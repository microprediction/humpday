"""Weekend experiment orchestrator (Tier A, CPU-only).

Runs E1-E5 sequentially. Each experiment is itself crash-safe (atomic per-step
checkpoints), and this runner records completion in runs/weekend_manifest.json so a
restart after a crash/outage skips finished experiments and resumes the rest. Run:

    ../../.venv/bin/python weekend_runner.py            # full weekend run
    ../../.venv/bin/python weekend_runner.py --quick    # tiny smoke of the whole pipeline
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

MANIFEST = Path("runs/weekend_manifest.json")

# (id, script, full-run args, output json)
EXPERIMENTS = [
    ("E1_rankcorr", "rankcorr.py",
     ["--seeds", "0,1,2", "--synth-dims", "5,20,40", "--real-demos", "30",
      "--budgets", "60,120,240", "--out", "runs/rankcorr.json"], "runs/rankcorr.json"),
    ("E3_crossover", "crossover_harness.py",
     ["--demos", "24", "--seeds", "0,1,2", "--trials", "120", "--out", "runs/crossover.json"],
     "runs/crossover.json"),
    ("E2_hardening", "e2_hardening.py",
     ["--demos", "20", "--seeds", "0,1,2,3,4", "--budgets", "60,120,240,480",
      "--out", "runs/discovered_hardened.json"], "runs/discovered_hardened.json"),
    ("E4_schur_final", "overnight_schur2.py", [], "runs/schur_final.json"),
    ("E5_generalisation", "e5_generalisation.py",
     ["--train-demos", "18", "--test-demos", "18", "--generations", "25",
      "--out", "runs/surrogate_generalisation.json"], "runs/surrogate_generalisation.json"),
]


def load_manifest():
    if MANIFEST.exists():
        try:
            return json.load(open(MANIFEST))
        except Exception:  # noqa: BLE001
            pass
    return {"completed": [], "started": None}


def save_manifest(m):
    tmp = Path(str(MANIFEST) + ".tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated experiment ids to run")
    a = ap.parse_args()
    py = sys.executable
    only = set(a.only.split(",")) if a.only else None

    m = load_manifest()
    print(f"weekend runner: {len(m['completed'])} already complete: {m['completed']}\n", flush=True)
    for eid, script, args, out in EXPERIMENTS:
        if only and eid not in only:
            continue
        if eid in m["completed"]:
            print(f"== SKIP {eid} (already complete) ==", flush=True)
            continue
        run_args = ["--quick"] if a.quick else list(args)
        # overnight_schur2 has no --quick / --out flags; run it bare
        if script == "overnight_schur2.py":
            run_args = []
        m["started"] = eid
        save_manifest(m)
        log = Path(f"runs/{eid}.log")
        print(f"== RUN {eid}: {script} {' '.join(run_args)} -> {log} ==", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call([py, script, *run_args], stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc == 0:
            m["completed"].append(eid)
            m["started"] = None
            save_manifest(m)
            print(f"== DONE {eid} in {dt/60:.1f} min ==\n", flush=True)
        else:
            print(f"== FAIL {eid} (rc={rc}) after {dt/60:.1f} min; see {log}. Continuing. ==\n",
                  flush=True)
    print("weekend runner finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
