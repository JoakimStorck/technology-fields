"""
run_all.py (experiment/)
------------------------
Run every dynamic producer d01..d12 in order, each in its own process, and
stop at the first failure. Every producer asserts its own frozen baselines,
so a clean pass certifies that every number in the manuscript's dynamic
sections is reproducible from the current tree. The static pipeline
(scripts/run_all.py) is a separate certification and is not run from here.

Default order and approximate runtimes:
  d01  calendar dynamics        the mismatch hump U(t) across tempos  (~1 min)
  d02  tempo regimes            headline destination result + gainer table
  d03  binding sensitivity      beta_m sweep; theta_L/theta_abs decomposition
  d04  survival gate            gate-off and comparative-advantage variants
  d05  binding law comparison   match-allocated vs size-multiplies
  d06  limit allocations        the homotopy sweep to the deep limits (~4 min)
  d07  sigma sweep              GE feedback across the assignment
                                elasticity (~8 min)
  d08  price feedback GE        sign, fixed point, contraction,
                                uniqueness (H4) (~2 min)
  d09  dynamic GE feedback      the dynamic close (~2 min)
  d10  capacity exponent        sublinear capacity sweep + generalised
                                Prop 2 verification (~7 min)
  d11  readiness update         the frozen-readiness release (~5 min)
  d12  baseline economy         the inherited-economy figure and asserts (<1 min)

A full pass is on the order of 35-45 minutes. After it, run
paper/sync_figures.py to refresh the manuscript figures.

Usage:
    python experiment/run_all.py                # d01..d12
    python experiment/run_all.py --only d10     # one producer
    python experiment/run_all.py --from d07     # d07 onward
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent

PIPELINE = [
    ("d01", "d01_calendar_dynamics.py"),
    ("d02", "d02_tempo_regimes.py"),
    ("d03", "d03_binding_sensitivity.py"),
    ("d04", "d04_survival_gate_robustness.py"),
    ("d05", "d05_binding_law_comparison.py"),
    ("d06", "d06_limit_allocations.py"),
    ("d07", "d07_sigma_sweep.py"),
    ("d08", "d08_price_feedback_ge.py"),
    ("d09", "d09_dynamic_gefeedback.py"),
    ("d10", "d10_capacity_exponent.py"),
    ("d11", "d11_readiness_update.py"),
    ("d12", "d12_baseline_economy.py"),
]


def run(script: Path) -> int:
    cmd = [sys.executable, str(script)]
    print(f"\n{'=' * 70}\n>>> {script.name}\n{'=' * 70}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single producer, e.g. d10")
    ap.add_argument("--from", dest="from_",
                    help="run from this producer onward, e.g. d07")
    args = ap.parse_args()

    steps = PIPELINE
    if args.only:
        steps = [(k, f) for k, f in PIPELINE if k == args.only]
        if not steps:
            sys.exit(f"unknown producer: {args.only}")
    elif args.from_:
        keys = [k for k, _ in PIPELINE]
        if args.from_ not in keys:
            sys.exit(f"unknown producer: {args.from_}")
        steps = PIPELINE[keys.index(args.from_):]

    t0 = time.time()
    timings: list[tuple[str, float]] = []
    for key, fname in steps:
        t1 = time.time()
        rc = run(EXPERIMENT / fname)
        timings.append((key, time.time() - t1))
        if rc != 0:
            print(f"\n{key} FAILED (exit {rc}); stopping. A failed frozen-"
                  f"baseline assert means the tree no longer reproduces the "
                  f"manuscript.", flush=True)
            sys.exit(rc)

    print(f"\n{'=' * 70}")
    for key, dt in timings:
        print(f"  {key}  {dt / 60:5.1f} min")
    print(f"all {len(steps)} producers passed in {(time.time() - t0) / 60:.1f} "
          f"min; every frozen baseline holds.")
    print("next: python paper/sync_figures.py")


if __name__ == "__main__":
    main()
