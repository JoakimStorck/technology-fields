"""
run_all.py (experiment/)
------------------------
Run every dynamic producer d00..d12 in order, each in its own process, and
stop at the first failure. Every producer asserts its own frozen baselines,
so a clean pass certifies that every number in the manuscript's dynamic
sections is reproducible from the current tree. The static pipeline
(scripts/run_all.py) is a separate certification and is not run from here.

Default order and approximate runtimes:
  d00  zero-field anchor        the anchoring guard: L0 is a rest point of
                                the zero-technology sorting map (~2 min)
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
  cld  structure audit          extracts the dependency graph from the core
                                and re-emits the stock-flow figure; fails on
                                any figure edge without a code witness (~1 s)

A full pass is on the order of 35-45 minutes. After it, run
paper/sync_figures.py to refresh the manuscript figures.

Usage:
    python experiment/run_all.py                # d00..d12
    python experiment/run_all.py --only d10     # one producer
    python experiment/run_all.py --from d07     # d07 onward
    python experiment/run_all.py --recalibrate  # collection pass, see below

Recalibration. After a deliberate calibration change (e.g. the anchored
sorting kernel), every frozen-baseline assert is EXPECTED to fail, which is
correct for certification but blocks collecting the new numbers: the
producers crash at the assert before writing their summaries. The
--recalibrate flag runs each producer under `python -O`, which disables
asserts entirely, so every producer runs to completion and writes its
summary with the NEW numbers to experiment/results/. The pass certifies
nothing: frozen-baseline checks AND structural sanity asserts (population
conservation, U draining) are both off, and any summary line that says
"asserted" refers to the plain pass. Workflow: run --recalibrate, freeze
the new numbers into each producer's baseline constants, then run the
plain pass, which must go clean before any number reaches the manuscript.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent

PIPELINE = [
    ("d00", "d00_zero_field_anchor.py"),
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
    ("cld", "cld/build_figure.py"),
]


def run(script: Path, recalibrate: bool = False) -> int:
    cmd = [sys.executable] + (["-O"] if recalibrate else []) + [str(script)]
    print(f"\n{'=' * 70}\n>>> {script.name}"
          + ("   [RECALIBRATE: asserts off]" if recalibrate else "")
          + f"\n{'=' * 70}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single producer, e.g. d10")
    ap.add_argument("--from", dest="from_",
                    help="run from this producer onward, e.g. d07")
    ap.add_argument("--recalibrate", action="store_true",
                    help="collection pass after a deliberate calibration "
                         "change: run every producer under python -O so "
                         "frozen-baseline asserts do not stop the pass; "
                         "certifies nothing")
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
    if args.recalibrate:
        print("=" * 70 + "\nRECALIBRATION PASS: asserts are OFF (python -O). "
              "Summaries in\nexperiment/results/ carry the NEW numbers; nothing "
              "is certified.\nFreeze the new baselines, then run the plain pass."
              + "\n" + "=" * 70, flush=True)
    timings: list[tuple[str, float]] = []
    for key, fname in steps:
        t1 = time.time()
        rc = run(EXPERIMENT / fname, recalibrate=args.recalibrate)
        timings.append((key, time.time() - t1))
        if rc != 0:
            if args.recalibrate:
                print(f"\n{key} FAILED (exit {rc}); stopping. Asserts are off "
                      f"in this mode, so this is a genuine error, not a "
                      f"frozen-baseline drift.", flush=True)
            else:
                print(f"\n{key} FAILED (exit {rc}); stopping. A failed frozen-"
                      f"baseline assert means the tree no longer reproduces the "
                      f"manuscript.", flush=True)
            sys.exit(rc)

    print(f"\n{'=' * 70}")
    for key, dt in timings:
        print(f"  {key}  {dt / 60:5.1f} min")
    if args.recalibrate:
        print(f"recalibration pass done in {(time.time() - t0) / 60:.1f} min; "
              f"no baseline was checked.")
        print("next: freeze the new numbers into the producers' baseline "
              "constants,\nthen run the plain pass: python experiment/run_all.py")
    else:
        print(f"all {len(steps)} producers passed in {(time.time() - t0) / 60:.1f} "
              f"min; every frozen baseline holds.")
        print("next: python paper/sync_figures.py")


if __name__ == "__main__":
    main()
