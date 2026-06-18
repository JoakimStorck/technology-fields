"""
run_all.py
----------
Run the analysis pipeline end to end. Every script reads exclusively from
data/ (frozen by scripts/00_freeze_inputs.py; provenance in data/MANIFEST.json),
so the pipeline is self-contained once the inputs are frozen.

Default order (01..08):
  01  wage field            estimate the price-field coefficients ln Pi
                            (replicates Paper 1, Table 3)
  02  price field           bundle pricing, support diagnostics, regime demo
  03  field vs Paper 1      directional-return cross-check
  04  residual structure    spatial structure above the field (wedge motivation)
  05  family wedge          measured family wage wedge eta_g
  06  capability fields     rank-2 plane test and the q_k coefficients
  07  build exposure        freeze Eloundou beta -> data/onet_task_exposure.csv
  08  calibrate technology  phi_K against the exposure surface; records the
                            exposure provenance in data/MANIFEST.json

Notes:
  - 00 (freeze inputs) is NOT in the default run. It vendors the Paper 1
    exports from a local geometry-of-work checkout and needs --geometry-root,
    so it is run on its own when refreshing the frozen inputs (or here via
    --freeze --geometry-root PATH, which runs 00 first).
  - 07 and 08 require the Eloundou labelset data/full_labelset.tsv
    (github.com/openai/GPTs-are-GPTs). It is committed in data/ and calls no
    API, so 07 now runs anywhere as part of the pipeline.

Usage:
    python scripts/run_all.py                          # 01..08
    python scripts/run_all.py --only 08                # one script
    python scripts/run_all.py --from 07                # 07 onward
    python scripts/run_all.py --freeze --geometry-root PATH   # prepend 00
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

PIPELINE = [
    ("01", "01_wage_field.py"),
    ("02", "02_price_field.py"),
    ("03", "03_field_vs_paper1.py"),
    ("04", "04_residual_structure.py"),
    ("05", "05_family_wedge.py"),
    ("06", "06_capability_fields.py"),
    ("07", "07_build_exposure.py"),
    ("08", "08_calibrate_technology.py"),
]

LABELSET_DEPENDENT = {"07", "08"}


def run(script: Path, args: list[str] | None = None) -> int:
    cmd = [sys.executable, str(script)] + (args or [])
    print(f"\n{'=' * 70}\n>>> {script.name}\n{'=' * 70}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single step, e.g. 08")
    ap.add_argument("--from", dest="from_", help="run from this step onward, e.g. 07")
    ap.add_argument("--freeze", action="store_true",
                    help="run 00_freeze_inputs.py first (needs --geometry-root)")
    ap.add_argument("--geometry-root", type=Path,
                    help="path to a geometry-of-work checkout (for --freeze)")
    args = ap.parse_args()

    steps = PIPELINE
    if args.only:
        steps = [s for s in PIPELINE if s[0] == args.only.zfill(2)]
        if not steps:
            ap.error(f"--only {args.only}: no such step")
    elif args.from_:
        keys = [s[0] for s in PIPELINE]
        start = args.from_.zfill(2)
        if start not in keys:
            ap.error(f"--from {args.from_}: no such step")
        steps = PIPELINE[keys.index(start):]

    if args.freeze:
        if not args.geometry_root:
            ap.error("--freeze needs --geometry-root PATH")
        rc = run(SCRIPTS / "00_freeze_inputs.py",
                 ["--geometry-root", str(args.geometry_root)])
        if rc != 0:
            sys.exit(f"00_freeze_inputs.py failed (exit {rc})")

    if any(k in LABELSET_DEPENDENT for k, _ in steps):
        labelset = SCRIPTS.parent / "data" / "full_labelset.tsv"
        if not labelset.exists():
            print(f"WARNING: {labelset} not found; steps 07/08 need the "
                  "Eloundou labelset (github.com/openai/GPTs-are-GPTs).")

    results: list[tuple[str, str, int]] = []
    for key, name in steps:
        rc = run(SCRIPTS / name)
        results.append((key, name, rc))
        if rc != 0:
            print(f"\n{name} failed (exit {rc}); stopping.")
            break

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for key, name, rc in results:
        print(f"  {key}  {name:32s}  {'OK' if rc == 0 else f'FAIL ({rc})'}")
    if any(rc != 0 for _, _, rc in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
