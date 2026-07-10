"""
run_all.py
----------
Run the analysis pipeline end to end. Every script reads exclusively from
data/ (frozen by scripts/00_freeze_inputs.py; provenance in data/MANIFEST.json),
so the pipeline is self-contained once the inputs are frozen.

Default order:
  01  wage field            estimate the price-field coefficients ln Pi
                            (replicates Paper 1, Table 3)
  02  price field           bundle pricing, support diagnostics, regime demo
  03  field vs Paper 1      directional-return cross-check
  05  family wedge          measured family wage wedge eta_g
  06  capability fields     rank-2 plane test and the q_k coefficients
  07  build exposure        freeze Eloundou beta -> data/onet_task_exposure.csv
  08  calibrate technology  phi_K against the task-level exposure surface;
                            records the exposure provenance in data/MANIFEST.json
  23  webb fields           Webb (2020) robot/software/ML fields fitted on the
                            occupation substrate beside Eloundou; carries the
                            robot-field centre and reach (the paper's Webb table and figure)
  26  freeze OEWS history   freezes the 1999/2003/2007 national wage window
                            (SOC-2000 through composed crosswalks) to
                            data/oews_history_wages.csv; inputs committed
  27  price field history   per-vintage price-field estimates Pi_1999..Pi_2007
                            beside the committed Pi_2023; the balanced-
                            composition shape-stability result
  09  equilibrium regime    worker-layer equilibrium: re-sorting, labour
                            share, candidate map, operated-regime illustration
  28  robot era equilibrium the Webb-located robot field through the full
                            equilibrium at Pi_1999 with 1999 employment,
                            gate scale on the A&R displacement moment; the
                            93-percent unbound result and the script-29
                            collinearity gate
  29  robot era directional bundle wage pressure vs observed wage growth
                            in each wave's own window; the four-cell
                            {robot, cognitive} x {1999-2007, 2019-2025}
                            table with placebos and the software horse
                            race
  10  demand channel        two fields through the economy; the eta sweep
                            and the second-order automation margin (Bessen)
  11  centroid-shift test   displacement-channel consistency vs OEWS 2019-2025
                            wage changes, confound analysis, window split
                            (needs OEWS medians in data/)
  12  price microfoundation derive Pi as a capability-assignment equilibrium;
                            selects the demand closure (rejects the scarcity and
                            productivity-complementarity accounts)
  14  sensitivity           attachment/gate/shape sweeps behind the headline
                            robustness ranges (labour share, unbound, re-sort)
  15  geometry map          occupational centroid map with poles and examples
                            (the geometry-map panel of the paper's first figure)
  24  bundle examples       five occupations opened as task bundles
                            (the bundle panel of the paper's first figure)
  25  bundle examples num.  numbered variant of 24 plus the task longtable
                            (paper appendix figure and table)
  16  placebo fields        pre-registered placebo test: robot field and the
                            rotated cognitive clone vs OEWS wage changes
  17  unbound decomposition pre-registered gamma-vs-ell decomposition: the
                            unbound share is a binding property, not seeding
  19  wage deformation      exact stripping/congestion/reinstatement
                            decomposition of occupation wage adjustments
  20  wage objects          consistency chain proj -> bundle -> value against
                            OEWS wage growth
  21  startup seeding       embeds the YC AI/robotics startups through the
                            frozen projection into the geometry
  22  startup enrichment    seeding-ring enrichment statistics and the
                            comparison figure of the startup section

Notes:
  - 00 (freeze inputs) is NOT in the default run. It vendors the Paper 1
    exports from a local geometry-of-work checkout and needs --geometry-root,
    so it is run on its own when refreshing the frozen inputs (or here via
    --freeze --geometry-root PATH, which runs 00 first).
  - Retired steps leave numbering gaps: 04 (residual diagnostic), 13
    (occupation-rent extension), 18 (binding counterfactual). See the
    PIPELINE comment and git history.
  - 07 and 08 require the Eloundou labelset data/full_labelset.tsv
    (github.com/openai/GPTs-are-GPTs). It is committed in data/ and calls no
    API, so 07 now runs anywhere as part of the pipeline.
  - 23 requires Webb's exposure file (data/webb2020_exposure.csv or a Stata
    file such as data/final_df_out.dta); without it the Webb side is skipped
    and only the Eloundou occupation-substrate fit is produced.
  - 26 rebuilds data/oews_history_wages.csv from the committed OEWS national
    files (1999, May 2003, May 2007) and SOC crosswalks in data/; like 07 it
    writes into data/ and the output is committed. 27 reads it; 28 reads 27
    and 23 and runs the anchored equilibrium in both eras (SMOKE=1 gives a
    coarse-grid mechanics check, never a result).
  - 21 and 22 place the AI startup ecosystem in the geometry. 21 needs
    data/geometry_projection.npz (recover_projection.py) and
    data/startups_ycombinator.csv (fetch_startups.py); its first run embeds via
    OpenAI (OPENAI_API_KEY) and caches to results/, so later runs need no API.
    22 reads 21's positions. The data-prep helpers (recover_projection.py,
    fetch_startups.py) are run on their own, like 00.

Usage:
    python scripts/run_all.py                          # full pipeline
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

# Numbering is a stable identity, not a position. Retired steps leave gaps by
# design: 04 (residual diagnostic, superseded by the assignment appendix),
# 13 (occupation-rent extension, deferred to separate work), 18 (binding
# counterfactual, an internal robustness check with no manuscript claim). See
# git history to revive any of them.
PIPELINE = [
    ("01", "01_wage_field.py"),
    ("02", "02_price_field.py"),
    ("03", "03_field_vs_paper1.py"),
    ("05", "05_family_wedge.py"),
    ("06", "06_capability_fields.py"),
    ("07", "07_build_exposure.py"),
    ("08", "08_calibrate_technology.py"),
    ("23", "23_webb_fields.py"),
    ("26", "26_freeze_oews_history.py"),
    ("27", "27_price_field_history.py"),
    ("09", "09_equilibrium_regime.py"),
    ("28", "28_robot_era_equilibrium.py"),
    ("29", "29_robot_era_directional.py"),
    ("10", "10_demand_channel.py"),
    ("11", "11_centroid_shift_test.py"),
    ("12", "12_price_microfoundation.py"),
    ("14", "14_sensitivity.py"),
    ("15", "15_geometry_map.py"),
    ("24", "24_bundle_examples.py"),
    ("25", "25_bundle_examples_numbered.py"),
    ("16", "16_placebo_field.py"),
    ("17", "17_unbound_decomposition.py"),
    ("19", "19_wage_field_deformation.py"),
    ("20", "20_wage_object_consistency.py"),
    ("21", "21_startup_seeding.py"),
    ("22", "22_startup_field_enrichment.py"),
]

LABELSET_DEPENDENT = {"07", "08"}
WEBB_DEPENDENT = {"23"}
OEWS_DEPENDENT = {"11", "16", "20"}
HISTORY_DEPENDENT = {"26"}
STARTUP_DEPENDENT = {"21", "22"}


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

    if any(k in WEBB_DEPENDENT for k, _ in steps):
        data_dir = SCRIPTS.parent / "data"
        candidates = [data_dir / "webb2020_exposure.csv",
                      data_dir / "webb2020_exposure.dta",
                      data_dir / "final_df_out.dta"]
        if not any(p.exists() for p in candidates):
            print("WARNING: no Webb exposure file found in data/ "
                  "(webb2020_exposure.csv/.dta or final_df_out.dta); "
                  "step 23 will skip the Webb side and fit only the "
                  "Eloundou occupation-substrate field.")

    if any(k in OEWS_DEPENDENT for k, _ in steps):
        oews = SCRIPTS.parent / "data" / "national_M2019_dl.xlsx"
        if not oews.exists():
            print(f"WARNING: {oews} not found; steps 11/16/20 need the BLS OEWS "
                  "national medians (2019, 2024, 2025) in data/.")

    if any(k in HISTORY_DEPENDENT for k, _ in steps):
        data_dir = SCRIPTS.parent / "data"
        needed = ["national_1999_dl.xls", "national_may2003_dl.xls",
                  "national_May2007_dl.xls", "soc_2000_to_2010_crosswalk.xls",
                  "soc_2010_to_2018_crosswalk.xlsx"]
        missing = [n for n in needed if not (data_dir / n).exists()]
        if missing:
            print(f"WARNING: {', '.join(missing)} not found; step 26 needs "
                  "the historical OEWS national files and SOC crosswalks in "
                  "data/ (see its docstring for sources). Steps 27 and 28 "
                  "read 26's frozen output.")

    if any(k in STARTUP_DEPENDENT for k, _ in steps):
        proj = SCRIPTS.parent / "data" / "geometry_projection.npz"
        corpus = SCRIPTS.parent / "data" / "startups_ycombinator.csv"
        missing = [p.name for p in (proj, corpus) if not p.exists()]
        if missing:
            print(f"WARNING: {', '.join(missing)} not found; step 21 needs the "
                  "projection basis (recover_projection.py) and the YC corpus "
                  "(fetch_startups.py), and OPENAI_API_KEY for the first embed "
                  "(cached thereafter). Step 22 reads 21's output.")

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