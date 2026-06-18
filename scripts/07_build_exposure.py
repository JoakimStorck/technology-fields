"""
07_build_exposure.py
--------------------
Freezes a task-level AI-exposure field over the O*NET task space as a
frozen input for the technology calibration.

A technology in this model is a field phi_K over the task disk. AI is
one instance of such a technology. We take the empirical AI-exposure
phi from Eloundou et al. (2024, "GPTs are GPTs", Science), who rate
every O*NET task for LLM exposure using a rubric applied by both human
annotators and GPT-4. We use their continuous `beta` measure
(E1 + 0.5*E2), defined directly on O*NET tasks in English. Because the
measure already lives on O*NET Task IDs, it joins onto the geometry
1:1 with no crosswalk, no translation, and no embedding transfer.

Source: https://github.com/openai/GPTs-are-GPTs  (data/full_labelset.tsv)
Join coverage: 17549/17606 geometry tasks (99.7%); the 57 unmatched
tasks are not systematic (their chi distribution matches the whole
population). The beta surface reproduces the Paper 1 geometry: exposure
rises toward the cognitive/office directions and falls with amplitude
chi, monotone across sectors from ~0.55 to ~0.07.

This script is deliberately trivial: read the published file, select
the measure, normalise the key, write the frozen field. It calls no
API and can run anywhere. (The earlier cross-lingual transfer pipeline
from ILO/Gmyrek Polish task scores is retired; Gmyrek phi may be kept
as an optional robustness correlation but is not part of this build.)

Output: data/onet_task_exposure.csv with columns
    onet_code, task_id, phi

Usage:
    python scripts/07_build_exposure.py \
        --labelset /path/to/full_labelset.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

# Eloundou's continuous exposure measure: E1 + 0.5*E2, in [0, 1].
EXPOSURE_COL = "beta"


def load_labelset(path: Path) -> pd.DataFrame:
    """Read Eloundou full_labelset.tsv and return onet_code, task_id, phi."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    needed = {"O*NET-SOC Code", "Task ID", EXPOSURE_COL}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"labelset missing columns: {miss}; "
                         f"found {list(df.columns)}")
    out = pd.DataFrame({
        "onet_code": df["O*NET-SOC Code"].astype(str).str.strip(),
        "task_id": pd.to_numeric(df["Task ID"], errors="coerce").astype("Int64"),
        "phi": pd.to_numeric(df[EXPOSURE_COL], errors="coerce"),
    })
    out = out.dropna(subset=["task_id", "phi"]).reset_index(drop=True)
    # one row per task
    out = out.drop_duplicates(subset="task_id", keep="first").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labelset", type=Path,
                    default=DATA / "full_labelset.tsv",
                    help="Eloundou full_labelset.tsv "
                         "(github.com/openai/GPTs-are-GPTs, data/). "
                         "Defaults to data/full_labelset.tsv.")
    args = ap.parse_args()

    print("Eloundou labelset:", args.labelset)
    out = load_labelset(args.labelset)

    DATA.mkdir(exist_ok=True)
    fp = DATA / "onet_task_exposure.csv"
    out.to_csv(fp, index=False)

    phi = out["phi"]
    print(f"wrote {fp}  ({len(out)} tasks)")
    print(f"  phi (Eloundou beta): mean {phi.mean():.3f}, sd {phi.std():.3f}, "
          f"min {phi.min():.3f}, max {phi.max():.3f}")
    print("  geometry join happens downstream in the calibration "
          "(on task_id; ~99.7% coverage expected).")


if __name__ == "__main__":
    main()