"""
00_freeze_inputs.py
-------------------
Vendor the Paper 1 inputs that the Paper 3 model fit depends on into this
repository, so that technology-fields is self-contained once frozen.

This is the ONLY script that reads from a geometry-of-work checkout. All
analysis scripts read exclusively from data/ in this repository.

Frozen inputs (data/):
  occupation_embeddings_polar_scaled.csv   occupation coordinates (xi, chi),
                                           reference encoder run
  task_embeddings_polar_scaled.csv         task coordinates, same run
                                           (bundles b_o(r) are built from these)
  national_M2023_dl.xlsx                   BLS OEWS May 2023 national wages
  occupation_rle.csv                       DERIVED: frequency-weighted mean
                                           Required Level of Education per
                                           occupation, computed here from the
                                           O*NET 30.1 "Education, Training,
                                           and Experience" table (pins the
                                           Paper 1 Mincer sample, N = 785)

Provenance is recorded in data/MANIFEST.json: source repository and commit,
encoder run tag, per-file source path and SHA-256, and the derivation recipe
for computed files. Re-running the freeze against a newer geometry-of-work
commit updates the manifest, so any drift in upstream data is explicit.

Usage:
    python scripts/00_freeze_inputs.py [--geometry-root PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

RUN_TAG = "embeddings__openai__text-embedding-3-large__d3072__year-2025__v30_1"
RUN_EXPORTS = Path("out/runs") / RUN_TAG / "exports"

COPY_FILES = {
    "occupation_embeddings_polar_scaled.csv":
        RUN_EXPORTS / "occupation_embeddings_polar_scaled.csv",
    "task_embeddings_polar_scaled.csv":
        RUN_EXPORTS / "task_embeddings_polar_scaled.csv",
    "national_M2023_dl.xlsx":
        Path("data/wages/national_M2023_dl.xlsx"),
}

ETE_REL = Path("data/onet/db_30_1/Education, Training, and Experience.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def derive_rle(ete_path: Path) -> pd.DataFrame:
    """Frequency-weighted mean Required Level of Education per occupation,
    replicating onet.education.rle_by_occupation in geometry-of-work."""
    ed = pd.read_csv(ete_path, sep="\t", na_values=["n/a"])
    ed.columns = ed.columns.str.strip()
    ed = ed[ed["Element Name"].astype(str).str.strip()
            == "Required Level of Education"].copy()
    ed["Category"] = pd.to_numeric(ed["Category"], errors="coerce")
    ed["Data Value"] = pd.to_numeric(ed["Data Value"], errors="coerce")
    ed = ed.dropna(subset=["O*NET-SOC Code", "Category", "Data Value"])
    g = ed.groupby("O*NET-SOC Code")
    return (g.apply(lambda d: np.average(d["Category"], weights=d["Data Value"]),
                    include_groups=False)
            .rename("rle_mean").reset_index()
            .rename(columns={"O*NET-SOC Code": "onet_code"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT.parent / "geometry-of-work",
        help="Path to a local checkout of JoakimStorck/geometry-of-work",
    )
    args = ap.parse_args()
    root = args.geometry_root

    DATA.mkdir(exist_ok=True)
    entries = {}

    for name, rel in COPY_FILES.items():
        src = root / rel
        dst = DATA / name
        shutil.copy2(src, dst)
        entries[name] = {
            "source_path": str(rel),
            "sha256_source": sha256(src),
            "derived": False,
        }
        print(f"frozen  {name}")

    rle = derive_rle(root / ETE_REL)
    rle.to_csv(DATA / "occupation_rle.csv", index=False)
    entries["occupation_rle.csv"] = {
        "source_path": str(ETE_REL),
        "sha256_source": sha256(root / ETE_REL),
        "derived": True,
        "recipe": ("Filter Element Name == 'Required Level of Education'; "
                   "rle_mean = weighted mean of Category with Data Value "
                   "weights, per O*NET-SOC Code (replicates "
                   "onet.education.rle_by_occupation)"),
    }
    print(f"derived occupation_rle.csv ({len(rle)} occupations)")

    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repository": "https://github.com/JoakimStorck/geometry-of-work",
        "source_commit": git_commit(root),
        "encoder_run_tag": RUN_TAG,
        "files": entries,
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest data/MANIFEST.json (source commit "
          f"{manifest['source_commit'][:12]})")


if __name__ == "__main__":
    main()
