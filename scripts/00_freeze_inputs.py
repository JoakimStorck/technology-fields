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
  occupation_cluster_intensity.csv         DERIVED: mean descriptor level per
                                           occupation within each capability
                                           cluster S1, S2, A1, A2 (replicates
                                           Paper 1, notebook 5 cell 18, from
                                           the cluster membership and overlay
                                           exports of the reference run)
  occupation_skills_levels.csv             DERIVED: wide occupation x 35
                                           Skills matrix of raw levels
  occupation_abilities_levels.csv          DERIVED: wide occupation x 52
                                           Abilities matrix of raw levels
                                           (both pivoted from the overlay
                                           long exports; inputs to the rank-2
                                           test and capability-plane fit)

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
MEMBERSHIP_REL = RUN_EXPORTS / "gradient_compass_clusters_membership.csv"
SKILLS_LONG_REL = RUN_EXPORTS / "skills_overlay_long.csv"
ABILITIES_LONG_REL = RUN_EXPORTS / "abilities_overlay_long.csv"


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


def derive_descriptor_matrix(root: Path, rel: Path, col: str) -> pd.DataFrame:
    """Wide occupation x descriptor matrix of raw levels ('value'),
    pivoted from the overlay long export. One row per occupation,
    one column per descriptor (35 Skills or 52 Abilities). Used by the
    rank-2 test and the capability-plane estimation in
    scripts/06_capability_fields.py."""
    long_df = pd.read_csv(root / rel)
    wide = (long_df.pivot(index="onet_code", columns=col, values="value")
            .reset_index())
    wide.columns.name = None
    return wide


def derive_cluster_intensity(root: Path) -> pd.DataFrame:
    """Mean descriptor level per occupation within each capability cluster
    (S1, S2, A1, A2), replicating Paper 1 notebook 5, cell 18: cluster_rank
    1 = social/cognitive pole, 2 = technical/physical pole; prefix S for
    skills, A for abilities; intensity = mean of raw 'value' over the
    cluster's descriptors."""
    membership = pd.read_csv(root / MEMBERSHIP_REL)
    membership["cluster"] = membership.apply(
        lambda r: ("S" if r["label"] == "skill" else "A")
                  + str(r["cluster_rank"]), axis=1)
    out = []
    for label, rel, col in [("skill", SKILLS_LONG_REL, "skill"),
                            ("ability", ABILITIES_LONG_REL, "ability")]:
        long_df = pd.read_csv(root / rel)
        m = membership.loc[membership["label"] == label, ["name", "cluster"]]
        merged = long_df.merge(m, left_on=col, right_on="name", how="inner")
        out.append(merged.groupby(["onet_code", "cluster"])["value"]
                   .mean().unstack("cluster"))
    return out[0].join(out[1], how="inner").reset_index()


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

    ci = derive_cluster_intensity(root)
    ci.to_csv(DATA / "occupation_cluster_intensity.csv", index=False)
    entries["occupation_cluster_intensity.csv"] = {
        "source_path": [str(MEMBERSHIP_REL), str(SKILLS_LONG_REL),
                        str(ABILITIES_LONG_REL)],
        "sha256_source": [sha256(root / MEMBERSHIP_REL),
                          sha256(root / SKILLS_LONG_REL),
                          sha256(root / ABILITIES_LONG_REL)],
        "derived": True,
        "recipe": ("Per occupation, mean raw descriptor 'value' within each "
                   "capability cluster S1/S2/A1/A2; clusters from "
                   "gradient_compass_clusters_membership (cluster_rank 1/2, "
                   "prefix S=skills, A=abilities). Replicates Paper 1 "
                   "notebook 5, cell 18."),
    }
    print(f"derived occupation_cluster_intensity.csv ({len(ci)} occupations)")

    for fname, rel, col, label in [
        ("occupation_skills_levels.csv", SKILLS_LONG_REL, "skill", "Skills"),
        ("occupation_abilities_levels.csv", ABILITIES_LONG_REL, "ability",
         "Abilities"),
    ]:
        wide = derive_descriptor_matrix(root, rel, col)
        wide.to_csv(DATA / fname, index=False)
        entries[fname] = {
            "source_path": str(rel),
            "sha256_source": sha256(root / rel),
            "derived": True,
            "recipe": (f"Pivot of the {label} overlay long export to a wide "
                       "occupation x descriptor matrix of raw levels "
                       "('value'); one row per onet_code, one column per "
                       "descriptor."),
        }
        print(f"derived {fname} ({wide.shape[0]} occupations x "
              f"{wide.shape[1] - 1} descriptors)")

    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repository": "https://github.com/JoakimStorck/geometry-of-work",
        "source_commit": git_commit(root),
        "encoder_run_tag": RUN_TAG,
        "files": entries,
    }
    # External inputs (e.g. the Eloundou exposure file, recorded by script 08)
    # have a different provenance kind from the Paper 1 freeze and live in a
    # separate top-level block. Carry any existing block forward so re-freezing
    # the Paper 1 inputs does not erase it.
    manifest_path = DATA / "MANIFEST.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if "external_inputs" in prior:
            manifest["external_inputs"] = prior["external_inputs"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest data/MANIFEST.json (source commit "
          f"{manifest['source_commit'][:12]})")


if __name__ == "__main__":
    main()
