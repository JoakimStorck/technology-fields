"""
31_freeze_manufacturing_share.py
--------------------------------
Freezes an occupation-level manufacturing employment share for the robot
window's trade-confound control (script 29). Robots and import
competition pressed the same western manufacturing arc over 1999-2007;
conditioning the window's wage test on how manufacturing-based each
occupation is asks whether robot pressure predicts wage growth BEYOND
an occupation's exposure to manufacturing-wide forces.

Source: BLS OES industry-specific occupational employment.
  Primary     data/oesm03in4/nat3d_may2003_dl.xls
              May 2003, the window midpoint, 3-digit NAICS: whole-economy
              industry coverage at the coarsest level, SOC-2000 codes.
  Robustness  data/oes02in4/nat4d_2002_dl.xls
              2002, 4-digit NAICS (optional; a second vintage and level).

Construction, per SOC-2000 occupation and file:
  mfg_share = sum of TOT_EMP over manufacturing industries (NAICS first
  two digits 31-33) / sum of TOT_EMP over ALL industries in the file.
  The denominator is the same file's industry sum, so numerator and
  denominator share the suppression regime; suppressed cells ('**') are
  missing and counted in the report. Shares are then pushed onto the
  frozen geometry's SOC-2018 codes through the composed 2000->2010->2018
  crosswalk (script 26's machinery), employment-weighted across SOC-2000
  sources.

No-go criterion (pre-stated): if fewer than 600 SOC-2000 codes carry a
denominator in the primary file, the control is not built and script 29
keeps the registered scope statement instead.

RESULTS (certified run): GO. 17,641 detailed industry-occupation rows
(1,210 suppressed dropped), 88 industries, 711 SOC-2000 occupations
with a denominator; median share 0.010, 207 occupations above 10
percent. The 2002 vintage rank-correlates with the May 2003 shares at
+0.975 (N=710): the measure is vintage-stable. 728 SOC-2018
occupations written from 751 SOC-2000 sources; script 29's coverage is
774/774.

Writes:
    data/occ_manufacturing_share.csv   soc2018, mfg_share (May 2003),
                                       mfg_share_2002 (if available),
                                       emp_total, n_sources
    MANIFEST.json                      external_inputs entry with hashes

Usage:
    python scripts/31_freeze_manufacturing_share.py
    python scripts/31_freeze_manufacturing_share.py --nat3d PATH [--nat4d PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frz = _load("26_freeze_oews_history.py")   # _norm, _to_num, read_xwalk, ...

DATA = REPO_ROOT / "data"
OUT = DATA / "occ_manufacturing_share.csv"
NAT3D = DATA / "oesm03in4" / "nat3d_may2003_dl.xls"
NAT4D = DATA / "oes02in4" / "nat4d_2002_dl.xls"
MIN_SOC2000 = 600


def read_industry_occupation(path: Path) -> pd.DataFrame:
    """(naics, OCC_CODE, emp) rows from an OES industry-specific file.
    Vintage-robust in 26's manner: scans sheets for the header row that
    carries an occupation-code column, a NAICS column, and an
    employment column; normalizes names; strips suppression flags."""
    xl = pd.ExcelFile(path)
    located = None
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, nrows=80)
        for i in range(len(raw)):
            cells = [frz._norm(v) for v in raw.iloc[i].tolist()]
            has_code = any(frz._is_code_name(c) for c in cells)
            has_naics = any("NAICS" in c and "TITLE" not in c for c in cells)
            has_emp = any("EMP" in c for c in cells)
            if has_code and has_naics and has_emp:
                located = (sheet, i)
                break
        if located:
            break
    if located is None:
        diag = []
        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None, nrows=4)
            head = ["; ".join(frz._norm(v) for v in raw.iloc[i].tolist()[:8])
                    for i in range(len(raw))]
            diag.append(f"  sheet '{sheet}':\n    " + "\n    ".join(head))
        sys.exit(f"{path.name}: no header row with an occupation-code "
                 f"column, a NAICS column, and an employment column. "
                 f"First rows per sheet:\n" + "\n".join(diag))

    sheet, hdr = located
    df = xl.parse(sheet, header=hdr)
    df.columns = [frz._norm(c) for c in df.columns]
    code_col = next(c for c in df.columns if frz._is_code_name(c))
    naics_col = next(c for c in df.columns
                     if "NAICS" in c and "TITLE" not in c)
    df = df.rename(columns={code_col: "OCC_CODE", naics_col: "naics"})
    if "TOT_EMP" not in df.columns:
        emp_col = next((c for c in df.columns
                        if "EMP" in c and "PRSE" not in c
                        and "PCT" not in c), None)
        if emp_col is None:
            sys.exit(f"{path.name}: no employment column found; columns: "
                     f"{sorted(df.columns)}")
        df = df.rename(columns={emp_col: "TOT_EMP"})

    if "O_GROUP" in df.columns:
        df = df[df["O_GROUP"].astype(str).str.lower() == "detailed"]
    elif "GROUP" in df.columns:
        g = df["GROUP"].astype(str).str.strip().str.lower()
        df = df[df["GROUP"].isna() | g.isin(["", "nan", "detailed"])]
    df = df.copy()
    df["OCC_CODE"] = df["OCC_CODE"].astype(str).str.strip()
    df = df[df["OCC_CODE"].str.match(frz.SOC_RE)]
    df = df[~df["OCC_CODE"].str.endswith("-0000")]
    df["naics"] = (df["naics"].astype(str).str.strip()
                   .str.extract(r"^(\d+)", expand=False))
    df = df.dropna(subset=["naics"])
    n_rows = len(df)
    df["emp"] = frz._to_num(df["TOT_EMP"])
    n_suppressed = int(df["emp"].isna().sum())
    df = df.dropna(subset=["emp"])
    df.attrs["n_rows"] = n_rows
    df.attrs["n_suppressed"] = n_suppressed
    return df[["naics", "OCC_CODE", "emp"]]


def mfg_share_soc2000(ind: pd.DataFrame) -> pd.DataFrame:
    """Per SOC-2000: manufacturing employment share and total emp."""
    is_mfg = ind["naics"].str[:2].isin(["31", "32", "33"])
    tot = ind.groupby("OCC_CODE")["emp"].sum()
    mfg = ind[is_mfg].groupby("OCC_CODE")["emp"].sum()
    out = pd.DataFrame({"emp_total": tot,
                        "emp_mfg": mfg.reindex(tot.index).fillna(0.0)})
    out = out[out["emp_total"] > 0]
    out["mfg_share"] = out["emp_mfg"] / out["emp_total"]
    return out.reset_index().rename(columns={"index": "OCC_CODE"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nat3d", default=str(NAT3D),
                    help="primary industry-occupation file (May 2003, 3d)")
    ap.add_argument("--nat4d", default=str(NAT4D),
                    help="robustness file (2002, 4d); skipped if absent")
    args = ap.parse_args()

    p3 = Path(args.nat3d)
    if not p3.exists():
        sys.exit(f"missing {p3}. Download the BLS OES industry-specific "
                 "national file (May 2003, 3-digit NAICS) into "
                 "data/oesm03in4/ and re-run.")

    ind3 = read_industry_occupation(p3)
    n_ind3 = ind3["naics"].nunique()
    s3 = mfg_share_soc2000(ind3)
    print(f"[{p3.name}] {ind3.attrs['n_rows']} detailed industry-occupation "
          f"rows, {ind3.attrs['n_suppressed']} suppressed dropped; "
          f"{n_ind3} industries; {len(s3)} SOC-2000 occupations with a "
          f"denominator; mfg share: median {s3['mfg_share'].median():.3f}, "
          f"share>10% for {int((s3['mfg_share'] > 0.10).sum())} occupations")
    if len(s3) < MIN_SOC2000:
        sys.exit(f"NO-GO: only {len(s3)} SOC-2000 codes covered "
                 f"(criterion {MIN_SOC2000}); the control is not built and "
                 "script 29 keeps the registered scope statement.")

    s2 = None
    p4 = Path(args.nat4d)
    if p4.exists():
        ind2 = read_industry_occupation(p4)
        s2 = mfg_share_soc2000(ind2)
        both = s3.merge(s2, on="OCC_CODE", suffixes=("_03", "_02"))
        from scipy.stats import spearmanr
        rho = spearmanr(both["mfg_share_03"], both["mfg_share_02"])[0]
        print(f"[{p4.name}] robustness vintage: {len(s2)} occupations; "
              f"rank corr with May 2003 shares {rho:+.3f} "
              f"(N={len(both)})")
    else:
        print(f"[robustness] {p4} not found; primary vintage only.")

    # compose the crosswalk exactly as script 26 does
    xw0010 = frz.read_xwalk(frz._find_xwalk("soc*2000*2010*.xls*", None,
                                            "SOC 2000->2010 crosswalk"),
                            "2000", "2010")
    xw1018 = frz.read_xwalk(frz._find_xwalk("soc*2010*2018*.xls*", None,
                                            "SOC 2010->2018 crosswalk"),
                            "2010", "2018")
    xw = (xw0010.merge(xw1018, on="soc2010")[["soc2000", "soc2018"]]
                .drop_duplicates())

    m = xw.merge(s3, left_on="soc2000", right_on="OCC_CODE", how="inner")
    if s2 is not None:
        m = m.merge(s2[["OCC_CODE", "mfg_share"]]
                    .rename(columns={"mfg_share": "mfg_share_2002"}),
                    on="OCC_CODE", how="left")
    rows = []
    for code, g in m.groupby("soc2018"):
        w = g["emp_total"].to_numpy(float)
        if w.sum() <= 0:
            continue
        row = {"soc2018": code,
               "mfg_share": float(np.average(g["mfg_share"], weights=w)),
               "emp_total": float(w.sum()),
               "n_sources": int(g["soc2000"].nunique())}
        if "mfg_share_2002" in g.columns and g["mfg_share_2002"].notna().any():
            gg = g.dropna(subset=["mfg_share_2002"])
            row["mfg_share_2002"] = float(
                np.average(gg["mfg_share_2002"],
                           weights=gg["emp_total"].to_numpy(float)))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("soc2018")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} SOC-2018 occupations "
          f"({out['n_sources'].sum()} SOC-2000 sources used); "
          f"mfg share range {out['mfg_share'].min():.3f}-"
          f"{out['mfg_share'].max():.3f}")

    # provenance
    manifest = (json.loads(frz.MANIFEST.read_text())
                if frz.MANIFEST.exists() else {})
    ext = manifest.get("external_inputs", {})
    files = {"nat3d_may2003": p3}
    if p4.exists():
        files["nat4d_2002"] = p4
    ext["occ_manufacturing_share"] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "source": ("BLS OES industry-specific occupational employment, "
                   "national, May 2003 3-digit NAICS (primary) and 2002 "
                   "4-digit NAICS (robustness), bls.gov/oes"),
        "files": {k: {"name": p.name, "sha256": frz._sha256(p)}
                  for k, p in files.items()},
        "derived": True,
        "recipe": ("Per SOC-2000: TOT_EMP summed over NAICS 31-33 over "
                   "TOT_EMP summed over all industries in the same file "
                   "(shared suppression regime); mapped to SOC-2018 by the "
                   "composed crosswalks, employment-weighted. Written by "
                   "scripts/31_freeze_manufacturing_share.py to "
                   "data/occ_manufacturing_share.csv."),
    }
    manifest["external_inputs"] = ext
    frz.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("MANIFEST updated (external_inputs/occ_manufacturing_share)")


if __name__ == "__main__":
    main()
