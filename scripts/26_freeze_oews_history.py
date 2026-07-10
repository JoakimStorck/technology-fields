"""
26_freeze_oews_history.py
-------------------------
Freezes the historical OEWS wage window for the robot-era analysis:
national occupational median wages for 1999, 2003, and 2007 (SOC-2000
throughout), carried onto the frozen O*NET-SOC 2019 geometry through the
official SOC 2000 -> 2010 -> 2018 crosswalks.

Motivation. The robot wave diffused 1993-2007 (Acemoglu & Restrepo 2020).
The longest classification-consistent OEWS wage window covering it is
1999 -> 2007: wages exist nationally from 1997, but 1997-1998 use the old
OES occupational codes, while 1999-2009 are SOC-2000. 2003 is frozen as
the midpoint for window splits. The frozen file is the data side of the
"each wave meets its own window" design: the robot field confronted with
its own diffusion window, with the LLM field as the placebo, mirroring
the paper's Section 8 in reverse.

Crosswalk. The geometry is keyed by O*NET-SOC 2019 (SOC-2018 detailed
codes after stripping the .xx suffix). Each SOC-2018 code is mapped to
its SOC-2000 sources by composing the two BLS crosswalks. Where a 2018
code has several 2000 sources, the frozen wage is the employment-weighted
mean of log wages across sources (weights: that year's TOT_EMP); where a
2000 code split into several 2018 codes, they share the source wage, the
same many-to-one convention as the paper's 2019-2025 join. A one_to_one
flag marks codes with a single 2000 source that maps only to them, for
the stable-code robustness subsample.

Wage measure. Per SOC-2000 row the hourly wage is taken in the
pre-registered preference order H_MEDIAN > A_MEDIAN/2080 > H_MEAN >
A_MEAN/2080 (annual-only occupations such as teachers fall back to the
annual median); the source used is recorded per row and aggregated to
the output.

AWAITING INPUT (all placed in data/, original names kept):
  OEWS national files, one per year, from bls.gov/oes/tables.htm:
      1999 National (annual), May 2003 National, May 2007 National.
      Extract the national .xls from each zip into data/; the script
      globs for it (patterns like *1999*nat*.xls*) or takes explicit
      paths via --wages-1999/--wages-2003/--wages-2007.
  SOC crosswalks, from bls.gov/soc:
      soc_2000_to_2010_crosswalk.xls
      soc_2010_to_2018_crosswalk.xlsx
      (explicit paths via --xwalk-2000-2010 / --xwalk-2010-2018).
  Readers: pip install xlrd openpyxl (.xls and .xlsx engines).

Reads:
    data/<oews national xls per year>              (external, see above)
    data/soc_2000_to_2010_crosswalk.xls            (external)
    data/soc_2010_to_2018_crosswalk.xlsx           (external)
    data/occupation_embeddings_polar_scaled.csv    coverage report only
Writes:
    data/oews_history_wages.csv    soc2018, year, wage_hourly, wage_source,
                                   tot_emp, n_soc2000, one_to_one
    data/MANIFEST.json             external_inputs entry with file hashes

Not in run_all yet: run standalone (like 00 and the data-prep helpers)
until the inputs are frozen and the downstream baselines exist.

Usage:
    python scripts/26_freeze_oews_history.py
    python scripts/26_freeze_oews_history.py --wages-1999 data/national_1999_dl.xls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
MANIFEST = DATA / "MANIFEST.json"
OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
OUT_FILE = DATA / "oews_history_wages.csv"

YEARS = (1999, 2003, 2007)
WAGE_PREFERENCE = ["H_MEDIAN", "A_MEDIAN", "H_MEAN", "A_MEAN"]
HOURS_PER_YEAR = 2080.0
SOC_RE = re.compile(r"^\d{2}-\d{4}$")


# ─────────────────────────────────────────────────────────────────────
# Input location
# ─────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_wage_file(year: int, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--wages-{year}: {p} not found")
        return p
    yy = str(year)[2:]
    pats = [f"*{year}*nat*", f"*nat*{year}*", f"*national*{year}*",
            f"*{year}*national*", f"*M{year}*", f"*m{year}*",
            f"*oes*{yy}*nat*", f"*oesm{yy}*"]
    hits = sorted({p for pat in pats for ext in (".xls", ".xlsx")
                   for p in DATA.glob(pat + ext)})
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"No OEWS national file for {year} found in data/ "
                 f"(tried patterns like *{year}*nat*.xls*). Download the "
                 f"{year} National file from bls.gov/oes/tables.htm, extract "
                 f"the .xls into data/, or pass --wages-{year} PATH.")
    sys.exit(f"Ambiguous OEWS files for {year}: "
             f"{', '.join(p.name for p in hits)}. Pass --wages-{year} PATH.")


def _find_xwalk(default_glob: str, explicit: str | None, label: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"{label}: {p} not found")
        return p
    hits = sorted(DATA.glob(default_glob))
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"Expected exactly one {label} in data/ matching "
             f"'{default_glob}' (found {len(hits)}). Download it from "
             f"bls.gov/soc or pass an explicit path.")


# ─────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────

CODE_NAMES = {"OCC_CODE", "SOC_CODE", "OCC", "SOC", "OCCCODE", "SOCCODE"}


def _norm(c) -> str:
    return re.sub(r"\s+", "_", str(c).strip().upper())


def _is_code_name(c: str) -> bool:
    return c in CODE_NAMES or (len(c) <= 24
                               and ("OCC" in c or "SOC" in c)
                               and "CODE" in c)


def _to_num(series: pd.Series) -> pd.Series:
    """Locale-safe numeric conversion. Numeric cells pass through; text
    cells are cleaned of $, spaces and NBSP; a trailing decimal comma
    ('31,13') becomes a dot, other commas are thousands separators;
    suppression/topcode flags (*, **, #) become NaN."""
    def conv(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        t = str(v).strip().replace("\xa0", "").replace(" ", "")
        t = t.replace("$", "")
        if re.fullmatch(r"\d+,\d{1,2}", t):
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
        try:
            return float(t)
        except ValueError:
            return np.nan
    return series.map(conv)


def read_oews(path: Path) -> pd.DataFrame:
    """Detailed SOC-2000 rows with an hourly wage under the preference
    order. Vintage-robust: scans every sheet for the header row (older
    files carry title rows above it and vary the code-column name),
    normalizes column names, and strips $/, from numeric fields."""
    xl = pd.ExcelFile(path)
    located = None
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, nrows=80)
        for i in range(len(raw)):
            cells = [_norm(v) for v in raw.iloc[i].tolist()]
            has_code = any(_is_code_name(c) for c in cells)
            has_wage = any(("MEDIAN" in c or "MEAN" in c or "EMP" in c)
                           for c in cells)
            if has_code and has_wage:
                located = (sheet, i)
                break
        if located:
            break
    if located is None:
        diag = []
        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None, nrows=4)
            head = ["; ".join(_norm(v) for v in raw.iloc[i].tolist()[:8])
                    for i in range(len(raw))]
            diag.append(f"  sheet '{sheet}':\n    " + "\n    ".join(head))
        sys.exit(f"{path.name}: found no header row containing both an "
                 f"occupation-code column (like OCC_CODE / SOC CODE) and a "
                 f"wage/employment column. First rows per sheet:\n"
                 + "\n".join(diag))

    sheet, hdr = located
    df = xl.parse(sheet, header=hdr)
    df.columns = [_norm(c) for c in df.columns]
    code_col = next(c for c in df.columns if _is_code_name(c))
    df = df.rename(columns={code_col: "OCC_CODE"})

    if "O_GROUP" in df.columns:
        df = df[df["O_GROUP"].astype(str).str.lower() == "detailed"]
    elif "GROUP" in df.columns:
        g = df["GROUP"].astype(str).str.strip().str.lower()
        df = df[df["GROUP"].isna() | g.isin(["", "nan", "detailed"])]
    df = df.copy()
    df["OCC_CODE"] = df["OCC_CODE"].astype(str).str.strip()
    df = df[df["OCC_CODE"].str.match(SOC_RE)]
    df = df[~df["OCC_CODE"].str.endswith("-0000")]

    present = [c for c in WAGE_PREFERENCE if c in df.columns]
    if not present:
        sys.exit(f"{path.name}: none of {WAGE_PREFERENCE} present; "
                 f"columns: {sorted(df.columns)}")
    n_topcoded = 0
    if "H_MEDIAN" in df.columns:
        n_topcoded = int((df["H_MEDIAN"].astype(str).str.strip() == "#").sum())
    for c in present + (["TOT_EMP"] if "TOT_EMP" in df.columns else []):
        df[c] = _to_num(df[c])  # '*'/'**'/'#' flags -> NaN

    wage = np.full(len(df), np.nan)
    kind = np.array([""] * len(df), dtype=object)
    for c in present:
        v = df[c].to_numpy(float)
        if c.startswith("A_"):
            v = v / HOURS_PER_YEAR
        take = np.isnan(wage) & np.isfinite(v) & (v > 0)
        wage[take] = v[take]
        kind[take] = c
    df["wage_hourly"] = wage
    df["wage_source"] = kind
    df = df.dropna(subset=["wage_hourly"])
    if "TOT_EMP" not in df.columns:
        df["TOT_EMP"] = np.nan
    out = (df.drop_duplicates("OCC_CODE")
             [["OCC_CODE", "wage_hourly", "wage_source", "TOT_EMP"]]
             .reset_index(drop=True))
    out.attrs["n_topcoded_median"] = n_topcoded
    return out


def read_xwalk(path: Path, left: str, right: str) -> pd.DataFrame:
    """Two-column code crosswalk [soc<left>, soc<right>], located by
    scanning for the header row that names both SOC vintages."""
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        hdr = None
        for i in range(min(len(raw), 20)):
            row = " | ".join(str(v) for v in raw.iloc[i].tolist()).upper()
            if f"{left} SOC" in row and f"{right} SOC" in row:
                hdr = i
                break
        if hdr is None:
            continue
        df = xl.parse(sheet, header=hdr)
        cols = {str(c).strip().upper(): c for c in df.columns}
        lcol = next((cols[k] for k in cols if left in k and "CODE" in k), None)
        rcol = next((cols[k] for k in cols if right in k and "CODE" in k), None)
        if lcol is None or rcol is None:
            continue
        out = df[[lcol, rcol]].copy()
        out.columns = [f"soc{left}", f"soc{right}"]
        for c in out.columns:
            out[c] = out[c].astype(str).str.strip()
        out = out[out[f"soc{left}"].str.match(SOC_RE)
                  & out[f"soc{right}"].str.match(SOC_RE)]
        return out.drop_duplicates().reset_index(drop=True)
    sys.exit(f"{path.name}: could not locate a header row naming both "
             f"'{left} SOC' and '{right} SOC'.")


# ─────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────

def record_provenance(files: dict[str, Path]) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    ext = manifest.get("external_inputs", {})
    ext["oews_history_1999_2007"] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ("BLS Occupational Employment and Wage Statistics, national "
                   "files 1999/2003/2007 (SOC-2000), bls.gov/oes/tables.htm; "
                   "SOC 2000->2010 and 2010->2018 crosswalks, bls.gov/soc"),
        "files": {label: {"name": p.name, "sha256": _sha256(p)}
                  for label, p in files.items()},
        "derived": True,
        "recipe": ("Detailed SOC-2000 rows; hourly wage by preference "
                   "H_MEDIAN > A_MEDIAN/2080 > H_MEAN > A_MEAN/2080; mapped "
                   "to SOC-2018 by composing the two crosswalks; per SOC-2018 "
                   "code the employment-weighted mean of log wages across "
                   "SOC-2000 sources (weights: same-year TOT_EMP). Written by "
                   "scripts/26_freeze_oews_history.py to "
                   "data/oews_history_wages.csv."),
    }
    manifest["external_inputs"] = ext
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for y in YEARS:
        ap.add_argument(f"--wages-{y}", dest=f"wages_{y}")
    ap.add_argument("--xwalk-2000-2010", dest="xw0010")
    ap.add_argument("--xwalk-2010-2018", dest="xw1018")
    args = ap.parse_args()

    wage_paths = {y: _find_wage_file(y, getattr(args, f"wages_{y}"))
                  for y in YEARS}
    xw0010_path = _find_xwalk("soc*2000*2010*.xls*", args.xw0010,
                              "SOC 2000->2010 crosswalk")
    xw1018_path = _find_xwalk("soc*2010*2018*.xls*", args.xw1018,
                              "SOC 2010->2018 crosswalk")

    xw0010 = read_xwalk(xw0010_path, "2000", "2010")
    xw1018 = read_xwalk(xw1018_path, "2010", "2018")
    xw = (xw0010.merge(xw1018, on="soc2010")[["soc2000", "soc2018"]]
                .drop_duplicates())
    print(f"Crosswalk 2000->2018 composed: {len(xw)} code pairs, "
          f"{xw['soc2000'].nunique()} SOC-2000 sources, "
          f"{xw['soc2018'].nunique()} SOC-2018 targets.")

    # one_to_one at the 2018 level: a single 2000 source that maps only here
    n_sources = xw.groupby("soc2018")["soc2000"].nunique()
    n_targets = xw.groupby("soc2000")["soc2018"].nunique()
    single_src = n_sources[n_sources == 1].index
    src_of = xw[xw["soc2018"].isin(single_src)].set_index("soc2018")["soc2000"]
    one_to_one = {c: (n_targets.get(src_of[c], 0) == 1) for c in single_src}

    rows = []
    for y in YEARS:
        w = read_oews(wage_paths[y])
        print(f"[{y}] {wage_paths[y].name}: {len(w)} detailed SOC-2000 rows "
              f"with a wage; sources "
              f"{w['wage_source'].value_counts().to_dict()}; "
              f"topcoded medians (#, fall back per preference): "
              f"{w.attrs.get('n_topcoded_median', 0)}")
        m = xw.merge(w, left_on="soc2000", right_on="OCC_CODE", how="inner")
        m["lnw"] = np.log(m["wage_hourly"])
        m["wgt"] = m["TOT_EMP"].fillna(0.0).clip(lower=0.0)

        for code, g in m.groupby("soc2018"):
            wgt = g["wgt"].to_numpy(float)
            if wgt.sum() <= 0:
                wgt = np.ones(len(g))
            src = g["wage_source"].unique()
            rows.append({
                "soc2018": code,
                "year": y,
                "wage_hourly": float(np.exp(np.average(g["lnw"], weights=wgt))),
                "wage_source": src[0] if len(src) == 1 else "mixed",
                "tot_emp": float(g["TOT_EMP"].sum(skipna=True)),
                "n_soc2000": int(g["soc2000"].nunique()),
                "one_to_one": bool(one_to_one.get(code, False)),
            })

    out = pd.DataFrame(rows).sort_values(["year", "soc2018"])
    out.to_csv(OUT_FILE, index=False)
    print(f"wrote {OUT_FILE}  ({len(out)} rows)")

    # coverage against the frozen geometry
    occ = pd.read_csv(OCC_FILE, usecols=["onet_code"])
    soc6 = (occ["onet_code"].astype(str)
            .str.replace(r"\..*", "", regex=True).str.strip().unique())
    print(f"Coverage vs the coordinate universe ({len(soc6)} SOC-2018 codes "
          f"behind 878 occupations):")
    got = {}
    for y in YEARS:
        codes = set(out.loc[out["year"] == y, "soc2018"])
        got[y] = codes
        n = sum(c in codes for c in soc6)
        n11 = sum(c in codes and
                  bool(out.loc[(out["year"] == y) & (out["soc2018"] == c),
                               "one_to_one"].iloc[0])
                  for c in soc6)
        print(f"  [{y}] matched {n} of {len(soc6)}  (one_to_one {n11})")
    common = set.intersection(*got.values())
    n_all = sum(c in common for c in soc6)
    print(f"  [all three years] matched {n_all} of {len(soc6)}")

    record_provenance({"wages_1999": wage_paths[1999],
                       "wages_2003": wage_paths[2003],
                       "wages_2007": wage_paths[2007],
                       "xwalk_2000_2010": xw0010_path,
                       "xwalk_2010_2018": xw1018_path})
    print("MANIFEST external_inputs entry recorded.")


if __name__ == "__main__":
    main()
