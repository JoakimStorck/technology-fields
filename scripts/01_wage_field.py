"""
01_wage_field.py
----------------
Litmus test for the price-of-skill field of Paper 3 (eq. 1):

    ln Pi(r) = m0 + m1 cos xi + m2 sin xi + chi (m3 + m4 cos xi + m5 sin xi)

Estimated on the occupation cross-section of Paper 1 (geometry-of-work):
occupation polar coordinates from the reference encoder run, merged with
BLS OEWS May 2023 median hourly wages. Sample construction replicates
the Mincer regression of Paper 1 (Table 3): detailed SOC, positive
H_MEDIAN, non-missing rle_mean, N = 785.

Reads exclusively from data/ in this repository (frozen by
scripts/00_freeze_inputs.py; provenance in data/MANIFEST.json).

Specifications:
  S0  replication:      ln w ~ cos xi + sin xi + chi          (Paper 1, Table 3 col 1)
  S1  field (m0-m5):    ln w ~ cos xi + sin xi + chi
                              + chi*cos xi + chi*sin xi
  S2  2nd harmonic:     S1 + chi*cos 2xi + chi*sin 2xi        (Wald test: harmonic
                                                               sufficiency of S1)
  S3  weighted:         S1 estimated by WLS with TOT_EMP weights

HC3 robust standard errors throughout. Derived quantities for S1:
level-gradient direction atan2(m2, m1); depth-return direction
atan2(m5, m4) with amplitude hypot(m4, m5); the angular interval where
beta_chi(xi) = m3 + m4 cos xi + m5 sin xi > 0.

Usage:
    python scripts/01_wage_field.py

Outputs: results/wage_field_coefficients.csv, results/wage_field_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
WAGE_FILE = DATA / "national_M2023_dl.xlsx"
RLE_FILE = DATA / "occupation_rle.csv"


def load_sample() -> pd.DataFrame:
    """Replicate the Paper 1 Mincer sample: occupation coordinates merged
    with BLS H_MEDIAN, restricted to positive wages and non-missing
    rle_mean (education), which pins N = 785."""
    occ = pd.read_csv(OCC_FILE)

    wages = pd.read_excel(WAGE_FILE, usecols=[8, 9, 11, 22])
    wages.columns = ["OCC_CODE", "OCC_TITLE", "TOT_EMP", "H_MEDIAN"]
    wages["OCC_CODE"] = (
        wages["OCC_CODE"].astype(str).str.replace(r"\..*", "", regex=True).str.strip()
    )
    wages["H_MEDIAN"] = pd.to_numeric(wages["H_MEDIAN"], errors="coerce")
    wages["TOT_EMP"] = pd.to_numeric(wages["TOT_EMP"], errors="coerce")

    occ = occ.copy()
    occ["OCC_CODE"] = (
        occ["onet_code"].astype(str).str.replace(r"\..*", "", regex=True).str.strip()
    )
    df = occ.merge(
        wages[["OCC_CODE", "TOT_EMP", "H_MEDIAN"]], on="OCC_CODE", how="left"
    )
    df = df.merge(pd.read_csv(RLE_FILE), on="onet_code", how="left")
    df = df.dropna(subset=["H_MEDIAN", "xi", "chi", "rle_mean"])
    df = df.loc[df["H_MEDIAN"] > 0].copy()

    df["ln_wage"] = np.log(df["H_MEDIAN"])
    df["cos_xi"] = np.cos(df["xi"])
    df["sin_xi"] = np.sin(df["xi"])
    df["chi_cos"] = df["chi"] * df["cos_xi"]
    df["chi_sin"] = df["chi"] * df["sin_xi"]
    df["chi_cos2"] = df["chi"] * np.cos(2 * df["xi"])
    df["chi_sin2"] = df["chi"] * np.sin(2 * df["xi"])
    return df.reset_index(drop=True)


def fit(df: pd.DataFrame, cols: list[str], weights: np.ndarray | None = None):
    X = sm.add_constant(df[cols].astype(float))
    y = df["ln_wage"].astype(float)
    if weights is None:
        return sm.OLS(y, X).fit(cov_type="HC3")
    return sm.WLS(y, X, weights=weights).fit(cov_type="HC3")


def beta_chi_interval(m3: float, m4: float, m5: float) -> tuple[float, float] | None:
    """Angular interval (degrees) where beta_chi(xi) > 0, if any.
    beta_chi(xi) = m3 + A cos(xi - phi), A = hypot(m4, m5),
    phi = atan2(m5, m4)."""
    A = float(np.hypot(m4, m5))
    phi = float(np.arctan2(m5, m4))
    if A == 0 or abs(m3) >= A:
        return None if m3 <= 0 else (0.0, 360.0)
    half = float(np.arccos(-m3 / A))
    lo = np.degrees(phi - half) % 360
    hi = np.degrees(phi + half) % 360
    return lo, hi


def main() -> None:
    df = load_sample()
    lines: list[str] = [f"Sample: N = {len(df)} occupations\n"]

    s0 = fit(df, ["cos_xi", "sin_xi", "chi"])
    s1_cols = ["cos_xi", "sin_xi", "chi", "chi_cos", "chi_sin"]
    s1 = fit(df, s1_cols)
    s2 = fit(df, s1_cols + ["chi_cos2", "chi_sin2"])
    w = df["TOT_EMP"].fillna(0).clip(lower=0)
    s3 = fit(df.loc[w > 0], s1_cols, weights=w.loc[w > 0])

    name_map = {
        "const": "m0",
        "cos_xi": "m1",
        "sin_xi": "m2",
        "chi": "m3",
        "chi_cos": "m4",
        "chi_sin": "m5",
        "chi_cos2": "m6",
        "chi_sin2": "m7",
    }
    rows = []
    for label, model in [("S0_replication", s0), ("S1_field", s1),
                         ("S2_second_harmonic", s2), ("S3_weighted", s3)]:
        lines.append(f"== {label}:  N = {int(model.nobs)}   "
                     f"R2 = {model.rsquared:.4f}  adjR2 = {model.rsquared_adj:.4f}")
        for var in model.params.index:
            rows.append(dict(
                spec=label, param=name_map.get(var, var), variable=var,
                coef=model.params[var], se=model.bse[var],
                t=model.tvalues[var], p=model.pvalues[var],
            ))
            lines.append(f"   {name_map.get(var, var):3s} ({var:9s})  "
                         f"{model.params[var]:+8.4f}  (se {model.bse[var]:.4f},"
                         f" p {model.pvalues[var]:.4f})")
        lines.append("")

    # Harmonic sufficiency: joint Wald test of m6 = m7 = 0 in S2
    wald = s2.wald_test("(chi_cos2 = 0), (chi_sin2 = 0)", scalar=True)
    lines.append(f"Second-harmonic Wald test (m6 = m7 = 0): "
                 f"stat = {float(wald.statistic):.3f}, p = {float(wald.pvalue):.4f}")

    # Derived quantities from S1
    p = s1.params
    lvl_dir = np.degrees(np.arctan2(p["sin_xi"], p["cos_xi"])) % 360
    lvl_amp = float(np.hypot(p["cos_xi"], p["sin_xi"]))
    dep_dir = np.degrees(np.arctan2(p["chi_sin"], p["chi_cos"])) % 360
    dep_amp = float(np.hypot(p["chi_cos"], p["chi_sin"]))
    interval = beta_chi_interval(p["chi"], p["chi_cos"], p["chi_sin"])
    lines += [
        "",
        f"Level gradient (m1, m2):  direction {lvl_dir:.1f} deg, amplitude {lvl_amp:.3f}",
        f"Depth return  (m4, m5):   direction {dep_dir:.1f} deg, amplitude {dep_amp:.3f}",
        f"Mean depth return m3:     {p['chi']:+.3f}",
        (f"beta_chi(xi) > 0 for xi in ({interval[0]:.1f}, {interval[1]:.1f}) deg"
         if interval else "beta_chi(xi) <= 0 everywhere"),
    ]

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "wage_field_coefficients.csv", index=False)
    (RESULTS / "wage_field_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
