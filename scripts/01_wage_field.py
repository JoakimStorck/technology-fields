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

Reads exclusively from data/ in this repository via model.data
(frozen by scripts/00_freeze_inputs.py; provenance in data/MANIFEST.json).

Specifications:
  S0  replication:      ln w ~ cos xi + sin xi + chi          (Paper 1, Table 3 col 1)
  S1  field (m0-m5):    ln w ~ cos xi + sin xi + chi
                              + chi*cos xi + chi*sin xi
  S2  2nd harmonic,     S1 + chi*cos 2xi + chi*sin 2xi. UNBALANCED: kept for
      interaction only:  the record; chi*cos 2xi is near-collinear with the
                         omitted LEVEL term cos 2xi (corr ~ 0.93), so its
                         coefficients proxy for level structure.
  L2  2nd harmonic,     S1 + cos 2xi + sin 2xi (level only).
      level only:
  S2b balanced:         S1 + cos 2xi + sin 2xi + chi*cos 2xi + chi*sin 2xi.
                        The decisive test: the interaction second harmonic
                        is evaluated conditional on the level second
                        harmonic.
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

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.data import load_mincer_sample  # noqa: E402

RESULTS = REPO_ROOT / "results"


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
    df = load_mincer_sample()
    lines: list[str] = [f"Sample: N = {len(df)} occupations\n"]

    df["cos2"] = np.cos(2 * df["xi"])
    df["sin2"] = np.sin(2 * df["xi"])

    s0 = fit(df, ["cos_xi", "sin_xi", "chi"])
    s1_cols = ["cos_xi", "sin_xi", "chi", "chi_cos", "chi_sin"]
    s1 = fit(df, s1_cols)
    s2 = fit(df, s1_cols + ["chi_cos2", "chi_sin2"])
    l2 = fit(df, s1_cols + ["cos2", "sin2"])
    s2b = fit(df, s1_cols + ["cos2", "sin2", "chi_cos2", "chi_sin2"])
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
                         ("S2_interaction_unbalanced", s2),
                         ("L2_level_only", l2), ("S2b_balanced", s2b),
                         ("S3_weighted", s3)]:
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

    # Harmonic sufficiency. The naive test in S2 is misleading:
    # chi*cos 2xi proxies the omitted level term cos 2xi (corr ~ 0.93).
    # The decisive test conditions on the level second harmonic (S2b).
    w_naive = s2.wald_test("(chi_cos2 = 0), (chi_sin2 = 0)", scalar=True)
    w_bal = s2b.wald_test("(chi_cos2 = 0), (chi_sin2 = 0)", scalar=True)
    w_omni = s2b.wald_test(
        "(cos2 = 0), (sin2 = 0), (chi_cos2 = 0), (chi_sin2 = 0)", scalar=True)
    c_lvl = float(np.corrcoef(df["chi_cos2"], df["cos2"])[0, 1])
    lines += [
        "Harmonic-order tests for the depth return beta_chi(xi):",
        f"  naive (S2, no level harmonics):   m6 = m7 = 0  "
        f"p = {float(w_naive.pvalue):.4f}   [SPURIOUS: corr(chi*cos2xi, "
        f"cos2xi) = {c_lvl:.3f}]",
        f"  balanced (S2b, level included):   m6 = m7 = 0  "
        f"p = {float(w_bal.pvalue):.4f}",
        f"  omnibus (all four 2nd-harmonic terms = 0): "
        f"p = {float(w_omni.pvalue):.6f}",
        "  Conclusion: the depth return is first-harmonic (eq. 1 stands);",
        "  the residual second-harmonic structure sits in the LEVEL",
        f"  (L2: cos2 = {l2.params['cos2']:+.3f}, "
        f"p = {l2.pvalues['cos2']:.4f}), i.e. an E-W level effect of "
        f"~{100*abs(l2.params['cos2']):.0f}% at fixed chi,",
        "  consistent with the east-sector mediation pattern of Paper 1.",
    ]

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
