"""
16_placebo_field.py
-------------------
Placebo-field test for the wage cross-section (paper section 'What the Wage
Cross-Section Can and Cannot Show'). Written and pre-registered BEFORE the
first run; adverse outcomes are reported honestly in this docstring, the
summary, and the manuscript.

Motivation. Section 8.2 argues that the 2019-2025 occupational cross-section
manufactures apparent support for ANY mechanism whose cross-sectional footprint
is a wage gradient, because (i) the price gate targets dear work, so predicted
directional pressure loads on the baseline wage for any field overlapping
priced regions, and (ii) the window's dominant wage shock (the pandemic
compression) is itself a wage-level gradient. As committed, that claim is an
inference from one mechanism (the calibrated cognitive field). This script
demonstrates it with placebo technologies that carry no claim whatsoever about
the period's technology:

  P1  the manual field of Table 3 (hand-placed industrial automation, west,
      narrow, deployed decades before the window), constants identical to
      scripts/10_demand_channel.py;
  P2  a rotated clone of the calibrated cognitive field (same chi_K, z_K, A_K;
      xi_K + 180 degrees), a technology that does not exist.

PRE-REGISTERED HYPOTHESES (before first run):
  H1  Each placebo's predicted directional pressure proj_o = Delta mu_o .
      grad ln Pi passes the naive check against observed 2019-2025 wage
      changes: |Spearman| >= 0.17 (half the cognitive field's +0.34) with
      p < 0.01, despite neither placebo measuring anything about 2019-2025.
  H1b The sign of the raw correlation is fully determined by the wage-level
      channel: sign(rho(proj, dlnw)) = -sign(rho(proj, w0)), since
      rho(w0, dlnw) < 0 in the compression window.
  H2  Conditioning kills it: the partial rank correlation of placebo pressure
      with wage growth given the baseline-wage rank is |rho| < 0.10.
  H3  The window split confines it: the raw correlation in the 2023-2025
      deployment window is |rho| < 0.10, and the 2019-2023 window carries at
      least 0.8 of the full-window magnitude.
If any hypothesis fails, the placebo carries position-specific information and
the universal 'any mechanism' claim of Section 8.2 must be weakened to match;
the failure is reported, not suppressed.

RESULTS (first run, recorded after pre-registration):
  P1  H1 PASS (raw -0.242, p=4e-11), H1b PASS (sign flipped exactly as the
      wage-level channel dictates: collinearity with w0 is +0.44), H2 PASS
      (partial +0.013), H3 PASS (early -0.262, late -0.016).
  P2  H1 FAIL: |raw| = 0.131 < 0.17. The rotated clone sits mostly in cheap
      territory, the gate barely opens, and its pressure loads on the wage
      level only at +0.22, so the manufactured correlation is weaker -- but
      it is still significant (p = 4e-4) with the predicted sign, killed by
      conditioning (H2 PASS, -0.009) and confined to the compression window
      (H3 PASS). CONCESSION carried to the manuscript: the magnitude of the
      manufactured support scales with how strongly the mechanism's footprint
      loads on the wage level; 'comparable magnitude for any field' is too
      strong, 'significant apparent support for any field, strongest where
      the footprint is steepest' is what the data show.

Frozen baseline assertions (guard against machinery drift): before computing
any placebo, the script reproduces the cognitive field's committed row --
raw Spearman +0.340, partial|w0 +0.021, N = 725 -- to 3 decimals against
results/centroid_shift_test_summary.txt values.

Machinery: imported from scripts/11_centroid_shift_test.py (post_centroids,
oews_median, partial ranks), which itself asserts machine-precision agreement
with model.regime at run time; this script therefore cannot drift from the
committed operator.

Outputs:
    results/placebo_field_windows.csv
    results/placebo_field_summary.txt

Usage:
    python scripts/16_placebo_field.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.technology import Technology

def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_setup = _load("_setup.py")
cst = _load("11_centroid_shift_test.py")

RESULTS = REPO_ROOT / "results"
ORIGIN_YEAR, MID_YEAR, ENDPOINT_YEAR = 2019, 2023, 2025

# frozen baselines (cognitive field, results/centroid_shift_test_summary.txt)
FROZEN_RAW, FROZEN_PARTIAL, FROZEN_N = +0.340, +0.021, 725

# committed manual field (identical to scripts/10_demand_channel.py MANUAL)
MANUAL = dict(xi_K=np.radians(180), chi_K=0.45, z_K=0.30, A_K=1.2, s_K=1.0)


def pressure_frame(inp, occ, tech, L0, ell):
    """proj_o for a technology, merged with OEWS origin/mid/end wages."""
    mu_pre, mu_post, _, _ = cst.post_centroids(inp, tech, L0, ell)
    dmu = mu_post - mu_pre
    xi_o, chi_o = occ["xi"].to_numpy(), occ["chi"].to_numpy()
    g_r, g_ang = inp.field.grad_log_pi(xi_o, chi_o)
    gx = g_r * np.cos(xi_o) - g_ang * np.sin(xi_o)
    gy = g_r * np.sin(xi_o) + g_ang * np.cos(xi_o)
    df = pd.DataFrame({"OCC_CODE": occ["OCC_CODE"].to_numpy(),
                       "proj": dmu[:, 0] * gx + dmu[:, 1] * gy})
    for lab, yr in (("w_origin", ORIGIN_YEAR), ("w_mid", MID_YEAR),
                    ("w_end", ENDPOINT_YEAR)):
        df[lab] = df["OCC_CODE"].map(cst.oews_median(yr))
    df = df.dropna(subset=["w_origin", "w_end"]).copy()
    df["dlnw"] = np.log(df["w_end"]) - np.log(df["w_origin"])
    return df


def report(lines, rows, name, df):
    raw, rawp = spearmanr(df["proj"], df["dlnw"])
    coll = spearmanr(df["proj"], df["w_origin"])[0]
    par, parp = cst._partial_rank(df["proj"], df["dlnw"], df["w_origin"])
    lines += [f"", f"[{name}]  N={len(df)}",
              f"  raw Spearman(proj, dlnw 2019-2025) = {raw:+.3f} (p={rawp:.1e})",
              f"  collinearity Spearman(proj, w0)    = {coll:+.3f}",
              f"  partial | rank(w0)                 = {par:+.3f} (p={parp:.2f})"]
    win = df.dropna(subset=["w_mid"]).copy()
    win["dlnw_early"] = np.log(win["w_mid"]) - np.log(win["w_origin"])
    win["dlnw_late"] = np.log(win["w_end"]) - np.log(win["w_mid"])
    for lab, dly, wz in ((f"{ORIGIN_YEAR}-{MID_YEAR}", win["dlnw_early"], win["w_origin"]),
                         (f"{MID_YEAR}-{ENDPOINT_YEAR}", win["dlnw_late"], win["w_mid"]),
                         (f"{ORIGIN_YEAR}-{ENDPOINT_YEAR}", win["dlnw"], win["w_origin"])):
        r_, rp_ = spearmanr(win["proj"], dly)
        p_, pp_ = cst._partial_rank(win["proj"], dly, wz)
        rows.append({"field": name, "window": lab, "N": len(win),
                     "spearman_raw": r_, "p_raw": rp_,
                     "partial_w0": p_, "p_partial": pp_})
        lines.append(f"    [{lab}]  raw {r_:+.3f} (p={rp_:.1e})  "
                     f"partial|w0 {p_:+.3f} (p={pp_:.2f})")
    return raw, coll, par


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    ell = _setup.interpretable_ell(inp)
    lines = ["Placebo-field test (pre-registered; see module docstring).",
             f"  operator at baseline L0; gamma {cst.GAMMA}, ell {ell:.4f}"]
    rows: list[dict] = []

    # -- frozen baseline: reproduce the cognitive row before any placebo -----
    cog = pressure_frame(inp, occ, _setup.load_tech(), L0, ell)
    raw_c, coll_c, par_c = report(lines, rows, "cognitive (calibrated, baseline)", cog)
    assert abs(raw_c - FROZEN_RAW) < 1e-3, f"baseline raw drifted: {raw_c:+.4f}"
    assert abs(par_c - FROZEN_PARTIAL) < 1e-3, f"baseline partial drifted: {par_c:+.4f}"
    assert len(cog) == FROZEN_N, f"baseline N drifted: {len(cog)}"
    lines.append("  frozen baseline reproduced (raw, partial, N).")

    # -- P1: the manual field of Table 3 -------------------------------------
    p1 = pressure_frame(inp, occ, Technology(**MANUAL), L0, ell)
    raw1, coll1, par1 = report(lines, rows, "P1 manual (Table 3, hand-placed)", p1)

    # -- P2: rotated cognitive clone ------------------------------------------
    t = _setup.load_tech()
    p2 = pressure_frame(inp, occ, Technology(
        xi_K=t.xi_K + np.pi, chi_K=t.chi_K, z_K=t.z_K, A_K=t.A_K, s_K=1.0),
        L0, ell)
    raw2, coll2, par2 = report(lines, rows, "P2 rotated cognitive clone (+180 deg)", p2)

    # -- hypothesis verdicts ---------------------------------------------------
    lines += ["", "Pre-registered hypothesis verdicts:"]
    wr = pd.DataFrame(rows)
    for name, raw_, coll_, par_ in (("P1", raw1, coll1, par1),
                                    ("P2", raw2, coll2, par2)):
        sub = wr[wr["field"].str.startswith(name)]
        early = float(sub.loc[sub["window"] == "2019-2023", "spearman_raw"].iloc[0])
        late = float(sub.loc[sub["window"] == "2023-2025", "spearman_raw"].iloc[0])
        full = float(sub.loc[sub["window"] == "2019-2025", "spearman_raw"].iloc[0])
        h1 = abs(raw_) >= 0.17
        h1b = np.sign(raw_) == -np.sign(coll_)
        h2 = abs(par_) < 0.10
        h3 = abs(late) < 0.10 and abs(early) >= 0.8 * abs(full)
        lines.append(f"  [{name}] H1 {'PASS' if h1 else 'FAIL'} (|raw|={abs(raw_):.3f})  "
                     f"H1b {'PASS' if h1b else 'FAIL'}  "
                     f"H2 {'PASS' if h2 else 'FAIL'} (|partial|={abs(par_):.3f})  "
                     f"H3 {'PASS' if h3 else 'FAIL'} (early {early:+.3f}, late {late:+.3f})")

    wr.to_csv(RESULTS / "placebo_field_windows.csv", index=False)
    (RESULTS / "placebo_field_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS/'placebo_field_windows.csv'} and _summary.txt")


if __name__ == "__main__":
    main()
