"""
11_centroid_shift_test.py
-------------------------
Empirical consistency test for the displacement channel (paper section
'Empirical Analysis'). EXPLICITLY NON-CAUSAL: a directional consistency check
of whether the takeover/centroid-shift framing points the right way, not an
identified effect.

Design:
  - rewrite each occupation's bundle with the calibrated AI field (displacement
    + survival-gated reinstatement, model.regime), evaluated at baseline
    employment L0, and take the shift in its (mass-normalised) bundle centroid
    Delta mu_o = mu_o^post - mu_o^pre;
  - project Delta mu_o onto grad ln Pi at the occupation's location: the
    displacement channel predicts bundles sliding DOWN-gradient (toward cheaper
    work) carry weaker wage growth, so proj_o = Delta mu_o . grad ln Pi is the
    model's predicted directional wage pressure;
  - compare proj_o to observed OEWS log median hourly wage changes 2019 -> 2025
    on the same SOC occupations: cross-sectional rank correlation (the common
    nominal drift is differenced out by the ranking). 2019 -> 2024 as a
    robustness endpoint.

The iota_o / centroid computation mirrors model.regime exactly and is validated
against regime()'s D_o and B_o at run time (asserts machine-precision match), so
this script cannot silently drift from the committed operator.

Outputs:
    results/centroid_shift_test.csv
    results/centroid_shift_test.png
    results/centroid_shift_test_summary.txt

Usage:
    python scripts/11_centroid_shift_test.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.regime import regime, _fit, _ring_density

_spec = importlib.util.spec_from_file_location("_setup",
                                               Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
ORIGIN_YEAR, ENDPOINT_YEAR, ROBUST_YEAR = 2019, 2025, 2024


def post_centroids(inp, tech, L, ell, rho=0.5, lam_over=1.0):
    """Pre- and post-technology mass-normalised bundle centroids (Cartesian),
    mirroring the iota_o computation in model.regime (no wedge, survival on).
    Returns mu_pre (n,2), mu_post (n,2), and (D_o, B_o) for validation."""
    field, grid = inp.field, inp.grid
    codes = inp.occ_codes()
    L = np.asarray(L, float)

    bx = inp.bundles
    b_xi, b_chi, b_w = (bx["xi"].to_numpy(), bx["chi"].to_numpy(),
                        bx["b"].to_numpy())
    row_of = pd.Index(codes).get_indexer(bx["onet_code"].to_numpy())
    a_task = tech.operated_share(b_xi, b_chi, field, R, TAU)
    D_o = np.bincount(row_of, weights=b_w * a_task, minlength=len(codes))

    a_grid = tech.operated_share(grid.xi, grid.chi, field, R, TAU)
    surv = 1.0 - a_grid
    M = GAMMA * float(np.sum(L * D_o))
    s = M * _ring_density(tech, grid)
    e = _fit(inp, ell, rho, lam_over)
    C = (L[:, None] * e).sum(axis=0)
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    share = np.where(C > 0, (L[:, None] * e) / C[None, :], 0.0)
    iota = (s * surv)[None, :] * Phi[None, :] * share
    B_o = (iota * grid.area[None, :]).sum(axis=1)

    xt, yt = b_chi * np.cos(b_xi), b_chi * np.sin(b_xi)
    mu_pre = np.column_stack([
        np.bincount(row_of, weights=b_w * xt, minlength=len(codes)),
        np.bincount(row_of, weights=b_w * yt, minlength=len(codes)),
    ])
    strip_m = 1.0 - D_o
    strip_x = np.bincount(row_of, weights=b_w * (1 - a_task) * xt, minlength=len(codes))
    strip_y = np.bincount(row_of, weights=b_w * (1 - a_task) * yt, minlength=len(codes))
    pw = np.where(L[:, None] > 0, iota / L[:, None], 0.0)   # iota/L (L cancels)
    refill_m = (pw * grid.area[None, :]).sum(axis=1)
    refill_x = (pw * (grid.x * grid.area)[None, :]).sum(axis=1)
    refill_y = (pw * (grid.y * grid.area)[None, :]).sum(axis=1)
    tot = strip_m + refill_m
    mu_post = np.column_stack([(strip_x + refill_x) / tot,
                               (strip_y + refill_y) / tot])
    return mu_pre, mu_post, D_o, B_o


def oews_median(year: int) -> pd.Series:
    """Detailed-SOC median hourly wage from OEWS national (handles the
    lowercase 2019 header and the '*'/'#' suppression flags)."""
    df = pd.ExcelFile(DATA / f"national_M{year}_dl.xlsx").parse(f"national_M{year}_dl")
    df.columns = [c.upper() for c in df.columns]
    det = df[df["O_GROUP"].astype(str).str.lower() == "detailed"].copy()
    det["H"] = pd.to_numeric(det["H_MEDIAN"], errors="coerce")
    return det.dropna(subset=["H"]).drop_duplicates("OCC_CODE").set_index("OCC_CODE")["H"]


def _corr_block(label, x, y):
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    return (f"  [{label}]  N={len(x)}  Spearman {sr:+.3f} (p={sp:.1e})  "
            f"Pearson {pr:+.3f} (p={pp:.1e})")


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    codes = inp.occ_codes()
    field = inp.field

    mu_pre, mu_post, D_o, B_o = post_centroids(inp, tech, L0, ell)

    # validate against the committed operator
    diag = regime(inp, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    gapD = float(np.max(np.abs(D_o - diag["D_o"])))
    gapB = float(np.max(np.abs(B_o - diag["B_o"])))
    assert gapD < 1e-9 and gapB < 1e-9, f"regime mismatch: D {gapD:.2e}, B {gapB:.2e}"

    dmu = mu_post - mu_pre
    xi_o, chi_o = occ["xi"].to_numpy(), occ["chi"].to_numpy()
    g_r, g_ang = field.grad_log_pi(xi_o, chi_o)
    gx = g_r * np.cos(xi_o) - g_ang * np.sin(xi_o)
    gy = g_r * np.sin(xi_o) + g_ang * np.cos(xi_o)
    proj = dmu[:, 0] * gx + dmu[:, 1] * gy        # predicted d ln Pi along the shift

    res = pd.DataFrame({
        "onet_code": codes, "OCC_CODE": occ["OCC_CODE"].to_numpy(),
        "Title": occ["Title"].to_numpy(), "L0": L0,
        "dmu_mag": np.hypot(dmu[:, 0], dmu[:, 1]),
        "proj": proj, "dW_bundle": diag["dW_bundle"],
    })

    h0 = oews_median(ORIGIN_YEAR)
    res["w_origin"] = res["OCC_CODE"].map(h0)
    res["w_end"] = res["OCC_CODE"].map(oews_median(ENDPOINT_YEAR))
    res = res.dropna(subset=["w_origin", "w_end"]).copy()
    res["dlnw"] = np.log(res["w_end"]) - np.log(res["w_origin"])

    lines = [
        "Centroid-shift consistency test (NON-CAUSAL directional check).",
        f"  calibrated AI field; operator at baseline L0; gamma {GAMMA}, ell {ell:.4f}",
        f"  validation vs regime(): max|dD_o| {gapD:.1e}, max|dB_o| {gapB:.1e}",
        f"  model occupations {len(codes)}; with OEWS {ORIGIN_YEAR}&{ENDPOINT_YEAR} "
        f"wage {len(res)} ({res['OCC_CODE'].nunique()} unique SOC)",
        f"  proj<0 share {100*np.mean(res['proj']<0):.1f}% (field pushes bundles "
        f"down-gradient); median proj {res['proj'].median():+.4f}",
        f"  overall mean dlnw {ORIGIN_YEAR}->{ENDPOINT_YEAR} {res['dlnw'].mean():+.4f} "
        f"(nominal; differenced out by the rank correlation)",
        "",
        f"Predicted directional pressure vs observed wage change {ORIGIN_YEAR}->{ENDPOINT_YEAR}:",
    ]
    lines.append(_corr_block("occupation level: proj vs dlnw", res["proj"], res["dlnw"]))
    lines.append(_corr_block("occupation level: dW_bundle vs dlnw", res["dW_bundle"], res["dlnw"]))

    # SOC-collapsed (employment-weighted proj)
    soc = (res.groupby("OCC_CODE")
           .apply(lambda g: pd.Series({"proj": np.average(g["proj"], weights=g["L0"]),
                                       "dlnw": g["dlnw"].iloc[0]}), include_groups=False)
           .reset_index())
    lines.append(_corr_block("SOC level (emp-wtd proj) vs dlnw", soc["proj"], soc["dlnw"]))

    # robustness endpoint
    res["w_rob"] = res["OCC_CODE"].map(oews_median(ROBUST_YEAR))
    rr = res.dropna(subset=["w_rob"]).copy()
    rr["dlnw_rob"] = np.log(rr["w_rob"]) - np.log(rr["w_origin"])
    socr = (rr.groupby("OCC_CODE")
            .apply(lambda g: pd.Series({"proj": np.average(g["proj"], weights=g["L0"]),
                                        "dlnw": g["dlnw_rob"].iloc[0]}), include_groups=False)
            .reset_index())
    lines += ["", f"Robustness endpoint {ORIGIN_YEAR}->{ROBUST_YEAR}:"]
    lines.append(_corr_block("SOC level (emp-wtd proj) vs dlnw", socr["proj"], socr["dlnw"]))

    res.to_csv(RESULTS / "centroid_shift_test.csv", index=False)
    (RESULTS / "centroid_shift_test_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # scatter
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(res["proj"], res["dlnw"], s=10 + 1.2e4 * res["L0"],
               c="#3b528b", alpha=0.45, edgecolors="none")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.axhline(res["dlnw"].mean(), color="0.6", lw=0.8, ls=":")
    sr = spearmanr(res["proj"], res["dlnw"])[0]
    ax.set_xlabel(r"predicted directional wage pressure  $\Delta\mu_o\cdot\nabla\ln\Pi$")
    ax.set_ylabel(rf"observed $\Delta\ln w$  ({ORIGIN_YEAR}$\to${ENDPOINT_YEAR})")
    ax.set_title(f"Centroid-shift consistency (Spearman {sr:+.2f}); non-causal")
    fig.tight_layout()
    fig.savefig(RESULTS / "centroid_shift_test.png", dpi=150)
    plt.close(fig)
    print(f"wrote {RESULTS/'centroid_shift_test.csv'}, .png, _summary.txt")


if __name__ == "__main__":
    main()
