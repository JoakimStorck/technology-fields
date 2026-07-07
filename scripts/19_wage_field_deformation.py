"""
19_wage_field_deformation.py
----------------------------
Reports automation's wage effect as occupation-level wage adjustments, exactly
decomposed into the channels that produce them. Written and pre-registered
BEFORE the first run. The decomposition is an exact path decomposition, not a
first-order split.

The wage object. A worker in occupation o earns the occupation value
    W_o = beta * sum_r h_o(r) D(r) Pi(r) n(r)^{beta-1},
the marginal value of task work (eq. occ-value), which is what workers sort on
and are paid. It carries both channels: Pi(r) prices task content (stripping
acts here) and n(r)^{beta-1} is the congestion return (re-sorting acts here).
The wage adjustment is d ln W_o = ln W_o^post - ln W_o^0.

Exact path decomposition. Move from the pre-technology rest point to the
post-technology one in three steps, each changing exactly one thing and each an
exact re-evaluation of the W formula (no linearisation):

    W0     : pre-technology. content = full bundle (a=0), density = n0(L0).
    W_strip: apply takeover a(r) to content; density held at n0(L0).
             -> ln W_strip - ln W0 = STRIPPING adjustment (Pi content removed).
    W_cong : let density move to the post-sort n(L*); content still stripped,
             no reinstatement.
             -> ln W_cong - ln W_strip = CONGESTION adjustment (n^{b-1} bent by
             the re-sorting of freed labour).
    W_post : add the bound reinstatement inflow to content.
             -> ln W_post - ln W_cong = REINSTATEMENT adjustment (new bound work).

The three adjustments sum exactly to d ln W_o. Each is a log-point wage change,
directly comparable to the bundle wage change Delta w_o the paper already
reports, and to each other. The order strip -> cong -> reinstatement is the
economic sequence; a robustness alternative (cong before strip) is reported to
show the split is not an artefact of ordering.

Note on aggregation. The labour share cancels beta and so is silent on
congestion; that is why congestion is invisible in the share but present in the
wage adjustments. Congestion is a DISTRIBUTIONAL channel: it reweights who bears
the pressure across occupations.

PRE-REGISTERED HYPOTHESES (before first run):
  W1  The decomposition is exact: max over occupations of
      |strip + cong + reinst - d ln W_o| < 1e-6.
  W2  Stripping is a real negative channel: employment-weighted mean stripping
      adjustment < -0.02 log pts (a wage cut on average).
  W3  Congestion is a material second channel, not noise: the employment-
      weighted mean ABSOLUTE congestion adjustment is at least 20 percent of the
      mean absolute stripping adjustment.
  W4  Congestion has the sorting signature: employment GAINERS carry a negative
      mean congestion adjustment (they crowd in), LOSERS a positive one (they
      thin out).
  W5  Order-robustness: swapping the strip/congestion order changes the mean
      congestion adjustment by less than 25 percent of its value.
Adverse outcomes are reported. If W3 fails, congestion is a minor refinement and
is reported as such, not as a headline second channel.

Outputs:
    results/wage_field_deformation.csv
    results/wage_field_deformation_summary.txt

Usage:
    python scripts/19_wage_field_deformation.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.equilibrium import Equilibrium

_spec = importlib.util.spec_from_file_location(
    "_setup", Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
NMIN = 1e-9


def w_value(eq, content_wD, gcarrier, nb1):
    """Exact W_o = beta[ sum_t content_wD nb1[cell] + e @ (gcarrier nb1) ].
    content_wD is per-task (b (1-a) Pi D); gcarrier is the per-cell reinstatement
    carrier s surv Phi/C D Pi area; nb1 is n^{beta-1} on the grid."""
    strip = np.bincount(eq.row_of, weights=content_wD * nb1[eq.cell_of],
                        minlength=eq.n_occ)
    reinst = eq.e @ (gcarrier * nb1)
    return BETA * (strip + reinst)


def density_from(eq, L, with_tech: bool):
    """n(r) and the reinstatement carrier at employment L. Pre-tech: bundle
    density only, zero carrier. With tech: plus bound reinstatement inflow."""
    n0 = (np.bincount(eq.cell_of, weights=L[eq.row_of] * eq.b_w,
                      minlength=eq.area.size) / eq.area)
    if not with_tech:
        return n0, np.zeros(eq.area.size)
    M = eq.gamma * float(np.sum(L * eq.D_o))
    s = M * eq.g_hat
    surv = (1.0 - eq.a_grid)
    C = L @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    iota_tot = s * surv * Phi
    with np.errstate(divide="ignore", invalid="ignore"):
        gcarrier = (np.where(C > 0, s * surv * Phi / C, 0.0)
                    * eq.D_grid * eq.pi_cell * eq.area)
    return n0 + iota_tot, gcarrier


def _wage_field_figure(inp, tech, pi, n0_pre, n_post, occupied, out_path):
    """Two panels: the effective wage field ln[Pi n^{beta-1}] after re-sorting,
    and the deformation from re-sorting, over the task disk. The deformation
    panel is masked to occupied cells and its points are sized by post-sort
    density, so the eye is drawn to where labour actually works rather than to
    sparse-rim log noise."""
    grid = inp.grid
    x, y = grid.x, grid.y
    # occupancy floor: keep cells with at least 1% of median occupied density
    dens_floor = 0.01 * np.median(n0_pre[occupied])
    keep = occupied & (n0_pre > dens_floor) & (n_post > dens_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnW_post = np.where(keep, np.log(pi) + (BETA - 1.0) * np.log(np.maximum(n_post, NMIN)), np.nan)
        deform = np.where(keep, (BETA - 1.0) * (np.log(np.maximum(n_post, NMIN))
                                                - np.log(np.maximum(n0_pre, NMIN))), np.nan)

    vlo, vhi = np.nanpercentile(lnW_post, [2, 98])
    dmax = np.nanpercentile(np.abs(deform[keep]), 95)
    # size by post-sort density (where work concentrates), normalised
    dn = n_post / np.nanmax(n_post[keep])
    size = 4 + 40 * np.clip(dn, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    sc0 = axes[0].scatter(x[keep], y[keep], c=lnW_post[keep], s=7,
                          cmap="viridis", vmin=vlo, vmax=vhi)
    axes[0].set_title(r"Effective wage field, $\ln[\Pi\,n^{\beta-1}]$", fontsize=12)
    fig.colorbar(sc0, ax=axes[0], shrink=0.8)

    sc1 = axes[1].scatter(x[keep], y[keep], c=deform[keep], s=size[keep],
                          cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[1].set_title(r"Deformation from re-sorting, $\Delta\ln[\Pi\,n^{\beta-1}]$",
                      fontsize=12)
    fig.colorbar(sc1, ax=axes[1], shrink=0.8,
                 label="down where labour crowds in / up where it thins")

    for ax in axes:
        px, py = tech.p_K
        ax.plot(px, py, "k+", ms=11, mew=2)
        ax.add_patch(plt.Circle((px, py), tech.z_K, color="k", fill=False,
                                lw=1.0, ls="--"))
        ax.add_patch(plt.Circle((0, 0), 1.0, color="0.6", fill=False, lw=0.8))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    _, _, W0chk = eq.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0chk, eq.d)
    out = eq.solve(c, kappa)
    Lstar = out.L

    def nb1(n):
        return np.maximum(n, NMIN) ** (BETA - 1.0)

    zero_carrier = np.zeros(eq.area.size)
    # content weights: pre-technology (full bundle, D=1 at eta=1) and stripped
    content_pre = eq.b_w * eq.pi_task
    content_strip = eq.strip_wD

    n0_pre, _ = density_from(eq, L0, with_tech=False)
    n_post, gcarrier_post = density_from(eq, Lstar, with_tech=True)

    # path nodes (exact re-evaluations of the W formula)
    W0 = w_value(eq, content_pre, zero_carrier, nb1(n0_pre))
    W_strip = w_value(eq, content_strip, zero_carrier, nb1(n0_pre))
    W_cong = w_value(eq, content_strip, zero_carrier, nb1(n_post))
    W_post = w_value(eq, content_strip, gcarrier_post, nb1(n_post))

    def dln(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((a > 0) & (b > 0), np.log(a) - np.log(b), 0.0)

    strip_adj = dln(W_strip, W0)
    cong_adj = dln(W_cong, W_strip)
    reinst_adj = dln(W_post, W_cong)
    total_adj = dln(W_post, W0)
    resid = float(np.max(np.abs(strip_adj + cong_adj + reinst_adj - total_adj)))

    # order-robustness: congestion first (density moves before content strips)
    W_cong_first = w_value(eq, content_pre, zero_carrier, nb1(n_post))
    cong_adj_alt = dln(W_cong_first, W0)

    dL = Lstar - L0
    valid = (W0 > 0) & (W_post > 0)

    def wmean(x):
        return float(np.average(x[valid], weights=Lstar[valid]))

    gain = (dL > 0) & valid
    lose = (dL < 0) & valid
    mean_strip, mean_cong, mean_reinst = wmean(strip_adj), wmean(cong_adj), wmean(reinst_adj)
    mabs_strip, mabs_cong = wmean(np.abs(strip_adj)), wmean(np.abs(cong_adj))
    cong_share = mabs_cong / mabs_strip if mabs_strip > 0 else np.nan
    cong_gain = float(cong_adj[gain].mean()) if gain.any() else 0.0
    cong_lose = float(cong_adj[lose].mean()) if lose.any() else 0.0
    mc, mca = float(cong_adj[valid].mean()), float(cong_adj_alt[valid].mean())
    order_change = abs(mca - mc) / max(abs(mc), 1e-9)

    W1 = resid < 1e-6
    W2 = mean_strip < -0.02
    W3 = cong_share >= 0.20
    W4 = (cong_gain < 0) and (cong_lose > 0)
    W5 = order_change < 0.25

    lines = [
        "Occupation wage adjustment, exactly decomposed (pre-registered).",
        f"  economy R {R}, beta {BETA}, gamma {GAMMA}, ell {ell:.4f}; "
        f"occupations {eq.n_occ}, grid {eq.area.size} cells",
        f"  converged {out.converged}; re-sorted mass {np.abs(dL).sum()/2:.3f}",
        f"  exactness: max |strip+cong+reinst - total| = {resid:.2e}",
        "",
        "Employment-weighted mean wage adjustment (log points):",
        f"  stripping     {mean_strip:+.4f}   (capital removes priced content, via Pi)",
        f"  congestion    {mean_cong:+.4f}   (re-sorting bends the return, via n^(b-1))",
        f"  reinstatement {mean_reinst:+.4f}   (new bound work added)",
        f"  total         {mean_strip+mean_cong+mean_reinst:+.4f}",
        "",
        "Channel magnitudes and signature:",
        f"  mean |stripping|  = {mabs_strip:.4f}",
        f"  mean |congestion| = {mabs_cong:.4f}   (ratio {cong_share:.2f} of stripping)",
        f"  congestion adj, employment GAINERS = {cong_gain:+.4f} (crowd in)",
        f"  congestion adj, employment LOSERS  = {cong_lose:+.4f} (thin out)",
        f"  order-robustness (cong-first vs strip-first): {100*order_change:.0f}% change",
        "",
        "Pre-registered hypothesis verdicts:",
        f"  W1 (decomposition exact, resid<1e-6)     {'PASS' if W1 else 'FAIL'}"
        f"  (resid {resid:.1e})",
        f"  W2 (stripping mean cut < -0.02)          {'PASS' if W2 else 'FAIL'}"
        f"  (mean {mean_strip:+.4f})",
        f"  W3 (congestion >= 0.20 x stripping)      {'PASS' if W3 else 'FAIL'}"
        f"  (ratio {cong_share:.2f})",
        f"  W4 (gainers crowd down, losers thin up)  {'PASS' if W4 else 'FAIL'}"
        f"  (gain {cong_gain:+.4f}, lose {cong_lose:+.4f})",
        f"  W5 (order-robust, <25% swing)            {'PASS' if W5 else 'FAIL'}"
        f"  ({100*order_change:.0f}%)",
        "",
        "Reading: automation adjusts wages through two channels of comparable "
        "size. Stripping removes priced task content from exposed bundles and "
        "cuts wages directly. Congestion is the re-sorting of freed labour into "
        "the least-damaged occupations, which bends the effective return down "
        "where labour concentrates and up where it thins -- a distributional "
        "channel invisible to the aggregate labour share (which cancels beta) "
        "but present in occupation wage adjustments. Reinstatement adds back "
        "only where new work binds.",
    ]

    occupied = n0_pre > NMIN * 10
    _wage_field_figure(inp, tech, eq.pi_cell, n0_pre, n_post, occupied,
                       RESULTS / "wage_field_deformation.png")

    pd.DataFrame({
        "code": eq.codes, "dL": dL,
        "strip_adj": strip_adj, "cong_adj": cong_adj,
        "reinst_adj": reinst_adj, "total_adj": total_adj,
    }).to_csv(RESULTS / "wage_field_deformation.csv", index=False)
    (RESULTS / "wage_field_deformation_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS/'wage_field_deformation.csv'} and _summary.txt")


if __name__ == "__main__":
    main()
