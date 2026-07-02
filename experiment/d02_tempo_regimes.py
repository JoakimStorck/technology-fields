"""
d02_tempo_regimes.py
--------------------
Producer of fig:tempo and the manuscript's headline correlation: the
destination of reinstated work under a gradual transition (theta = 1 year,
redistribution faster than the T_shock = 5 year diffusion) against a congested
one (theta = 15 years, redistribution much slower), through the same
technology and the same field. theta sets theta_L and theta_abs together.

Owns one nameable part of the argument: the final allocation of reinstated
task mass across occupations depends on the tempo. The headline number is the
Pearson correlation of final reinstated mass across the pre-existing
occupations, unweighted; the rank (Spearman) and employment-weighted Pearson
correlations are reported alongside as sensitivity. All numbers are written to
experiment/results/ and asserted against the frozen baseline below.

Births are disabled (max_births = 0) so the occupation set is identical across
regimes and the correlation is well defined; the birth layer belongs to the
birth extension (probe_allocation.py).

Usage: python experiment/d02_tempo_regimes.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

THETA_GRADUAL, THETA_CONGESTED = 1.0, 15.0
T_SHOCK, T_MAX, DT = 5.0, 60.0, 0.2

# Frozen baseline (this machine, this calibration), absolute tolerance 0.02.
BASELINE_PEARSON = 0.371
BASELINE_SPEARMAN = 0.308


def weighted_pearson(x, y, w):
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))


def main():
    layer = iface.load_static_layer()
    runs = {}
    for th in (THETA_GRADUAL, THETA_CONGESTED):
        dyn, rec, occ = rd.main(theta_L=th, theta_abs=th, T_shock=T_SHOCK,
                                T_max=T_MAX, dt=DT, max_births=0,
                                verbose=False, layer=layer)
        assert rec["U_tot"][-1] < 1e-8, f"U does not drain at theta={th}"
        runs[th] = dyn.reinst[:dyn.n0].copy()
    n0 = runs[THETA_GRADUAL].size
    rg, rc = runs[THETA_GRADUAL], runs[THETA_CONGESTED]

    pear = float(pearsonr(rg, rc)[0])
    spear = float(spearmanr(rg, rc)[0])
    wpear = float(weighted_pearson(rg, rc, layer.L0))
    assert abs(pear - BASELINE_PEARSON) < 0.02, f"Pearson drifted: {pear:.3f}"
    assert abs(spear - BASELINE_SPEARMAN) < 0.02, f"Spearman drifted: {spear:.3f}"

    occ = layer.occ
    lines = [
        "tempo_regimes -- the destination of reinstated work depends on the tempo",
        f"gradual theta = {THETA_GRADUAL:g}, congested theta = {THETA_CONGESTED:g} "
        f"(theta_L = theta_abs = theta), T_shock = {T_SHOCK:g} years, births off",
        f"final reinstated mass per occupation, n = {n0} pre-existing occupations",
        "",
        f"corr(gradual, congested), Pearson, unweighted:      {pear:+.3f}   (the headline)",
        f"corr(gradual, congested), Spearman rank:            {spear:+.3f}",
        f"corr(gradual, congested), Pearson, L0-weighted:     {wpear:+.3f}",
        f"reinstated mass totals: gradual {rg.sum():.4f}, congested {rc.sum():.4f}",
        "",
        "top gainers, gradual regime:",
    ]
    for i in np.argsort(rg)[::-1][:8]:
        lines.append(f"  {rg[i]:.5f}  {occ['Title'].iloc[i]}")
    lines.append("top gainers, congested regime:")
    for i in np.argsort(rc)[::-1][:8]:
        lines.append(f"  {rc[i]:.5f}  {occ['Title'].iloc[i]}")

    # ---- csv ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"OCC_CODE": occ["OCC_CODE"], "Title": occ["Title"],
                  "L0": layer.L0, "reinst_gradual": rg, "reinst_congested": rc}
                 ).to_csv(iface.RESULTS / "tempo_regimes.csv", index=False)

    # ---- figure: two disk panels ----
    grid = layer.inp.grid
    gx = grid.chi * np.cos(grid.xi)
    gy = grid.chi * np.sin(grid.xi)
    a_mature = layer.set_maturity(layer.tech.A_K)
    seed = layer.eq.g_hat * (1.0 - a_mature)          # survival-gated seeding field
    seed = seed / seed.max()
    mux = occ["chi"].to_numpy() * np.cos(occ["xi"].to_numpy())
    muy = occ["chi"].to_numpy() * np.sin(occ["xi"].to_numpy())
    kx = layer.tech.chi_K * np.cos(layer.tech.xi_K)
    ky = layer.tech.chi_K * np.sin(layer.tech.xi_K)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    panels = [(axes[0], rg, f"Gradual transition ($\\theta = {THETA_GRADUAL:g} < "
                            f"T_{{shock}} = {T_SHOCK:g}$)", "#2C5A57"),
              (axes[1], rc, f"Congested transition ($\\theta = {THETA_CONGESTED:g} > "
                            f"T_{{shock}}$)", "#B5532A")]
    smax = max(rg.max(), rc.max())
    for ax, r, title, col in panels:
        ax.scatter(gx, gy, s=6, c="0.55", alpha=0.85 * seed, linewidths=0)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1)
        ax.plot(kx + layer.tech.z_K * np.cos(th), ky + layer.tech.z_K * np.sin(th),
                ls="--", color="0.35", lw=1.2)
        ax.scatter(mux, muy, s=600 * r / smax + 1, c=col, alpha=0.7, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
    fig.suptitle("Reinstated task mass gained per occupation "
                 f"(final allocations correlate at {pear:+.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "tempo_regimes.png", dpi=150)

    iface.write_summary("tempo_regimes", lines)


if __name__ == "__main__":
    main()
