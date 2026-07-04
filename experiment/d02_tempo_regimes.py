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
correlations are reported alongside as sensitivity.

Also produces the per-occupation mechanism evidence behind fig:tempo. An
occupation's UNCONSTRAINED CLAIM is the seed-weighted share the claim law
(eq. claim, beta_m = 3) would allocate with no size cap, computed on the
survival-gated seeding field at the mature technology (an end-state
approximation of the flow-weighted claim); its SIZE is the pre-shock task
mass that caps the absorption rate. Per regime the script reports the
correlation of absorption with each, and the quartile shares of absorbed
mass by claim rank and by size rank. All numbers are written to
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

    # ---- mechanism evidence: unconstrained claim vs size ----
    dyn_ref = None
    # rerun-free: recover FIT and original from a fresh Dyn at the layer state
    import importlib.util as _il
    _dspec = _il.spec_from_file_location("run_dynamic", REPO / "experiment" / "run_dynamic.py")
    a_mature = layer.set_maturity(layer.tech.A_K)
    seed_w = layer.eq.g_hat * (1.0 - a_mature) * layer.eq.area
    seed_w = seed_w / seed_w.sum()
    dyn_ref = rd.Dyn(layer.eq, layer.inp, layer.L0, layer.ell,
                     layer.rho, lam_over=layer.lam_over)
    BETA_M = 3.0
    Wb = dyn_ref.FIT[:n0] ** BETA_M
    claim = (Wb / Wb.sum(0)[None, :]) @ seed_w
    size = dyn_ref.original[:n0]

    def quartile_shares(r, key):
        q = np.quantile(key, [0.25, 0.5, 0.75])
        idx = np.digitize(key, q)
        return [float(r[idx == k].sum() / r.sum()) for k in range(4)]

    mech = {}
    for name, r in (("gradual", rg), ("congested", rc)):
        mech[name] = dict(
            claim_p=float(pearsonr(r, claim)[0]), claim_s=float(spearmanr(r, claim)[0]),
            size_p=float(pearsonr(r, size)[0]), size_s=float(spearmanr(r, size)[0]),
            q_claim=quartile_shares(r, claim), q_size=quartile_shares(r, size))
    assert abs(mech["gradual"]["claim_p"] - 0.833) < 0.02
    assert abs(mech["congested"]["size_s"] - 0.950) < 0.02

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
        "mechanism evidence (claim = unconstrained seed-weighted claim share at "
        f"beta_m = {BETA_M:g}; size = pre-shock task mass):",
    ]
    for name in ("gradual", "congested"):
        m = mech[name]
        lines += [
            f"  {name}:",
            f"    corr(absorption, claim): pearson {m['claim_p']:+.3f}  "
            f"spearman {m['claim_s']:+.3f}",
            f"    corr(absorption, size):  pearson {m['size_p']:+.3f}  "
            f"spearman {m['size_s']:+.3f}",
            "    share of reinstated mass by claim quartile (low->top): "
            + "  ".join(f"{100*x:.0f}%" for x in m["q_claim"]),
            "    share of reinstated mass by size quartile  (low->top): "
            + "  ".join(f"{100*x:.0f}%" for x in m["q_size"]),
        ]
    # gainer table (P5): top ten per regime by absolute reinstated mass,
    # with shares of the regime total; top-three identities frozen.
    TOP3_G = ("Preventive Medicine Physicians", "Physicists",
              "Natural Sciences Managers")
    TOP3_C = ("Fast Food and Counter Workers", "Preventive Medicine Physicians",
              "Waiters and Waitresses")
    ig = np.argsort(rg)[::-1][:10]
    ic = np.argsort(rc)[::-1][:10]
    assert tuple(occ["Title"].iloc[ig[:3]]) == TOP3_G, \
        tuple(occ["Title"].iloc[ig[:3]])
    assert tuple(occ["Title"].iloc[ic[:3]]) == TOP3_C, \
        tuple(occ["Title"].iloc[ic[:3]])
    lines += [
        "",
        "top gainers (share of regime total reinstated mass), gradual:",
    ]
    for i in ig:
        lines.append(f"  {rg[i]/rg.sum():6.2%}  {occ['Title'].iloc[i]}")
    lines.append("top gainers, congested:")
    for i in ic:
        lines.append(f"  {rc[i]/rc.sum():6.2%}  {occ['Title'].iloc[i]}")

    # ---- csv ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"OCC_CODE": occ["OCC_CODE"], "Title": occ["Title"],
                  "L0": layer.L0, "reinst_gradual": rg, "reinst_congested": rc,
                  "claim_unconstrained": claim, "size_task_mass": size}
                 ).to_csv(iface.RESULTS / "tempo_regimes.csv", index=False)
    pd.DataFrame({"rank": np.arange(1, 11),
                  "gradual_title": occ["Title"].iloc[ig].to_numpy(),
                  "gradual_share": (rg[ig]/rg.sum()).round(4),
                  "congested_title": occ["Title"].iloc[ic].to_numpy(),
                  "congested_share": (rc[ic]/rc.sum()).round(4)}
                 ).to_csv(iface.RESULTS / "tempo_gainers.csv", index=False)

    # ---- figure: two disk panels ----
    grid = layer.inp.grid
    gx = grid.chi * np.cos(grid.xi)
    gy = grid.chi * np.sin(grid.xi)
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
