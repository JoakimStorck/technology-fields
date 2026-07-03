"""
d12_baseline_economy.py
-----------------------
The inherited economy in one figure (manuscript sec. 2 and appendix "The
inherited economy", P1 of the referee response): the unit disk with the
estimated price field ln Pi as contours, the occupation centroids sized by
pre-shock employment L0, and the calibrated technology field phi_K with its
gradient ring at radius z_K -- drawn from the same StaticLayer objects every
d-script consumes, so the figure is the interface made visible.

Owns: the baseline figure of the self-containedness revision. No new
quantities. The summary restates the identifying numbers of the static
layer -- price-field coefficients (spec S1_field of
results/wage_field_coefficients.csv, the estimates of the geometry paper's
wage-harmonic regression), gate weights v_k (spec V_mediation, priced
clusters only), the technology calibration, the economy parameters, and the
mobility reference at the dynamic evaluation state A_K = 0 -- and asserts
each against its frozen baseline, so a drifted calibration fails loudly
here before it reaches the appendix tables.

Usage: python experiment/d12_baseline_economy.py   (under a minute)
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
import pandas as pd


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")

# ── frozen baselines ──────────────────────────────────────────────────
# Price field, spec S1_field (results/wage_field_coefficients.csv).
PRICE_BASE = {"m0": 3.4195, "m1": 0.1921, "m2": 0.0889,
              "m3": -0.0647, "m4": 0.0082, "m5": 0.8913}
PRICE_SE = {"m0": 0.0355, "m1": 0.0484, "m2": 0.0512,
            "m3": 0.0897, "m4": 0.1123, "m5": 0.1338}
# Gate weights, spec V_mediation, priced clusters.
V_BASE = {"S1": 0.3280, "S2": 0.0493}
# Technology calibration (cognitive field, static Table technology-calibration).
TECH_BASE = {"xi_deg": 38.4, "chi": 0.4633, "z": 0.5825, "A": 0.6032}
# Interpretable readiness scale and mobility reference at A_K = 0.
ELL_BASE, KAPPA_BASE, C_BASE = 0.133, 11.84, 23.02
N_OCC_BASE = 849


def main():
    layer = iface.load_static_layer()
    eq, tech, occ = layer.eq, layer.tech, layer.occ

    # ── assert the inherited numbers before drawing them ──────────────
    field = eq.field if hasattr(eq, "field") else None
    coefs = pd.read_csv(REPO / "results" / "wage_field_coefficients.csv")
    s1 = coefs[coefs["spec"] == "S1_field"].set_index("param")
    for p, v in PRICE_BASE.items():
        assert abs(float(s1.loc[p, "coef"]) - v) < 5e-4, (p, float(s1.loc[p, "coef"]))
        assert abs(float(s1.loc[p, "se"]) - PRICE_SE[p]) < 5e-4, (p, "se")

    from model.capability_field import CapabilityField, PRICED
    cap = CapabilityField.from_results()
    assert PRICED == ("S1", "S2")
    for k, v in V_BASE.items():
        assert abs(cap.v[k] - v) < 5e-4, (k, cap.v[k])

    assert abs(np.degrees(tech.xi_K) - TECH_BASE["xi_deg"]) < 0.05
    assert abs(tech.chi_K - TECH_BASE["chi"]) < 5e-4
    assert abs(tech.z_K - TECH_BASE["z"]) < 5e-4
    assert abs(tech.A_K - TECH_BASE["A"]) < 5e-4

    assert abs(layer.ell - ELL_BASE) < 5e-4, layer.ell
    assert abs(layer.kappa - KAPPA_BASE) < 5e-3, layer.kappa
    assert abs(layer.c - C_BASE) < 5e-3, layer.c
    assert (layer.R, layer.tau, layer.beta, layer.gamma) == (18.0, 0.08, 0.5, 0.5)
    assert (layer.rho, layer.lam_over) == (0.5, 1.0)
    assert eq.n_occ == N_OCC_BASE, eq.n_occ

    # ── the figure: price contours, centroids, technology field ───────
    g = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= 1.0
    XI, CHI = np.arctan2(Y, X), np.hypot(X, Y)

    from model.price_field import PriceField
    pf = PriceField.from_results()
    LNPI = np.where(inside, pf.log_pi(XI, CHI), np.nan)
    PHI = np.where(inside, tech.phi(XI, CHI), np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.3", lw=1.0))

    cs = ax.contour(X, Y, LNPI, levels=10, colors="0.55", linewidths=0.8)
    ax.clabel(cs, fmt="%.1f", fontsize=7, colors="0.35")

    L0 = layer.L0 / layer.L0.sum()
    cx = occ["chi"].to_numpy() * np.cos(occ["xi"].to_numpy())
    cy = occ["chi"].to_numpy() * np.sin(occ["xi"].to_numpy())
    ax.scatter(cx, cy, s=3000 * L0, c="#1f3d7a", alpha=0.45, lw=0, zorder=3,
               label="occupation centroids (area = $L_0$ share)")

    ct = ax.contour(X, Y, PHI, levels=[0.1 * tech.A_K, 0.3 * tech.A_K,
                                       0.6 * tech.A_K, 0.9 * tech.A_K],
                    colors="#b3401f", linewidths=1.1)
    pkx = tech.chi_K * np.cos(tech.xi_K)
    pky = tech.chi_K * np.sin(tech.xi_K)
    ax.plot(pkx, pky, "x", color="#b3401f", ms=9, mew=2, zorder=4)
    ring = plt.Circle((pkx, pky), tech.z_K, fill=False, color="#b3401f",
                      ls="--", lw=1.2)
    ax.add_patch(ring)

    ax.plot([], [], color="0.55", lw=0.8, label=r"$\ln\Pi(\mathbf{r})$ contours")
    ax.plot([], [], color="#b3401f", lw=1.1,
            label=r"technology $\phi_K$ at $\bar A_K$; ring at $z_K$ (dashed)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-1.05, 1.05), ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "baseline_economy.png", dpi=150)
    plt.close(fig)

    # ── summary ────────────────────────────────────────────────────────
    lines = [
        "The inherited economy: everything the dynamics consume across",
        "experiment/_interface.py, asserted against frozen baselines.",
        "",
        f"occupations N = {eq.n_occ}",
        "",
        "price field ln Pi (spec S1_field, HC3 se in parens; N = 785, R2 = 0.523):",
    ] + [
        f"   {p} = {float(s1.loc[p, 'coef']):+.4f}  ({float(s1.loc[p, 'se']):.4f})"
        for p in PRICE_BASE
    ] + [
        "",
        "gate weights v_k (spec V_mediation, priced clusters K = {S1, S2}):",
        f"   S1 = {cap.v['S1']:.4f}   S2 = {cap.v['S2']:.4f}",
        "",
        "technology (cognitive field, calibrated to task-level AI exposure):",
        f"   xi_K = {np.degrees(tech.xi_K):.1f} deg   chi_K = {tech.chi_K:.4f}"
        f"   z_K = {tech.z_K:.4f}   A_K = {tech.A_K:.4f}",
        "",
        "economy parameters (static Table economy-parameters):",
        f"   R = {layer.R}   tau = {layer.tau}   beta = {layer.beta}"
        f"   gamma = {layer.gamma}",
        f"   ell = {layer.ell:.3f} (one within-direction SD of the priced-"
        "capability index)",
        f"   rho = {layer.rho}   lam_over = {layer.lam_over}",
        "",
        "mobility reference at the dynamic evaluation state A_K = 0:",
        f"   nu (kappa) = {layer.kappa:.2f}   c_m = {layer.c:.2f}"
        "   (static table at calibrated A_K: 11.6, 22.6)",
    ]
    iface.write_summary("baseline_economy", lines)


if __name__ == "__main__":
    main()
