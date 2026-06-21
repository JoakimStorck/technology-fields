"""
10_demand_channel.py
--------------------
The demand channel and Bessen's threshold, over two technology fields.

Competitive pricing passes the automation cost saving to consumers; isoelastic
demand of elasticity eta scales each place's revenue by D(r) = (c(r)/Pi)^(1-eta),
which multiplies the price field in the occupation value W_o (closure 1: a
revenue multiplier, so the demand boost splits between higher wages and more
employment through the congestion beta). eta = 1 (unit elastic) gives D = 1 and
reproduces the cost-invariant equilibrium of script 09.

Two fields are run through the same economy:
  - COGNITIVE: the calibrated AI technology (north, broad; script 08).
  - MANUAL:    a hand-placed industrial/agricultural field (west, narrow),
               illustrative, with an open path to historical calibration
               (RTI / robot exposure for the field, Bessen elasticities for eta).

For each field we sweep eta and read the employment change in the field's own
automated region. The price gate makes the per-unit saving vanish at the
adoption margin (Pi/c -> 1 as a -> 1/2), so the demand channel is weak there
and displacement dominates for realistic eta; the Bessen growth regime (manual
employment rising with automation) appears only at high elasticity or large
infra-marginal saving. This reproduces the second-order-at-the-margin property
of Acemoglu & Restrepo (2018).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_setup",
                                               Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

import sys
sys.path.insert(0, str(REPO_ROOT))
from model.equilibrium import Equilibrium
from model.technology import Technology

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5

# manual field: hand-placed, illustrative (west / technical-physical pole,
# narrow reach, high effectiveness so it clears the price gate against cheap
# labour). See the historical-calibration note in the paper draft.
MANUAL = dict(xi_K=np.radians(180), chi_K=0.45, z_K=0.30, A_K=1.2, s_K=1.0)

ETAS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])


def _flip_eta(etas, pct):
    """Linear-interpolated eta where the employment change crosses zero
    (NaN if it does not cross in the swept range)."""
    s = np.sign(pct)
    for i in range(len(etas) - 1):
        if s[i] < 0 <= s[i + 1] or s[i] <= 0 < s[i + 1]:
            x0, x1, y0, y1 = etas[i], etas[i + 1], pct[i], pct[i + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return np.nan


def _margin_PioverC(tech, grid, field):
    """Pi/c at the adoption margin (a in [0.4, 0.6]) and deep in the core
    (a > 0.9): the gate sends Pi/c -> 1 at the margin (zero saving)."""
    a = tech.operated_share(grid.xi, grid.chi, field, R, TAU)
    phi = tech.phi(grid.xi, grid.chi)
    pi = field.pi(grid.xi, grid.chi)
    cratio = np.where(phi > 1e-9, R / (tech.s_K * phi * pi), 1.0)
    psi = (1 - a) + a * cratio
    margin = (a > 0.4) & (a < 0.6)
    core = a > 0.9
    mm = float((1 / psi[margin]).mean()) if margin.any() else np.nan
    cc = float((1 / psi[core]).mean()) if core.any() else np.nan
    return mm, cc, float(a[a > 0.05].mean())


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    ell = _setup.interpretable_ell(inp)
    wedge = _setup.load_wedge(occ)
    xi_o = occ["xi"].to_numpy()
    grid, field = inp.grid, inp.field

    techC = _setup.load_tech()
    techW = Technology(**MANUAL)

    # mobility reference from the baseline value (shared economy)
    eq0 = Equilibrium(inp, techC, R, TAU, GAMMA, ell, BETA, wedge=None)
    eq0.L0 = L0
    _, _, W0 = eq0.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0, eq0.d)

    fields = {
        "cognitive": (techC, (xi_o < np.radians(90)) | (xi_o > np.radians(270))),
        "manual":    (techW, (xi_o > np.radians(135)) & (xi_o < np.radians(225))),
    }

    rows, curves, lines = [], {}, [
        "Demand channel (closure 1: revenue multiplier on W_o, survival gate on).",
        f"  economy: R {R}, tau {TAU}, beta {BETA}, gamma {GAMMA}",
        f"  mobility: c {c:.3f}, kappa {kappa:.3f}",
        "",
    ]

    for name, (tech, region) in fields.items():
        mm, cc, amean = _margin_PioverC(tech, grid, field)
        L0reg = L0[region].sum()
        pct = np.empty_like(ETAS)
        lam = np.empty_like(ETAS)
        for j, eta in enumerate(ETAS):
            eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=wedge,
                             eta=float(eta), survival=True)
            eq.L0 = L0
            out = eq.solve(c, kappa)
            dL = out.L - L0
            pct[j] = 100.0 * dL[region].sum() / L0reg
            lam[j] = eq.labor_share(out.L)
            rows.append(dict(field=name, eta=float(eta),
                             pct_dL_region=pct[j], labor_share=lam[j],
                             converged=out.converged))
        curves[name] = pct
        flip = _flip_eta(ETAS, pct)
        lines += [
            f"[{name}] tech xi_K={np.degrees(tech.xi_K):.0f}deg chi_K={tech.chi_K:.2f} "
            f"z_K={tech.z_K:.2f} A_K={tech.A_K:.2f}",
            f"   mean operated share (a>0.05): {amean:.3f}",
            f"   Pi/c at margin (a~0.5): {mm:.3f}   deep core (a>0.9): {cc:.3f}",
            f"   employment in automated region at eta=1: {pct[ETAS==1][0]:+.2f}%",
            f"   flip eta* (region employment crosses 0): "
            + (f"{flip:.2f}" if np.isfinite(flip) else "none in [0.25,12]"),
            "",
        ]

    pd.DataFrame(rows).to_csv(RESULTS / "demand_channel_eta.csv", index=False)
    _figure(ETAS, curves, RESULTS / "demand_channel_eta.png")
    (RESULTS / "demand_channel_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS / 'demand_channel_eta.csv'}")
    print(f"wrote {RESULTS / 'demand_channel_eta.png'}")


def _figure(etas, curves, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhspan(-100, 100, xmin=0, xmax=0, color="none")  # keep ylim sane below
    ax.axvspan(0.5, 3.0, color="0.92", label="plausible demand elasticity")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    styles = {"cognitive": ("#1f3b6e", "o", "cognitive (north, broad)"),
              "manual":    ("#8c2d04", "s", "manual (west, narrow)")}
    for name, pct in curves.items():
        col, mk, lab = styles[name]
        ax.plot(etas, pct, color=col, marker=mk, ms=4, lw=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"demand elasticity $\eta$  (log scale)")
    ax.set_ylabel("employment change in the automated region (%)")
    ax.set_title("Automation releases or grows employment by demand elasticity")
    ax.annotate(r"$\eta=1$: cost-invariant" + "\n(reproduces script 09)",
                xy=(1.0, ax.get_ylim()[0]), xytext=(1.15, ax.get_ylim()[0] * 0.6),
                fontsize=8, color="0.3")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
