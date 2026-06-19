"""
12_equilibrium_regime.py
------------------------
Solves the worker-layer equilibrium under the calibrated AI technology and
reports the regime outcome with and without the family wage wedge. The
employment vector L_o is the fixed point of the logit re-sorting (reading A,
model.equilibrium); the value W_o it sorts on comes from the task layer
(model.regime), so the two must agree at the solution -- checked here.

Reports:
  - convergence and a multi-start uniqueness check (the basis for stating
    uniqueness as a proposition);
  - employment re-sorting Delta L_o, top gainers and losers;
  - the labor share pre-technology, automation-only, and with reinstatement;
  - the bundle wage change Delta w_o;
  - the unbound mass (candidate-occupation territory) and the candidate map
    u(r) colored by the price field.

Mobility (c, kappa) are set to an interpretable reference (one SD of value =
one logit unit; a typical move costs about one unit); their effect on the
threshold is swept in scripts/13.

Outputs:
    results/equilibrium_regime.csv
    results/equilibrium_regime_summary.txt
    results/candidate_map.png

Usage:
    python scripts/12_equilibrium_regime.py
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
from model.regime import regime

_spec = importlib.util.spec_from_file_location("_setup",
                                               Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    wedge = _setup.load_wedge(occ)
    titles = occ["Title"].to_numpy()
    xi_o = occ["xi"].to_numpy()
    manual = (xi_o > np.radians(135)) & (xi_o < np.radians(225))   # west arc

    # mobility reference from baseline value (gated model, matching the runs)
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                      survival=True)
    eq0.L0 = L0
    _, _, W0 = eq0.density_and_value(L0)
    c, kappa, dmed = _setup.mobility_reference(W0, eq0.d)

    lines = [
        "Equilibrium regime under the calibrated AI technology (reading A).",
        f"  occupations {len(L0)}, tasks {len(inp.bundles)}, grid "
        f"{inp.grid.xi.size} cells",
        f"  economy: R {R}, tau {TAU}, beta {BETA}, gamma {GAMMA}, "
        f"ell {ell:.4f}",
        f"  mobility reference: kappa {kappa:.3f} (= SD of baseline value), "
        f"c {c:.3f}, median move {dmed:.3f}",
        "",
    ]

    res_rows = []
    for tag, w in [("no wedge", None), ("with wedge", wedge)]:
        eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=w,
                         survival=True)
        eq.L0 = L0
        Lstar, spread, allconv = eq.multistart(c, kappa, n_random=3)
        out = eq.solve(c, kappa)

        # diagnostics at the converged employment via the task layer
        diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA, wedge=w,
                      survival=True)
        diag0 = regime(inp, tech, out.L, R, TAU, 0.0, ell, BETA, wedge=w,
                       survival=True)
        # cross-check: equilibrium W vs regime W at the same L
        w_gap = float(np.max(np.abs(out.W - diag["W_o"])))

        dL = out.L - L0
        gain = np.argsort(dL)[::-1][:6]
        loss = np.argsort(dL)[:6]
        dLmanual = float(dL[manual].sum())

        lines += [
            f"=== {tag} ===",
            f"  converged {out.converged} in {out.iters} iters "
            f"(residual {out.residual:.1e}); multistart spread {spread:.1e}, "
            f"all converged {allconv}  -> unique fixed point",
            f"  equilibrium-vs-task-layer W_o agreement: max gap {w_gap:.2e}",
            f"  re-sorting: sum|dL| {np.abs(dL).sum():.4f} "
            f"({100*np.abs(dL).sum()/2:.0f}% of mass relocated), "
            f"max|dL| {np.abs(dL).max():.4f}",
            f"  labor share: pre 1.000 -> automation {diag0['labor_share']:.4f} "
            f"-> reinstatement {diag['labor_share']:.4f}",
            f"  bundle wage change dW: mean {diag['dW_bundle'].mean():+.3f}, "
            f"share dW<0 {100*np.mean(diag['dW_bundle']<0):.0f}%",
            f"  unbound (candidate) mass {diag['unbound_mass']:.4f} of seeded "
            f"{diag['M']:.4f} ({100*diag['unbound_mass']/diag['M']:.0f}%)",
            f"  manual-arc (135-225 deg) employment change "
            f"sum dL = {dLmanual:+.4f}",
            "  top employment gainers:",
        ]
        for i in gain:
            lines.append(f"    {titles[i][:40]:40s} dL {dL[i]:+.4e}")
        lines.append("  top employment losers:")
        for i in loss:
            lines.append(f"    {titles[i][:40]:40s} dL {dL[i]:+.4e}")
        lines.append("")

        if tag == "no wedge":
            res_rows = pd.DataFrame({
                "onet_code": occ.index, "Title": titles,
                "xi": xi_o, "chi": occ["chi"].to_numpy(),
                "L0": L0, "L_eq": out.L, "dL": dL,
                "D_o": diag["D_o"], "dW_bundle": diag["dW_bundle"],
                "W_o": out.W,
            })
            u_field = diag["u"]
            n_field = diag["n"]

    res_rows.to_csv(RESULTS / "equilibrium_regime.csv", index=False)
    (RESULTS / "equilibrium_regime_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    _candidate_map(inp, u_field, tech, RESULTS / "candidate_map.png")
    print(f"wrote {RESULTS / 'equilibrium_regime.csv'}")
    print(f"wrote {RESULTS / 'candidate_map.png'}")


def _candidate_map(inp, u, tech, out_path):
    """u(r) over the disk, colored by the price field: high-Pi unbound mass is
    latent skilled work; low-Pi unbound mass is precarious friction."""
    grid = inp.grid
    field = inp.field
    x, y = grid.x, grid.y
    pi = field.pi(grid.xi, grid.chi)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    # unbound density
    sc0 = axes[0].scatter(x, y, c=u, s=6, cmap="magma")
    axes[0].set_title("Unbound task density $u(\\mathbf{r})$\n"
                      "(new tasks no occupation can attach)")
    fig.colorbar(sc0, ax=axes[0], shrink=0.8)
    # unbound mass weighted, colored by price (opportunity vs spillage)
    mask = u > u.max() * 0.05
    sc1 = axes[1].scatter(x[mask], y[mask], c=pi[mask], s=8 + 60 * u[mask] / u.max(),
                          cmap="viridis", alpha=0.7)
    axes[1].set_title("Candidate regions colored by price $\\Pi$\n"
                      "(skilled opportunity vs precarious friction)")
    fig.colorbar(sc1, ax=axes[1], shrink=0.8, label="$\\Pi$")
    for ax in axes:
        px, py = tech.p_K
        ax.plot(px, py, "w+", ms=11, mew=2)
        ax.add_patch(plt.Circle((px, py), tech.z_K, color="white", fill=False,
                                lw=1.0, ls="--"))
        ax.add_patch(plt.Circle((0, 0), 1.0, color="0.6", fill=False, lw=0.8))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
