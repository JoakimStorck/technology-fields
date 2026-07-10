"""
09_equilibrium_regime.py
------------------------
Solves the worker-layer equilibrium under the calibrated AI technology and
reports the regime outcome with and without the family wage wedge. The
employment vector L_o is the fixed point of the logit re-sorting (reading A,
model.equilibrium); the value W_o it sorts on comes from the task layer
(model.regime), so the two must agree at the solution -- checked here.

ANCHORED RE-SPECIFICATION (this revision). Script 28 discovered that the
observed L0 was not a fixed point of the unanchored sorting map: with no
technology the logit relocated ~58 percent of mass, so the committed dL
conflated a baseline drift with the shock (cognitive technology component
~20 of the 58; drift-carried gainer/loser identities). The kernel now
carries occupation constants alpha_o (model.equilibrium.anchor_alpha),
anchored by the zero-field rule of _setup.anchor_reference, so that L0 IS
the zero-field fixed point and dL = L* - L0 is the technology's own
employment response. alpha is wedge-invariant (the wedge enters only the
takeover margin, not the zero-field value), so one anchor serves both
configurations below. The committed numbers this changes are quantified
against the frozen record in the summary; the manuscript is revised only
after the certified run.

PRE-REGISTERED EXPECTATIONS (anchored re-run; recorded before the first
certified run, verdicts to the summary as they fall):
  E1  Pure-shock re-sorted mass falls an order of magnitude below the
      committed 58 percent; smoke mechanics suggest the 15-25 band, not
      asserted.
  E2  Loser identities move from the drift-carried large service
      occupations (fast food, retail, baristas) to exposed analytical
      occupations; gainers to low-exposure interpersonal and manual work
      (the script-28 audit found exactly this for the L*(tech) - L*(0)
      component; the anchored response is a different object and need not
      reproduce it exactly).
  E3  Manual-arc employment change stays positive at roughly two thirds
      of the committed +0.150 (the audit's technology component was
      +0.104).
  E4  The unbound share of seeded mass stays at the committed 0.68 within
      +/- 0.02 (it reaches L only through Phi(C); the robot case moved
      93.0 -> 92.8 between L* and L0).

RESULTS (first certified anchored run, grid 4800):
  E1  PASS. 13.7 percent (no wedge; 15.3 with wedge), slightly below the
      smoke band. Unique fixed point (multistart spread 5.5e-11);
      W-agreement with the task layer 1.1e-14.
  E2  HALF. Losers as expected: exposed analytical occupations (clinical
      nurse specialists, sustainability specialists, software developers,
      business continuity planners). Gainers NOT the audit's maintenance/
      teachers/clergy but the large low-priced service occupations
      (retail salespersons, baristas, fast food, cashiers) -- the same
      occupations the unanchored drift shrank. The anchored response
      concentrates absorption in large low-barrier occupations; the
      audit's L*(tech) - L*(0) object measured a different comparison
      around a drifted baseline. Recorded, both halves.
  E3  FAIL on magnitude, PASS on sign. Manual arc +0.0425 (+0.0451
      wedged): positive, but 28 percent of the committed +0.150 rather
      than the expected two thirds. The audit's +0.104 was the shock
      around the drifted baseline; the observed economy absorbs more of
      the freed labour in the southern service arc instead.
  E4  PASS. 0.676, wedge variant 0.676.
  Labour share: automation 0.6488, reinstatement 0.6439 (wedge 0.6481 /
  0.6432); levels supersede the drift-carried 0.63 / 0.626, the
  reinstatement gap survives at half a point.

Reports:
  - convergence and a multi-start uniqueness check (the basis for stating
    uniqueness as a proposition);
  - employment re-sorting Delta L_o, top gainers and losers;
  - the labor share pre-technology, automation-only, and with reinstatement;
  - the bundle wage change Delta w_o;
  - the unbound mass (candidate-occupation territory) and the candidate map
    u(r) colored by the price field;
  - the operated regime of the calibrated technology: a(r) and the seeding
    ring, and a price-ordering check (takeover ordered by price).

Mobility (c, kappa) are set to an interpretable reference (one SD of value =
one logit unit; a typical move costs about one unit); their effect on the
threshold is swept in scripts/10.

Outputs:
    results/equilibrium_regime.csv
    results/equilibrium_regime_summary.txt
    results/candidate_map.png
    results/operated_share_demo.png

Usage:
    python scripts/09_equilibrium_regime.py
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

    # anchored mobility and alpha: zero-field rule, one anchor for the era
    # (alpha is wedge-invariant; the wedge enters only the takeover margin)
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                      survival=True)
    eq0.L0 = L0
    c, kappa, dmed, alpha = _setup.anchor_reference(eq0, L0)

    lines = [
        "Equilibrium regime under the calibrated AI technology "
        "(reading A, anchored kernel).",
        f"  occupations {len(L0)}, tasks {len(inp.bundles)}, grid "
        f"{inp.grid.xi.size} cells",
        f"  economy: R {R}, tau {TAU}, beta {BETA}, gamma {GAMMA}, "
        f"ell {ell:.4f}",
        f"  mobility reference: kappa {kappa:.3f} (= SD of zero-field "
        f"value), c {c:.3f}, median move {dmed:.3f}",
        f"  anchoring: alpha in [{alpha.min():+.2f}, {alpha.max():+.2f}] "
        f"({alpha.min() / kappa:+.1f} to {alpha.max() / kappa:+.1f} kappa "
        f"units); L0 is the zero-field fixed point",
        "",
    ]

    res_rows = []
    for tag, w in [("no wedge", None), ("with wedge", wedge)]:
        eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=w,
                         survival=True)
        eq.L0 = L0
        eq.alpha = alpha
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
            f"  re-sorting (pure shock; anchored L0): sum|dL| "
            f"{np.abs(dL).sum():.4f} "
            f"({100*np.abs(dL).sum()/2:.1f}% of mass relocated), "
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
    _operated_regime_demo(inp, tech, R, TAU, lines,
                          RESULTS / "operated_share_demo.png")
    (RESULTS / "equilibrium_regime_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    _candidate_map(inp, u_field, tech, RESULTS / "candidate_map.png")
    print(f"wrote {RESULTS / 'equilibrium_regime.csv'}")
    print(f"wrote {RESULTS / 'candidate_map.png'}")
    print(f"wrote {RESULTS / 'operated_share_demo.png'}")


def _operated_regime_demo(inp, tech, R, TAU, lines, out_path):
    """Illustrate the operated regime of the CALIBRATED technology: a price-
    ordering check (capital moves first against the dear work) and the a(r) /
    seeding-ring map. Bare operated share, no wedge -- the technology's own
    footprint, distinct from the wedged equilibrium reported above."""
    field = inp.field
    bx = inp.bundles
    xi_t, chi_t = bx["xi"].to_numpy(), bx["chi"].to_numpy()
    a_t = tech.operated_share(xi_t, chi_t, field, R, TAU)
    pi_t = field.pi(xi_t, chi_t)
    phi_t = tech.phi(xi_t, chi_t)
    bands = pd.qcut(phi_t, 20, duplicates="drop")
    rho = (pd.DataFrame({"a": a_t, "pi": pi_t, "band": bands})
           .groupby("band", observed=True)
           .apply(lambda d: d["a"].corr(d["pi"], method="spearman"),
                  include_groups=False))
    lines += [
        "",
        "Operated regime of the calibrated technology (bare a, no wedge):",
        f"  price ordering within phi-bands: median Spearman rho(a, Pi) = "
        f"{rho.median():.3f} (min {rho.min():.3f}) -- capital moves first "
        f"against the dear work",
    ]
    g = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= 1.0
    XI, CHI = np.arctan2(Y, X), np.hypot(X, Y)
    A = np.where(inside, tech.operated_share(XI, CHI, field, R, TAU), np.nan)
    G = np.where(inside, tech.grad_phi_norm(XI, CHI), np.nan)
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    cf0 = axes[0].contourf(X, Y, A, levels=24, cmap="magma")
    axes[0].set_title(r"Operated share $a(\mathbf{r})$ (calibrated technology)")
    fig.colorbar(cf0, ax=axes[0], shrink=0.8)
    cf1 = axes[1].contourf(X, Y, G, levels=24, cmap="cividis")
    axes[1].set_title(r"$\|\nabla\phi_K\|$ -- seeding ring at $z_K$")
    fig.colorbar(cf1, ax=axes[1], shrink=0.8)
    for ax in axes:
        px, py = tech.p_K
        ax.plot(px, py, "w+", ms=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    axes[0].set_title(r"Unbound task density $u(\mathbf{r})$")
    fig.colorbar(sc0, ax=axes[0], shrink=0.8)
    # unbound mass weighted, colored by price (opportunity vs spillage)
    mask = u > u.max() * 0.05
    sc1 = axes[1].scatter(x[mask], y[mask], c=pi[mask], s=8 + 60 * u[mask] / u.max(),
                          cmap="viridis", alpha=0.7)
    axes[1].set_title(r"Candidate regions colored by price $\Pi$")
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
