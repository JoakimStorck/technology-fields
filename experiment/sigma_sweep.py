"""
sigma_sweep.py
--------------
Sensitivity of the GE price-feedback close to the assignment/task elasticity
sigma. sigma enters ONLY the feedback law

    Delta ln Pi(r) = (1/sigma) [ ln(1 - a(r)) - ln( n_L(r) / n_L^0(r) ) ],

not the underlying equilibrium (which is set by R, tau, beta, gamma), so the
sweep isolates the price-feedback coefficient 1/sigma.

Two sweeps over a sigma grid:
  (1) static GE fixed point (price_feedback_ge logic): the labour-share lift on
      the paper-consistent re-solved-allocation (out.L) basis, 0.626 -> X(sigma),
      plus D_o and the price level; the L0 basis is reported alongside.
  (2) dynamic GE feedback (run_dynamic_gefeedback): the unbound-mass (mismatch)
      hump damping and the end-state, across the same grid.

The point sigma = 0.5 is the Acemoglu-Restrepo (2026) anchor: there the feedback
coefficient 1/sigma = 2 equals their wage-law coefficient 1/lambda (lambda = 0.5),
so it is where this region-wise law matches their task elasticity. Note that AR
operate at lambda < 1 (tasks gross complements) whereas the calibrated model uses
sigma = 3 (tasks gross substitutes); the sweep brackets both regimes.

Usage: python experiment/sigma_sweep.py
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.equilibrium import Equilibrium
from model.regime import regime, _cell_index

_spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts/_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
TOL, MAXIT = 4e-3, 200
SIGMA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]
AR_ANCHOR = 0.5


class AdjustedField:
    def __init__(self, base, dlnPi, grid):
        self._base, self._dlnPi, self._grid = base, dlnPi, grid

    def pi(self, xi, chi):
        cells = _cell_index(self._grid, np.asarray(xi, float), np.asarray(chi, float))
        return self._base.pi(xi, chi) * np.exp(self._dlnPi[cells])

    def __getattr__(self, name):
        return getattr(self._base, name)


def static_fixpoint(sigma, inp, L0, tech, ell, grid, c, kappa, nL0, occ_cell, cl, rw, bw, area):
    """Iterate the ar_level feedback to the GE fixed point at this sigma.

    The feedback coefficient is 1/sigma, so strong feedback (small sigma) needs
    gentler damping to avoid oscillation; we scale the relaxation with sigma.
    """
    damp = 0.5 * min(1.0, sigma / 2.0)
    d = np.zeros(grid.xi.size)
    converged, iters = False, MAXIT
    with np.errstate(over="ignore"):
        for it in range(MAXIT):
            adj = AdjustedField(inp.field, d, grid)
            inp_k = replace(inp, field=adj)
            eqk = Equilibrium(inp_k, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
            eqk.L0 = L0
            Lk = eqk.solve(c, kappa).L
            ak = np.clip(eqk.a_grid, 0.0, 0.999)
            nLk = np.bincount(cl, weights=Lk[rw] * bw, minlength=grid.xi.size) / area
            tgt = np.zeros(grid.xi.size)
            ratio = (nLk[occ_cell] + 1e-12) / (nL0[occ_cell] + 1e-12)
            tgt[occ_cell] = (1.0 / sigma) * (np.log(1.0 - ak[occ_cell]) - np.log(ratio))
            d_new = (1.0 - damp) * d + damp * tgt
            change = np.max(np.abs((d_new - d)[occ_cell]))
            d = d_new
            if change < TOL:
                converged, iters = True, it + 1
                break
    # converged diagnostics on both bases
    inp_star = replace(inp, field=AdjustedField(inp.field, d, grid))
    reg_L0 = regime(inp_star, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq_star = Equilibrium(inp_star, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq_star.L0 = L0
    L_star = eq_star.solve(c, kappa).L
    reg_out = regime(inp_star, tech, L_star, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    return {
        "sigma": sigma, "converged": converged, "iters": iters,
        "level": float(d[occ_cell].mean()),
        "ls_L0": reg_L0["labor_share"], "ls_out": reg_out["labor_share"],
        "Do": float(reg_out["D_o"].mean()),
    }


def main():
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    grid = inp.grid
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq0.L0 = L0
    _, _, W0 = eq0.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0, eq0.d)
    rw, bw, cl, area = eq0.row_of, eq0.b_w, eq0.cell_of, eq0.area
    nL0 = np.bincount(cl, weights=L0[rw] * bw, minlength=grid.xi.size) / area
    occ_cell = nL0 > 0

    # fixed-price baselines (sigma-independent)
    ls_fix_L0 = regime(inp, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)["labor_share"]
    L0_out = eq0.solve(c, kappa).L
    ls_fix_out = regime(inp, tech, L0_out, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)["labor_share"]

    emit("sigma sweep of the GE price feedback (R 18, tau 0.08, beta 0.5, gamma 0.5)")
    emit(f"feedback coefficient = 1/sigma; AR(2026) anchor at sigma={AR_ANCHOR} (1/sigma=2=1/lambda)")
    emit(f"fixed-price baseline: labour share  L0 {ls_fix_L0:.4f} | out.L {ls_fix_out:.4f};  "
         f"D_o(out.L) {regime(inp,tech,L0_out,R,TAU,GAMMA,ell,BETA,wedge=None,survival=True)['D_o'].mean():.4f}")
    emit("")
    emit("=== (1) static GE fixed point ===")
    emit(f"  {'sigma':>6} {'1/sigma':>8} {'conv':>5} {'it':>3} "
         f"{'dlnPi':>8} {'LS(out.L)':>10} {'lift':>7} {'LS(L0)':>8} {'D_o':>8}")
    static = []
    for s in SIGMA_GRID:
        r = static_fixpoint(s, inp, L0, tech, ell, grid, c, kappa, nL0, occ_cell, cl, rw, bw, area)
        static.append(r)
        tag = "  <- AR" if abs(s - AR_ANCHOR) < 1e-9 else ("  <- baseline" if abs(s - 3.0) < 1e-9 else "")
        emit(f"  {s:>6.2f} {1.0/s:>8.3f} {str(r['converged']):>5} {r['iters']:>3} "
             f"{r['level']:>+8.4f} {r['ls_out']:>10.4f} {r['ls_out']-ls_fix_out:>+7.4f} "
             f"{r['ls_L0']:>8.4f} {r['Do']:>8.4f}{tag}")

    # === (2) dynamic feedback ===
    emit("")
    emit("=== (2) dynamic GE feedback (unbound-mass hump and end-state) ===")
    _rd = importlib.util.spec_from_file_location(
        "run_dynamic_gefeedback", REPO / "experiment" / "run_dynamic_gefeedback.py")
    rd = importlib.util.module_from_spec(_rd)
    _rd.loader.exec_module(rd)

    rec_off, ls_off, _ = rd.run(price_feedback=False)   # sigma-independent baseline
    u_peak_off = float(rec_off["U_tot"].max())
    emit(f"  feedback OFF: U_tot peak {u_peak_off:.4f}, end labour share {ls_off:.4f}")
    emit(f"  {'sigma':>6} {'1/sigma':>8} {'U_peak_fb':>10} {'damping':>8} "
         f"{'LS_end':>8} {'level_end':>10}")
    dyn = []
    for s in SIGMA_GRID:
        rd.SIGMA = s                                    # feedback law uses module global
        rec_on, ls_on, lvl_on = rd.run(price_feedback=True)
        u_peak_on = float(rec_on["U_tot"].max())
        damp = 1.0 - u_peak_on / u_peak_off if u_peak_off > 0 else float("nan")
        dyn.append({"sigma": s, "u_peak": u_peak_on, "damp": damp, "ls_end": ls_on, "lvl_end": lvl_on})
        tag = "  <- AR" if abs(s - AR_ANCHOR) < 1e-9 else ("  <- baseline" if abs(s - 3.0) < 1e-9 else "")
        emit(f"  {s:>6.2f} {1.0/s:>8.3f} {u_peak_on:>10.4f} {100*damp:>7.1f}% "
             f"{ls_on:>8.4f} {lvl_on:>+10.4f}{tag}")

    (REPO / "experiment" / "sigma_sweep_summary.txt").write_text("\n".join(lines) + "\n")

    # optional figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sig = np.array(SIGMA_GRID)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].axhline(ls_fix_out, ls="--", c="0.6", lw=1, label=f"fixed price {ls_fix_out:.2f}")
        ax[0].plot(sig, [r["ls_out"] for r in static], "o-", c="C0")
        ax[0].axvline(AR_ANCHOR, ls=":", c="C3", lw=1)
        ax[0].annotate("AR anchor\n$1/\\sigma=1/\\lambda=2$", (AR_ANCHOR, ls_fix_out),
                       textcoords="offset points", xytext=(8, 8), color="C3", fontsize=8)
        ax[0].set_xlabel("$\\sigma$ (assignment/task elasticity)")
        ax[0].set_ylabel("GE labour share (out.L basis)")
        ax[0].set_title("Static fixed-point lift vs $\\sigma$")
        ax[0].legend(fontsize=8)
        ax[1].plot(sig, [100 * d["damp"] for d in dyn], "s-", c="C1")
        ax[1].axvline(AR_ANCHOR, ls=":", c="C3", lw=1)
        ax[1].set_xlabel("$\\sigma$ (assignment/task elasticity)")
        ax[1].set_ylabel("unbound-mass hump damping (%)")
        ax[1].set_title("Dynamic mismatch damping vs $\\sigma$")
        fig.tight_layout()
        fig.savefig(REPO / "experiment" / "sigma_sweep.png", dpi=120)
        emit("\nwrote experiment/sigma_sweep.png and experiment/sigma_sweep_summary.txt")
    except Exception as e:  # pragma: no cover
        emit(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
