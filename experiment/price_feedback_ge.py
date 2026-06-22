"""
price_feedback_ge.py
--------------------
Full general-equilibrium close of the AR-style price feedback. Iterate to a
self-consistent price field:

    Pi* = Pi_0 * exp( Delta ln Pi(Pi*) ),
    Delta ln Pi(r) = (1/sigma) [ ln(1 - a(Pi)) - ln( nL(Pi) / nL0 ) ],

where a(Pi) is the operated share at the current price (the takeover gate) and
nL(Pi) the re-sorted labour density. The loop is a negative feedback -- a lower
price closes the gate (a falls), which raises the price back -- so it contracts
to an interior fixed point between the one-pass extremes (labour share 0.33
scarcity / 0.83 one-pass ar_level, against 0.58 fixed-price).

Usage:
    python experiment/price_feedback_ge.py
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
SIGMA = 3.0
DAMP, TOL, MAXIT = 0.5, 4e-3, 60


class AdjustedField:
    def __init__(self, base, dlnPi, grid):
        self._base, self._dlnPi, self._grid = base, dlnPi, grid

    def pi(self, xi, chi):
        cells = _cell_index(self._grid, np.asarray(xi, float), np.asarray(chi, float))
        return self._base.pi(xi, chi) * np.exp(self._dlnPi[cells])

    def __getattr__(self, name):
        return getattr(self._base, name)


def main():
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
    reg0 = regime(inp, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    print(f"fixed-Pi baseline: labour share {reg0['labor_share']:.4f}, "
          f"D_o mean {reg0['D_o'].mean():.4e}\n")
    print(f"GE fixed point (sigma {SIGMA}, damp {DAMP}):")
    print(f"  {'it':>3} {'max|d(dlnPi)|':>14} {'mean dlnPi':>11} {'labour share':>13} {'D_o mean':>10}")

    d = np.zeros(grid.xi.size)
    for it in range(MAXIT):
        adj = AdjustedField(inp.field, d, grid)
        inp_k = replace(inp, field=adj)
        eqk = Equilibrium(inp_k, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
        eqk.L0 = L0
        Lk = eqk.solve(c, kappa).L
        ak = np.clip(eqk.a_grid, 0.0, 0.999)
        nLk = np.bincount(cl, weights=Lk[rw] * bw, minlength=grid.xi.size) / area
        d_target = np.zeros(grid.xi.size)
        ratio = (nLk[occ_cell] + 1e-12) / (nL0[occ_cell] + 1e-12)
        d_target[occ_cell] = (1.0 / SIGMA) * (np.log(1.0 - ak[occ_cell]) - np.log(ratio))
        d_new = (1.0 - DAMP) * d + DAMP * d_target
        change = np.max(np.abs((d_new - d)[occ_cell]))
        d = d_new
        regk = regime(inp_k, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
        if it < 6 or it % 5 == 0 or change < TOL:
            print(f"  {it:>3} {change:>14.5f} {d[occ_cell].mean():>+11.4f} "
                  f"{regk['labor_share']:>13.4f} {regk['D_o'].mean():>10.4e}")
        if change < TOL:
            break

    # converged diagnostics
    adj = AdjustedField(inp.field, d, grid)
    inp_star = replace(inp, field=adj)
    regk = regime(inp_star, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True) \
        if False else regime(inp_star, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    lnP0 = np.log(inp.field.pi(grid.xi, grid.chi))
    xi, chi = grid.xi, grid.chi
    north = (np.cos(xi) > 0.5) & (chi > 0.4) & occ_cell
    south = (np.cos(xi) < -0.3) & (chi > 0.3) & occ_cell
    dw0, dwk = np.asarray(reg0['dW_bundle']), np.asarray(regk['dW_bundle'])
    print("\n=== converged fixed point vs fixed-Pi baseline ===")
    print(f"  Delta ln Pi*: mean {d[occ_cell].mean():+.4f} (level), sd {d[occ_cell].std():.4f}, "
          f"north-south {np.median(d[north]) - np.median(d[south]):+.4f}")
    print(f"  corr(Delta ln Pi*, operated share a) = {spearmanr(d[occ_cell], np.clip(eqk.a_grid,0,0.999)[occ_cell])[0]:+.3f}")
    print(f"  labour share   {reg0['labor_share']:.4f} -> {regk['labor_share']:.4f}")
    print(f"  D_o mean       {reg0['D_o'].mean():.4e} -> {regk['D_o'].mean():.4e}  "
          f"(corr {spearmanr(reg0['D_o'], regk['D_o'])[0]:.3f})")
    print(f"  dW mean        {dw0.mean():.2f} -> {dwk.mean():.2f}  "
          f"(share<0 {100*np.mean(dw0<0):.0f}% -> {100*np.mean(dwk<0):.0f}%)")
    print("\n  reference one-pass extremes: scarcity 0.333 | ar_level 0.827 | fixed 0.582")

    # paper-consistent (re-solved allocation, out.L) basis: the labour-share
    # diagnostics above hold the allocation at L0 to isolate the price channel;
    # the paper reports the share at the re-solved equilibrium allocation, so we
    # also report the pair on that basis (matches script 09's 0.626 at fixed Pi).
    L0_out = eq0.solve(c, kappa).L
    ls_fix_out = regime(inp, tech, L0_out, R, TAU, GAMMA, ell, BETA,
                        wedge=None, survival=True)['labor_share']
    eq_star = Equilibrium(inp_star, tech, R, TAU, GAMMA, ell, BETA,
                          wedge=None, survival=True)
    eq_star.L0 = L0
    L_star = eq_star.solve(c, kappa).L
    ls_ge_out = regime(inp_star, tech, L_star, R, TAU, GAMMA, ell, BETA,
                       wedge=None, survival=True)['labor_share']
    print(f"\n  labour share, re-solved allocation (out.L, paper basis): "
          f"{ls_fix_out:.4f} -> {ls_ge_out:.4f}")
    print("    (price and allocation both equilibrate; this is the pair the "
          "static paper cites)")


if __name__ == "__main__":
    main()
