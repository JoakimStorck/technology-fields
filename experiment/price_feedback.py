"""
price_feedback.py
-----------------
Post-shock price feedback -- the AR-style general-equilibrium close. Level from
the estimated price field (the empirical anchor); CHANGE from the shock. Two
closures, to see how much the sign convention matters:

  "scarcity"  : Delta ln Pi = -(1/sigma) Delta ln n_human, mean-zeroed.
                Treats automation as a SUPPLY shock to human output -- less human
                work done -> scarcer -> dearer. Redistributive (level fixed).
                This is destabilising: dearer human work opens the takeover gate.

  "ar_level"  : Delta ln Pi = (1/sigma) [ ln(1 - a) - ln(L_post / L_0) ].
                AR's wage law w ~ (tasks-per-worker)^(1/sigma): automation is a
                DEMAND shock to labour (the task leaves the human set, so the
                wage falls), and displaced labour crowding the rest lowers it
                further. NOT mean-zeroed -- the price LEVEL falls, AR's
                stabilising force.

For each closure we apply Delta ln Pi to the estimated baseline and re-run the
shock, comparing displacement, labour share, and incidence against fixed Pi.

Usage:
    python experiment/price_feedback.py
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
    Lpost = eq0.solve(c, kappa).L
    reg0 = regime(inp, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)

    rw, bw, cl = eq0.row_of, eq0.b_w, eq0.cell_of
    area = eq0.area
    nL0 = np.bincount(cl, weights=L0[rw] * bw, minlength=grid.xi.size) / area
    nLpost = np.bincount(cl, weights=Lpost[rw] * bw, minlength=grid.xi.size) / area
    nHpost = np.bincount(cl, weights=Lpost[rw] * bw * (1.0 - eq0.a_task),
                         minlength=grid.xi.size) / area
    a = np.clip(eq0.a_grid, 0.0, 0.999)
    occ_cell = nL0 > 0
    lnP0 = np.log(inp.field.pi(grid.xi, grid.chi))
    xi, chi = grid.xi, grid.chi
    north = (np.cos(xi) > 0.5) & (chi > 0.4) & occ_cell
    south = (np.cos(xi) < -0.3) & (chi > 0.3) & occ_cell

    print("Post-shock price feedback. sigma 3.0; mobility reference held fixed.")
    print(f"fixed-Pi baseline: labour share {reg0['labor_share']:.4f}, "
          f"D_o mean {reg0['D_o'].mean():.4e}, dW mean {np.mean(reg0['dW_bundle']):.2f}\n")

    for closure in ("scarcity", "ar_level"):
        d = np.zeros(grid.xi.size)
        if closure == "scarcity":
            d[occ_cell] = -(1.0 / SIGMA) * (np.log(nHpost[occ_cell] + 1e-12)
                                            - np.log(nL0[occ_cell] + 1e-12))
            d[occ_cell] -= d[occ_cell].mean()                       # redistributive
        else:  # ar_level: wage ~ (tasks-per-worker)^(1/sigma), level free to fall
            ratio = (nLpost[occ_cell] + 1e-12) / (nL0[occ_cell] + 1e-12)
            d[occ_cell] = (1.0 / SIGMA) * (np.log(1.0 - a[occ_cell]) - np.log(ratio))

        adj = AdjustedField(inp.field, d, grid)
        inp_post = replace(inp, field=adj)
        reg1 = regime(inp_post, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
        eq1 = Equilibrium(inp_post, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
        eq1.L0 = L0
        L1 = eq1.solve(c, kappa).L
        dw0, dw1 = np.asarray(reg0['dW_bundle']), np.asarray(reg1['dW_bundle'])
        print(f"=== closure: {closure} ===")
        print(f"  Delta ln Pi: mean {d[occ_cell].mean():+.4f} (level), sd {d[occ_cell].std():.4f}, "
              f"north-south {np.median(d[north]) - np.median(d[south]):+.4f}")
        print(f"  corr(Delta ln Pi, operated share a) = {spearmanr(d[occ_cell], a[occ_cell])[0]:+.3f}")
        print(f"  labour share   {reg0['labor_share']:.4f} -> {reg1['labor_share']:.4f}")
        print(f"  D_o mean       {reg0['D_o'].mean():.4e} -> {reg1['D_o'].mean():.4e}  "
              f"(corr {spearmanr(reg0['D_o'], reg1['D_o'])[0]:.3f})")
        print(f"  dW mean        {dw0.mean():.2f} -> {dw1.mean():.2f}  "
              f"(share<0 {100*np.mean(dw0<0):.0f}% -> {100*np.mean(dw1<0):.0f}%)")
        print(f"  re-sorted L: sum|dL| vs fixed {np.abs(L1 - Lpost).sum():.4f}\n")


if __name__ == "__main__":
    main()
