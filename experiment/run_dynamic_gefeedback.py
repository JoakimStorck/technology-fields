"""
run_dynamic_gefeedback.py
-------------------------
The dynamic transition WITH the AR-style price feedback wired into the time loop
-- the genuine GE-dynamic close. At each step the price responds to the current
adoption and labour allocation,

    Delta ln Pi_t(r) = (1/sigma) [ ln(1 - a_t(r)) - ln( density_t(r) / density_0(r) ) ],

applied to the estimated baseline (level from data, change from the shock), with
a one-step lag to avoid within-step circularity and a mild relaxation. The
updated price feeds the adoption gate, the place value, and births, so the wage
adjustment shapes the whole trajectory -- not just the end state.

We run the transition with the feedback OFF (the existing fixed-price dynamic
model) and ON, and compare the U/B trajectories, the price path, and the
end-state labour share against the static GE fixed point (0.73, Delta ln Pi
level -0.26).

Usage:
    python experiment/run_dynamic_gefeedback.py
"""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_rd = importlib.util.spec_from_file_location("run_dynamic", REPO / "experiment" / "run_dynamic.py")
rd = importlib.util.module_from_spec(_rd)
_rd.loader.exec_module(rd)
from model.equilibrium import Equilibrium
from model.regime import regime, _cell_index
from model.technology import Technology

_setup = rd._setup
R, TAU, BETA, GAMMA = rd.R, rd.TAU, rd.BETA, rd.GAMMA
SIGMA, RELAX = 3.0, 0.3


class AdjustedField:
    def __init__(self, base, dlnPi, grid):
        self._base, self._dlnPi, self._grid = base, dlnPi, grid

    def pi(self, xi, chi):
        c = _cell_index(self._grid, np.asarray(xi, float), np.asarray(chi, float))
        return self._base.pi(xi, chi) * np.exp(self._dlnPi[c])

    def __getattr__(self, name):
        return getattr(self._base, name)


def run(price_feedback, T_max=20.0, dt=0.2, theta_L=3.0, lam_b=1.0, rho=0.5,
        theta_abs=3.0, lam_over=1.0, match_beta=3.0, T_shock=5.0):
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    A_final = tech.A_K
    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq.L0 = L0
    unit = Technology(xi_K=tech.xi_K, chi_K=tech.chi_K, z_K=tech.z_K, A_K=1.0, s_K=1.0)
    g0_grid = unit.phi(inp.grid.xi, inp.grid.chi)
    g0_task = unit.phi(eq.b_xi, eq.b_chi)
    rd.set_AK(eq, 0.0, g0_grid, g0_task)
    _, _, W0 = eq.density_and_value(L0)
    kappa = float(np.std(W0))
    c = kappa / float(np.median(eq.d[eq.d > 0]))
    dyn = rd.Dyn(eq, inp, L0, ell, rho, lam_over=lam_over)

    pi_cell_0 = eq.pi_cell.copy()
    pi_task_0 = eq.pi_task.copy()
    nL0 = dyn.density().copy()              # baseline density (B=0, L=L0, a=0)
    okc = nL0 > 0
    dlnPi = np.zeros(dyn.ncell)

    k_shock = 5.88 / T_shock
    A_of = lambda t: A_final / (1.0 + np.exp(-k_shock * (t - 0.5 * T_shock)))
    ts = np.arange(0.0, T_max + dt, dt)
    GammaD_prev = None
    rec = {k: [] for k in ("t", "A_K", "U_tot", "B_tot", "dlnPi_mean")}

    for it, t in enumerate(ts):
        A_K = A_of(t)
        a_grid = rd.set_AK(eq, A_K, g0_grid, g0_task)      # uses current (lagged) prices
        n = dyn.density()
        GammaD = float(np.sum(dyn.L[:dyn.n0] * eq.D_o))
        dGamma = 0.0 if GammaD_prev is None else max((GammaD - GammaD_prev) / dt, 0.0)
        GammaD_prev = GammaD
        sdot = GAMMA * dGamma * eq.g_hat * (1.0 - a_grid)
        dyn.U = dyn.U + dt * sdot
        M = dyn.original + dyn.reinst
        Wb = dyn.FIT ** match_beta
        Wsum = Wb.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            claim = np.where(Wsum > 0, (dyn.U * dyn.area) / Wsum, 0.0)
        t_o = Wb * claim[None, :]
        des = t_o.sum(1)
        capo = (dt / theta_abs) * M
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(des > 1e-15, np.minimum(1.0, capo / des), 0.0)
        dyn.reinst[:] = dyn.reinst + des * f
        absorbed = (t_o * f[:, None]).sum(0)
        dyn.U = dyn.U - absorbed / dyn.area
        dyn.B = dyn.B + absorbed / dyn.area
        W = dyn.values(dyn.density())
        Tgt = rd.softmax_target(dyn, W, c, kappa)
        dyn.L = dyn.L + (dt / theta_L) * (Tgt - dyn.L)

        if price_feedback:                                 # update Pi for the NEXT step
            dens = dyn.density()
            aclip = np.clip(a_grid, 0.0, 0.999)
            tgt = np.zeros(dyn.ncell)
            ratio = (dens[okc] + 1e-9) / (nL0[okc] + 1e-9)
            tgt[okc] = (1.0 / SIGMA) * (np.log(1.0 - aclip[okc]) - np.log(ratio))
            dlnPi = (1.0 - RELAX) * dlnPi + RELAX * tgt
            eq.pi_cell = pi_cell_0 * np.exp(dlnPi)
            eq.pi_task = pi_task_0 * np.exp(dlnPi[eq.cell_of])
            dyn.Pi = eq.pi_cell

        rec["t"].append(t); rec["A_K"].append(A_K)
        rec["U_tot"].append(float(np.sum(dyn.U * dyn.area)))
        rec["B_tot"].append(float(np.sum(dyn.B * dyn.area)))
        rec["dlnPi_mean"].append(float(dlnPi[okc].mean()))

    # end-state labour share via regime with the evolved price field
    adj = AdjustedField(inp.field, dlnPi, inp.grid)
    reg = regime(replace(inp, field=adj), tech, L0, R, TAU, GAMMA, ell, BETA,
                 wedge=None, survival=True)
    rec = {k: np.array(v) for k, v in rec.items()}
    return rec, reg["labor_share"], dlnPi[okc].mean()


def main():
    print("Dynamic transition: fixed price vs AR price feedback in the loop.\n")
    rec0, ls0, _ = run(price_feedback=False)
    rec1, ls1, lvl1 = run(price_feedback=True)

    def at(rec, tt):
        i = int(np.argmin(np.abs(rec["t"] - tt)))
        return rec["U_tot"][i], rec["B_tot"][i], rec["dlnPi_mean"][i]

    print(f"  {'t':>4}  {'A_K':>5} | {'U_tot fix':>9} {'U_tot fb':>9} | "
          f"{'B_tot fix':>9} {'B_tot fb':>9} | {'dlnPi_mean':>10}")
    for tt in (1, 2, 3, 5, 8, 12, 20):
        u0, b0, _ = at(rec0, tt); u1, b1, lv = at(rec1, tt)
        ak = rec0["A_K"][int(np.argmin(np.abs(rec0["t"] - tt)))]
        print(f"  {tt:>4}  {ak:>5.2f} | {u0:>9.4f} {u1:>9.4f} | {b0:>9.4f} {b1:>9.4f} | {lv:>+10.4f}")

    print(f"\n  U_tot peak: fixed {rec0['U_tot'].max():.4f} (t={rec0['t'][rec0['U_tot'].argmax()]:.1f}) "
          f"-> feedback {rec1['U_tot'].max():.4f} (t={rec1['t'][rec1['U_tot'].argmax()]:.1f})")
    print(f"  end B_tot: fixed {rec0['B_tot'][-1]:.4f} -> feedback {rec1['B_tot'][-1]:.4f}")
    print(f"\n  === consistency with the static GE fixed point ===")
    print(f"  dynamic end-state price level  Delta ln Pi = {lvl1:+.4f}   (static GE fixed point: -0.26)")
    print(f"  dynamic end-state labour share = {ls1:.4f}   (fixed-Pi {ls0:.4f}; static GE 0.73)")


if __name__ == "__main__":
    main()
