"""
probe_U_hump.py
---------------
The first genuine new stock: U(r,t), unbound (newly seeded, not-yet-bound) task
work, integrated in time over a technology-maturation trajectory A_K(t).

Per the whitepaper (sec. 5.3):
    dU/dt = sdot(r,t) - lambda_b * Phi(C(r,t)) * U(r,t)
    sdot  = gamma * [d/dt Gamma^D]_+ * g_hat(r) * (1 - a(r,t))     (survival gate)
Seeding follows the RATE of displacement (new tasks are born while the
technology matures); binding drains U at a capacity-dependent rate. Tasks are
permanent: what does not bind stays in U until capacity arrives.

The hump in U_tot(t) is the headline observable -- the gap between destruction
and creation, as a transient. It requires binding to LAG seeding, which it does
for a spatial reason: capacity Phi(C(r)) on the seeding ring is low until the
population re-sorts toward it. So the hump grows when the technology matures
faster than workers can move. We sweep mobility speed (theta_L) to show this.

Population re-sorts with moving origin on the standard task-layer value W (the
verified density_and_value); U is integrated alongside as the new stock. The
feedback of bound work into W is a refinement deliberately left out of this
minimal probe -- the hump lives in U regardless. eta = 1 (demand channel off),
survival on. Touches no committed code.
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
from model.equilibrium import Equilibrium
from model.technology import Technology

_spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts" / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
OUT = REPO / "scripts"


def set_AK(eq, A_K, g0_grid, g0_task):
    """Update only the A_K-dependent arrays (s_K = 1, no wedge, eta = 1).
    Leaves the fixed readiness e, ring g_hat, prices, and geometry untouched."""
    a_grid = 1.0 / (1.0 + np.exp(-(A_K * g0_grid - R / eq.pi_cell) / TAU))
    a_task = 1.0 / (1.0 + np.exp(-(A_K * g0_task - R / eq.pi_task) / TAU))
    eq.a_grid = a_grid
    eq.a_task = a_task
    eq.D_o = np.bincount(eq.row_of, weights=eq.b_w * a_task, minlength=eq.n_occ)
    eq.strip_w = eq.b_w * (1.0 - a_task) * eq.pi_task
    eq.strip_wD = eq.strip_w                      # D_task = 1 at eta = 1
    return a_grid


def softmax_P(eq, W, c, kappa):
    U = (W[None, :] - c * eq.d) / kappa
    U -= U.max(axis=1, keepdims=True)
    P = np.exp(U); P /= P.sum(axis=1, keepdims=True)
    return P


def run_trajectory(eq, L0, g0_grid, g0_task, c, kappa,
                   A_final, rho, t_mid, theta_L, lam_b,
                   T_max=70.0, dt=0.15):
    """Integrate (A_K, L, U) forward. Returns time series."""
    ts = np.arange(0.0, T_max + dt, dt)
    A_of = lambda t: A_final / (1.0 + np.exp(-rho * (t - t_mid)))

    L = L0.copy()
    U = np.zeros(eq.area.size)
    GammaD_prev = None
    rec = {k: [] for k in ("t", "A_K", "GammaD", "dGamma", "U_tot",
                            "bind", "u_inst", "share_unbound")}
    for t in ts:
        A_K = A_of(t)
        a_grid = set_AK(eq, A_K, g0_grid, g0_task)
        n, C, W = eq.density_and_value(L)
        GammaD = float(np.sum(L * eq.D_o))
        dGamma = 0.0 if GammaD_prev is None else max((GammaD - GammaD_prev) / dt, 0.0)
        GammaD_prev = GammaD

        surv = 1.0 - a_grid
        Phi = np.where(C > 0, C / (1.0 + C), 0.0)
        sdot = GAMMA * dGamma * eq.g_hat * surv          # seeding rate density
        drain = lam_b * Phi * U                          # binding rate density
        U = np.maximum(U + dt * (sdot - drain), 0.0)

        U_tot = float(np.sum(U * eq.area))
        bind = float(np.sum(drain * eq.area))
        # what the memoryless static model would report as unbound at this instant
        M = GAMMA * GammaD
        u_inst = float(np.sum(M * eq.g_hat * surv * (1.0 - Phi) * eq.area))

        rec["t"].append(t); rec["A_K"].append(A_K)
        rec["GammaD"].append(GammaD); rec["dGamma"].append(dGamma)
        rec["U_tot"].append(U_tot); rec["bind"].append(bind)
        rec["u_inst"].append(u_inst)
        rec["share_unbound"].append(U_tot / M if M > 1e-12 else 0.0)

        # population re-sorts (moving origin) on the standard task-layer value
        T = L @ softmax_P(eq, W, c, kappa)
        L = L + (dt / theta_L) * (T - L)

    return {k: np.array(v) for k, v in rec.items()}


def main():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    A_final = tech.A_K
    print(f"  calibrated A_K (mature) = {A_final:.3f}; trajectory 0 -> {A_final:.3f}")

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq.L0 = L0
    # fixed A_K-independent shape g0(r) = exp(-1/2 (d/z_K)^2)
    unit = Technology(xi_K=tech.xi_K, chi_K=tech.chi_K, z_K=tech.z_K, A_K=1.0, s_K=1.0)
    g0_grid = unit.phi(inp.grid.xi, inp.grid.chi)
    g0_task = unit.phi(eq.b_xi, eq.b_chi)

    # mobility reference (as script 12): set A_K = 0 baseline value for kappa
    set_AK(eq, 0.0, g0_grid, g0_task)
    _, _, W0 = eq.density_and_value(L0)
    kappa = float(np.std(W0)); dmed = float(np.median(eq.d[eq.d > 0]))
    c = kappa / dmed
    print(f"  mobility: kappa {kappa:.3f}, c {c:.3f}; maturation rho 0.5 at t_mid 20\n")

    cases = {"fast workers (theta_L=1)": 1.0,
             "reference (theta_L=3)": 3.0,
             "slow workers (theta_L=8)": 8.0}
    colors = {"fast workers (theta_L=1)": "#2C5A57",
              "reference (theta_L=3)": "#B5532A",
              "slow workers (theta_L=8)": "#6D3C8E"}
    res = {}
    print(f"  {'case':28} {'peak U_tot':>11} {'t(peakU)':>9} "
          f"{'t(peak dGamma)':>14} {'lag':>6}")
    for name, theta in cases.items():
        r = run_trajectory(eq, L0, g0_grid, g0_task, c, kappa,
                           A_final, rho=0.5, t_mid=20.0, theta_L=theta, lam_b=1.0)
        res[name] = r
        i_u = int(np.argmax(r["U_tot"])); i_s = int(np.argmax(r["dGamma"]))
        lag = r["t"][i_u] - r["t"][i_s]
        print(f"  {name:28} {r['U_tot'][i_u]:11.4f} {r['t'][i_u]:9.1f} "
              f"{r['t'][i_s]:14.1f} {lag:+6.1f}")

    # figure: U_tot(t) hump for each mobility speed, with the seeding-rate driver
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in res.items():
        ax1.plot(r["t"], r["U_tot"], lw=2, color=colors[name], label=name)
    ax1.set_xlabel("time"); ax1.set_ylabel(r"unbound stock $U_{tot}(t)$")
    ax1.set_title("The limbo hump grows when technology outpaces mobility")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    r = res["reference (theta_L=3)"]
    ax2.plot(r["t"], r["dGamma"] / max(r["dGamma"].max(), 1e-9), lw=1.6,
             color="0.4", ls="--", label=r"seeding driver $[\dot\Gamma^D]_+$ (norm)")
    ax2.plot(r["t"], r["U_tot"] / max(r["U_tot"].max(), 1e-9), lw=2,
             color="#B5532A", label=r"unbound stock $U_{tot}$ (norm)")
    ax2.plot(r["t"], r["bind"] / max(r["bind"].max(), 1e-9), lw=1.6,
             color="#2C5A57", label="binding flow (norm)")
    ax2.plot(r["t"], r["u_inst"] / max(r["u_inst"].max(), 1e-9), lw=1.2,
             color="0.6", ls=":", label="static instantaneous $u$ (norm)")
    ax2.set_xlabel("time"); ax2.set_ylabel("normalized")
    ax2.set_title("Lead-lag (reference): destruction leads, binding lags")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "U_hump.png", dpi=150)
    print(f"\n  wrote {OUT / 'U_hump.png'}")


if __name__ == "__main__":
    main()
