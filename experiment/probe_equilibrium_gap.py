"""
probe_equilibrium_gap.py
-------------------------
Whitepaper section 9.6 / 10, first step: measure the gap between

  (S) the STATIC equilibrium -- the frozen-origin fixed point
      L* = L0 @ P(W(L*)),  origin frozen at the pre-technology shares L0,
      computed as a fixed point (model.equilibrium.solve, the verified static).

  (D) the DYNAMIC rest point -- the moving-origin law of motion
      dL/dt = (1/theta)[ T(L) - L ],   T(L) = L @ P(W(L)),  origin = current L,
      INTEGRATED forward in time from the pre-technology start image, until
      it settles. The rest point satisfies L** = L** @ P(W(L**)).

These are two different computations (a fixed-point solve vs. a time
integration) of two different objects. The gap ||L* - L**|| measures how much
the frozen-origin assumption (everyone re-chooses from the pre-tech
distribution) differs from the crept-along outcome (the population drags its
current distribution). If the gap is small relative to how far each moved from
L0, finite-speed history barely matters; if comparable, it is a real finding.

Also: integrate the dynamic from several starts to test path dependence
(hysteresis) -- does the moving-origin map have one attractor or several?

Reads only frozen inputs + the verified calibration; touches no committed code.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.equilibrium import Equilibrium

_spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts" / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5


def softmax_P(eq, W, c, kappa):
    """Row-stochastic re-sorting kernel P[o', o] = P(dest o | origin o'),
    identical to Equilibrium.resort but returned as the matrix."""
    U = (W[None, :] - c * eq.d) / kappa
    U -= U.max(axis=1, keepdims=True)
    P = np.exp(U)
    P /= P.sum(axis=1, keepdims=True)
    return P


def integrate_dynamic(eq, c, kappa, L_start, theta=1.0, dt=0.2,
                      tol=1e-11, maxsteps=200000):
    """Forward-time integration of the moving-origin law of motion by explicit
    Euler. Origin is the CURRENT L (not a frozen L0). Returns the rest point and
    the number of steps (proportional to settling time)."""
    L = L_start.copy()
    for step in range(1, maxsteps + 1):
        _, _, W = eq.density_and_value(L)
        P = softmax_P(eq, W, c, kappa)
        T = L @ P                       # moving origin: current L as row weights
        drift = T - L
        res = np.abs(drift).sum()
        if res < tol:
            return L, step, res
        L = L + (dt / theta) * drift    # genuine time step, not a solver damp
    return L, maxsteps, res


def run(tag, wedge):
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=wedge,
                     survival=True)
    eq.L0 = L0

    # mobility reference from the baseline (pre-tech) value, as in script 12
    _, _, W0 = eq.density_and_value(L0)
    c, kappa, dmed = _setup.mobility_reference(W0, eq.d)

    # (S) static frozen-origin fixed point
    static = eq.solve(c, kappa)
    Lstar = static.L

    # (D) dynamic moving-origin rest point, integrated from the pre-tech start
    Ldyn, steps, dres = integrate_dynamic(eq, c, kappa, L0)

    # gap measures (L1 = sum of abs差, "share of mass that sits differently")
    def l1(a, b):
        return float(np.abs(a - b).sum())

    gap = l1(Lstar, Ldyn)
    move_static = l1(Lstar, L0)     # how far static moved from pre-tech
    move_dyn = l1(Ldyn, L0)         # how far dynamics moved from pre-tech

    # path dependence: integrate dynamics from several starts
    rng = np.random.default_rng(0)
    starts = {"pre-tech L0": L0,
              "uniform": np.full(len(L0), 1.0 / len(L0)),
              "static L*": Lstar}
    for _ in range(2):
        r = rng.random(len(L0)); starts[f"random"] = r / r.sum()
    sols = {k: integrate_dynamic(eq, c, kappa, s)[0] for k, s in starts.items()}
    keys = list(sols)
    hyst = max(l1(sols[a], sols[b]) for a in keys for b in keys)

    print(f"\n================  {tag}  ================")
    print(f"  occupations {len(L0)},  ell {ell:.4f},  "
          f"kappa {kappa:.3f}, c {c:.3f}, median move {dmed:.3f}")
    print(f"  static : converged {static.converged} in {static.iters} iters "
          f"(fixed-point solve, frozen origin)")
    print(f"  dynamic: settled in {steps} Euler steps "
          f"(time integration, moving origin), residual {dres:.1e}")
    print(f"\n  -- displacement from the pre-technology start (L1) --")
    print(f"     static  ||L* - L0||   = {move_static:.4f}  "
          f"({100*move_static/2:.1f}% of mass relocated)")
    print(f"     dynamic ||L**- L0||   = {move_dyn:.4f}  "
          f"({100*move_dyn/2:.1f}% of mass relocated)")
    print(f"\n  -- the gap between the two equilibria --")
    print(f"     ||L* - L**||          = {gap:.4f}  "
          f"({100*gap/2:.1f}% of mass sits differently)")
    print(f"     gap / static move     = {gap/move_static:.3f}")
    print(f"     gap / dynamic move    = {gap/move_dyn:.3f}")
    print(f"\n  -- path dependence (hysteresis) --")
    print(f"     max spread over {len(sols)} starts = {hyst:.2e}  "
          f"({'unique attractor' if hyst < 1e-6 else 'PATH DEPENDENT'})")

    # which occupations differ most between the two equilibria
    diff = Lstar - Ldyn
    titles = occ["Title"].to_numpy()
    order = np.argsort(np.abs(diff))[::-1][:6]
    print(f"\n  -- occupations where the two equilibria differ most --")
    for i in order:
        print(f"     {titles[i][:42]:42s} L*={Lstar[i]:.3e} "
              f"L**={Ldyn[i]:.3e}  d={diff[i]:+.2e}")
    return gap, move_static, hyst


if __name__ == "__main__":
    g1, m1, h1 = run("no wedge", None)
    inp, L0, occ = _setup.build_inputs()
    g2, m2, h2 = run("with wedge", _setup.load_wedge(occ))
    print("\n================  verdict  ================")
    print(f"  no wedge : gap {g1:.4f}  = {100*g1/m1:.0f}% of the static "
          f"re-sorting; hysteresis {h1:.1e}")
    print(f"  with wedge: gap {g2:.4f}  = {100*g2/m2:.0f}% of the static "
          f"re-sorting; hysteresis {h2:.1e}")
