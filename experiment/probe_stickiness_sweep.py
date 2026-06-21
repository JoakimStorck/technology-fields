"""
probe_stickiness_sweep.py
-------------------------
Follow-up to the gap measurement. Sweep mobility stickiness -- the cost of a
typical move in logit units, s = c * d_median / kappa (reference s = 1) -- and
trace two things across it:

  gap(s)      = ||L*(s) - L**(s)||_1, the distance between the static
                frozen-origin fixed point and the dynamic moving-origin rest
                point at that stickiness.
  hyst(s)     = max spread of the dynamic rest point over several start points,
                i.e. whether the moving-origin map still has a UNIQUE attractor
                or has split into path-dependent basins.

Expectation: gap is a HUMP in stickiness. At s -> 0 (free mobility) the origin
is irrelevant, so frozen and moving origin coincide (gap -> 0). At s -> inf
(frozen solid) nobody moves from L0 in either model, so they coincide again
(gap -> 0). The gap peaks at intermediate stickiness where origin matters AND
people still move. Hysteresis, if it appears at all, appears on the sticky side.

Stickiness is swept by scaling c at fixed kappa: c = s * (kappa / d_median).
Builds the equilibrium ONCE (the field pre-computation is independent of c,
kappa); only the re-sorting cost varies. Touches no committed code.
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

_spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts" / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
OUT = REPO / "scripts"


def softmax_P(eq, W, c, kappa):
    U = (W[None, :] - c * eq.d) / kappa
    U -= U.max(axis=1, keepdims=True)
    P = np.exp(U)
    P /= P.sum(axis=1, keepdims=True)
    return P


def dyn_rest(eq, c, kappa, L_start, theta=1.0, dt=0.25, tol=1e-8, maxsteps=6000):
    """Forward-time integration of the moving-origin law of motion."""
    L = L_start.copy()
    for step in range(1, maxsteps + 1):
        _, _, W = eq.density_and_value(L)
        T = L @ softmax_P(eq, W, c, kappa)
        drift = T - L
        if np.abs(drift).sum() < tol:
            return L, step
        L = L + (dt / theta) * drift
    return L, maxsteps


def main():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq.L0 = L0

    _, _, W0 = eq.density_and_value(L0)
    kappa = float(np.std(W0))
    dmed = float(np.median(eq.d[eq.d > 0]))
    c_ref = kappa / dmed                      # stickiness s = 1 at reference

    l1 = lambda a, b: float(np.abs(a - b).sum())
    rng = np.random.default_rng(0)
    uniform = np.full(len(L0), 1.0 / len(L0))

    sticks = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                       4.0, 6.0, 8.0, 12.0, 20.0])

    # ---- pass 1: gap only (fast: one solve + one integration per point) ----
    print(f"  reference: kappa {kappa:.3f}, median move {dmed:.3f}, "
          f"c_ref {c_ref:.3f}  (stickiness s = c*d_med/kappa)\n", flush=True)
    print(f"  {'s':>6} {'gap':>9} {'gap%mass':>9} {'gap/move':>9} "
          f"{'steps':>7}", flush=True)
    rows, Lstars = [], {}
    for s in sticks:
        c = s * c_ref
        Lstar = eq.solve(c, kappa).L
        Ldyn, steps = dyn_rest(eq, c, kappa, L0)
        gap = l1(Lstar, Ldyn)
        move = max(l1(Lstar, L0), 1e-12)
        Lstars[s] = Lstar
        rows.append([s, gap, gap / 2, gap / move, np.nan])
        print(f"  {s:6.2f} {gap:9.4f} {100*gap/2:8.1f}% {gap/move:9.3f} "
              f"{steps:7d}", flush=True)
    rows = np.array(rows)

    # ---- pass 2: hysteresis only at the 3 stickiest points (where, if ever,
    #              the attractor would split) ----
    print(f"\n  hysteresis check at the stickiest points:", flush=True)
    for idx in np.argsort(rows[:, 0])[-3:]:
        s = rows[idx, 0]; c = s * c_ref
        starts = [L0, uniform, Lstars[s],
                  rng.random(len(L0)) / rng.random(len(L0)).sum()]
        sols = [dyn_rest(eq, c, kappa, st)[0] for st in starts]
        hyst = max(l1(a, b) for a in sols for b in sols)
        rows[idx, 4] = hyst
        print(f"    s={s:6.2f}  spread {hyst:.2e}  "
              f"{'unique' if hyst < 1e-6 else 'SPLIT'}", flush=True)

    peak = rows[np.argmax(rows[:, 1])]
    print(f"\n  gap peaks at stickiness s = {peak[0]:.2f}: "
          f"gap {peak[1]:.4f} ({100*peak[2]:.1f}% of mass)", flush=True)

    # figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(rows[:, 0], 100 * rows[:, 2], "o-", color="#B5532A", lw=1.8,
                 label="gap (% of mass)")
    ax1.axvline(1.0, color="0.6", ls="--", lw=1, label="reference (1 logit unit)")
    ax1.set_xlabel("mobility stickiness  $s = c\\,d_{med}/\\kappa$  "
                   "(cost of a typical move, logit units)")
    ax1.set_ylabel("gap $\\|L^*-L^{**}\\|_1$  (% of mass)", color="#B5532A")
    ax1.tick_params(axis="y", labelcolor="#B5532A")
    ax2 = ax1.twinx()
    hmask = np.isfinite(rows[:, 4])
    ax2.semilogx(rows[hmask, 0], rows[hmask, 4], "s--", color="#2C5A57", lw=1.3,
                 ms=5, label="hysteresis spread (stickiest pts)")
    ax2.set_ylabel("hysteresis (max spread over starts)", color="#2C5A57")
    ax2.tick_params(axis="y", labelcolor="#2C5A57")
    ax1.set_title("Frozen-vs-moving origin gap is a hump in mobility stickiness")
    fig.tight_layout()
    fig.savefig(OUT / "stickiness_sweep.png", dpi=150)
    print(f"\n  wrote {OUT / 'stickiness_sweep.png'}")


if __name__ == "__main__":
    main()
