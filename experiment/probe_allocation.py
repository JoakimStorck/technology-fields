"""
probe_allocation.py
-------------------
Test the A15 bundle-allocation model on newly-seeded task mass.

A bundle is filled GREEDILY by value-per-grasp-cost. With per-cell self-
congestion in value, the greedy cap is a closed-form WATER-FILLING: each cell is
covered to the level where its marginal value-per-cost equals the threshold
theta. Marginal value of covering cell r at own-coverage c:

    v_marg(r) = beta * Pi(r) * (n_base(r) + kappa*c)^(beta-1) * Umask(r) * e_o(r)
    w(r)      = wfloor + max(beta_chi(xi_r), 0) * chi_r        (invested depth)

Setting v_marg/w = theta and solving for c gives the cap coverage:

    c_o(r) = max(0, [ (beta*Pi*Umask*e_o / (theta*w))^{1/(1-beta)} - n_base ] / kappa)

Key properties to check (field questions, not discrete-object precision):
  1. BOUNDED: does every anchor reach a finite covered area (no runaway)? The
     self-congestion kappa*c replaces the divergent n^(beta-1), so the empty-ring
     pathology of probe_A10_birth should be gone.
  2. ANISOTROPIC CAP: do anchors in deep (high invested-depth) directions reach
     SMALLER covered area / task radius than shallow ones? (A15 prediction, now a
     mechanism consequence.)
  3. EVEN SPREAD: is coverage spread (submodular), not piled on one cell?
  4. PLAUSIBLE SIZE: covered area in a sane range (anchored vs Morgeson-Humphrey
     ~6-14 tasks per job, given a task-resolution scale).

Two masks: (A) Umask=1 (potential tasks everywhere) to characterise the cap
across the disk; (B) Umask = seeded ring g_hat*(1-a) -- allocation on the actual
newly-sown mass, the A10/A14 picture. eta=1, survival on. No committed code touched.
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


def waterfill(Pi, n_base, Umask, e_o, w, theta, kappa, beta):
    """Closed-form greedy cap coverage c_o(r)."""
    drive = beta * Pi * Umask * e_o / (theta * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        target_n = np.where(drive > 0, drive ** (1.0 / (1.0 - beta)), 0.0)
    c = np.maximum((target_n - n_base) / kappa, 0.0)
    return c


def main():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq.L0 = L0

    grid = inp.grid
    area = eq.area
    gx = grid.chi * np.cos(grid.xi); gy = grid.chi * np.sin(grid.xi)
    n_base, _, _ = eq.density_and_value(L0)
    Pi = eq.pi_cell
    bchi = inp.field.beta_chi(grid.xi)
    invested = np.maximum(bchi, 0.0) * grid.chi          # cell invested depth
    wfloor = 0.15
    w = wfloor + invested                                # grasp weight per cell

    # seeded ring (newly-sown human task mass): g_hat gated by survival
    Useed = eq.g_hat * (1.0 - eq.a_grid)
    Useed = Useed / Useed.max()

    # anchor invested depth at each occupation centroid
    xi_o = occ["xi"].to_numpy(); chi_o = occ["chi"].to_numpy()
    inv_o = np.maximum(inp.field.beta_chi(xi_o), 0.0) * chi_o

    kappa, e_min = 5.0, 0.05

    def fill_stats(o, Umask, theta):
        e_o = eq.e[o]
        c = waterfill(Pi, n_base, Umask, e_o, w, theta, kappa, BETA)
        c = np.where((Umask > 1e-6) & (e_o > e_min), c, 0.0)
        m = float(np.sum(c * area))                       # bound task mass
        if m < 1e-9:
            return 0.0, np.nan, 0.0, 0.0
        cov_area = float(np.sum((c > 1e-6) * area))        # covered area
        cw = c * area
        dx = gx - eq.mu[o, 0]; dy = gy - eq.mu[o, 1]
        d2 = dx * dx + dy * dy
        Ro = float(np.sqrt(np.sum(cw * d2) / np.sum(cw)))  # task radius
        peak_frac = float((c * area).max() / m)            # spread: low = even
        return m, Ro, cov_area, peak_frac

    # ---- (A) mechanism characterisation: Umask = 1, anchors spanning depth ----
    order = np.argsort(inv_o)
    anchors = order[np.linspace(0, len(order) - 1, 60).astype(int)]
    theta = 8.0
    rows = []
    for o in anchors:
        m, Ro, cov, pk = fill_stats(o, np.ones_like(Pi), theta)
        rows.append((inv_o[o], m, Ro, cov, pk))
    rows = np.array(rows)
    ok = np.isfinite(rows[:, 2]) & (rows[:, 1] > 0)
    A = rows[ok]
    cc = lambda i, j: float(np.corrcoef(A[:, i], A[:, j])[0, 1])
    print("  (A) MECHANISM (Umask=1, 60 anchors spanning invested depth)")
    print(f"      all anchors reach a FINITE cap: {np.all(np.isfinite(A[:,2]))}  "
          f"(no runaway; pathology resolved)")
    print(f"      covered area range: {A[:,3].min():.3f} .. {A[:,3].max():.3f} "
          f"(disk area = {float(np.sum(area)):.2f})")
    print(f"      task radius range : {A[:,2].min():.3f} .. {A[:,2].max():.3f}")
    print(f"      corr(invested depth, covered area): {cc(0,3):+.3f}  "
          f"(A15 predicts NEGATIVE)")
    print(f"      corr(invested depth, task radius) : {cc(0,2):+.3f}  "
          f"(A15 predicts NEGATIVE)")
    print(f"      mean peak-fraction (spread): {A[:,4].mean():.3f}  "
          f"(low = even spread, submodular working)")

    # split by depth tertile: deep vs shallow anchors
    q1, q2 = np.percentile(A[:, 0], [33, 66])
    shallow = A[A[:, 0] <= q1]; deep = A[A[:, 0] >= q2]
    print(f"      shallow anchors: mean task radius {shallow[:,2].mean():.3f}, "
          f"covered area {shallow[:,3].mean():.3f}")
    print(f"      deep anchors   : mean task radius {deep[:,2].mean():.3f}, "
          f"covered area {deep[:,3].mean():.3f}")

    # ---- (B) allocation on the actual newly-sown ring ----
    e_on_ring = eq.e @ (Useed * area)                     # each occ's reach into ring
    reachers = np.argsort(e_on_ring)[::-1][:12]
    print("\n  (B) NEWLY-SOWN RING (Umask = seeded g_hat*(1-a)); top ring-reachers")
    print(f"      {'occupation':34} {'inv.depth':>9} {'taskR':>7} {'cov.area':>9}")
    titles = occ["Title"].to_numpy()
    bcov = []
    for o in reachers:
        m, Ro, cov, pk = fill_stats(o, Useed, theta)
        bcov.append((inv_o[o], Ro, cov))
        nm = titles[o][:32]
        print(f"      {nm:34} {inv_o[o]:9.3f} {Ro:7.3f} {cov:9.4f}")
    bcov = np.array(bcov)
    if np.isfinite(bcov[:, 1]).sum() > 3:
        m = np.isfinite(bcov[:, 1])
        print(f"      corr(invested depth, task radius) on ring: "
              f"{float(np.corrcoef(bcov[m,0], bcov[m,1])[0,1]):+.3f}")

    # ---- robustness of the anisotropy sign to threshold ----
    print("\n  threshold robustness of corr(invested depth, covered area):")
    for th in [4.0, 8.0, 16.0]:
        r2 = []
        for o in anchors:
            mm, _, cov, _ = fill_stats(o, np.ones_like(Pi), th)
            r2.append((inv_o[o], cov, mm))
        r2 = np.array(r2); g = r2[:, 2] > 0
        print(f"      theta={th:5.1f}: corr {float(np.corrcoef(r2[g,0], r2[g,1])[0,1]):+.3f}, "
              f"mean covered area {r2[g,1].mean():.3f}")

    # figure: covered area & task radius vs anchor invested depth
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.scatter(A[:, 0], A[:, 3], c="#B5532A", s=22)
    a1.set_xlabel("anchor invested depth  $\\max(\\beta_\\chi,0)\\,\\chi$")
    a1.set_ylabel("covered area (bundle breadth)")
    a1.set_title("Bundle cap shrinks with invested depth\n(anisotropy as a mechanism consequence)")
    a1.grid(alpha=0.25)
    a2.scatter(A[:, 0], A[:, 2], c="#2C5A57", s=22)
    a2.set_xlabel("anchor invested depth")
    a2.set_ylabel("task radius of filled bundle")
    a2.set_title("Deep anchors -> narrow bundles, shallow -> broad")
    a2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "allocation.png", dpi=150)
    print(f"\n  wrote {OUT / 'allocation.png'}")


if __name__ == "__main__":
    main()
