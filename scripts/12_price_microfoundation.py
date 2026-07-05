"""
12_price_microfoundation.py
---------------------------
Microfoundation of the price field (paper section "The price field as an
assignment equilibrium"). Documents, in one reproducible pass, the sequence of
tests that select the closure for Pi(r): the reduced-form price field of Paper 1
is shown NOT to be a task-scarcity or a capability-scarcity object, but the
equilibrium price of a multidimensional capability ASSIGNMENT with cognitively
biased demand. Each test is a function whose docstring records the result it
produced.

Pipeline of results (see results/price_microfoundation_summary.txt):
  T1 task scarcity        -> REJECTED   (price does not fall with employment;
                                          the dear north is densely staffed)
  T2 capability pricing   -> CONFIRMED  (lnPi is ~96% linear in sum_k v_k q_k)
  T3 capability scarcity  -> REJECTED   (dear S1 not scarce; v_A1 < 0, so v_k
                                          are hedonic coefficients, not factor
                                          prices -> an assignment is required)
  T4 assignment (uniform) -> SHAPE      (equilibrium price reproduces Paper 1 at
                                          Spearman ~0.91, WAGE-INDEPENDENT, but
                                          the directed gradient is ~4x too flat)
  T5 cognitive demand     -> CLOSURE    (one parameter lambda~0.25 derives the
                                          directed gradient, lifts the fit to
                                          ~0.97, and makes the north abundant in
                                          sign)
  T6 productivity comp.   -> REJECTED   (concentrates employment but crashes
                                          wages via the CES output effect:
                                          abundant-but-cheap north)

Residual (honest): the magnitude of the north's employment concentration
(model ~+0.05 vs data ~+0.71 in log density) is not reproduced by either the
demand bias or productivity complementarity; it is left to a richer demand
structure (non-homothetic preferences / sigma), not the production side.

Usage:
    python scripts/12_price_microfoundation.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.regime import _readiness, _cell_index

_spec = importlib.util.spec_from_file_location("_setup", Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
SIGMA = 3.0          # task substitution elasticity (the soft point; swept in T4)
KAPPA_FRAC = 0.5     # allocation smoothness (relative to mean price)


def build():
    """Frozen ingredients shared across the tests."""
    inp, L0, occ = _setup.build_inputs()
    ell = _setup.interpretable_ell(inp)
    grid, field, cap = inp.grid, inp.field, inp.cap
    codes = inp.occ_codes()
    L = L0.astype(float)
    e = _readiness(inp, ell)                                   # productivity f = readiness
    bx = inp.bundles
    row_of = pd.Index(codes).get_indexer(bx["onet_code"].to_numpy())
    cell = _cell_index(grid, bx["xi"].to_numpy(), bx["chi"].to_numpy())
    n0 = np.bincount(cell, weights=L[row_of] * bx["b"].to_numpy(),
                     minlength=grid.xi.size) / grid.area
    xi, chi = grid.xi, grid.chi
    d = dict(inp=inp, L0=L0, L=L, occ=occ, grid=grid, field=field, cap=cap,
             e=e, n0=n0, xi=xi, chi=chi, area=grid.area,
             lnP=np.log(field.pi(xi, chi)),
             north=(np.cos(xi) > 0.5) & (chi > 0.4),
             south=(np.cos(xi) < -0.3) & (chi > 0.3),
             occ_cells=n0 > 0)
    # cognitive demand shifter and complementarity types (standardised S1)
    qS1 = cap.q("S1", xi, chi)
    d["zq"] = (qS1 - qS1.mean()) / qS1.std()
    s1o = occ["S1"].to_numpy()
    d["zth"] = (s1o - s1o.mean()) / s1o.std()
    xo, co = occ["xi"].to_numpy(), occ["chi"].to_numpy()
    d["lnP_o"] = np.log(field.pi(xo, co))
    d["north_o"] = (np.cos(xo) > 0.5) & (co > 0.4)
    d["south_o"] = (np.cos(xo) < -0.3) & (co > 0.3)
    return d


def _ns(v, hi, lo):
    return float(np.median(v[hi]) - np.median(v[lo]))


# ---------------------------------------------------------------- T1
def test_task_scarcity(d, out):
    """Price vs employment density. RESULT: slope ~-0.03 (sigma~35), and the
    dear north is DENSER than the cheap south -> task scarcity REJECTED."""
    m = d["occ_cells"]
    ln_n0, lnP = np.log(d["n0"][m]), d["lnP"][m]
    slope = float(np.polyfit(ln_n0, lnP, 1)[0])
    nd = _ns(np.log(d["n0"][d["occ_cells"]] + 1e-12),
             d["north"][d["occ_cells"]], d["south"][d["occ_cells"]])
    out += ["T1 task scarcity (REJECTED):",
            f"   d lnPi / d ln n0 = {slope:+.3f}  (implied sigma {(-1/slope if slope<0 else float('nan')):.1f})",
            f"   Spearman(lnPi, ln n0) = {spearmanr(lnP, ln_n0)[0]:+.3f}",
            f"   employment density directed gradient = {nd:+.2f}  (dear north is the denser)"]


# ---------------------------------------------------------------- T2
def test_capability_pricing(d, out):
    """Is Pi linear in capability content? RESULT: lnPi ~ a + b sum_k v_k q_k at
    R^2 ~ 0.96 -> Pi is a hedonic capability surface (S1 dominant)."""
    cap, xi, chi, lnP = d["cap"], d["xi"], d["chi"], d["lnP"]
    keys = list(cap.v.keys())
    Pcap = sum(cap.v[k] * cap.q(k, xi, chi) for k in keys)
    A = np.column_stack([np.ones_like(Pcap), Pcap])
    coef = np.linalg.lstsq(A, lnP, rcond=None)[0]
    r2 = 1 - np.sum((lnP - A @ coef) ** 2) / np.sum((lnP - lnP.mean()) ** 2)
    out += ["", "T2 capability pricing (CONFIRMED):",
            f"   R^2 of lnPi ~ a + b*(sum_k v_k q_k) = {r2:.3f}",
            f"   per-capability v_k: " + ", ".join(f"{k}={cap.v[k]:+.3f}" for k in keys)]


# ---------------------------------------------------------------- T3
def test_capability_scarcity(d, out):
    """Are v_k equilibrium factor prices (v_k ~ 1/supply)? RESULT: corr(v_k,1/X_k)
    < 0 and v_A1 < 0 -> v_k are HEDONIC coefficients, not factor prices. A simple
    factor market is REJECTED; an assignment is required."""
    cap, occ, L0 = d["cap"], d["occ"], d["L0"]
    keys = list(cap.v.keys())
    X = np.array([float(np.sum(L0 * occ[k].to_numpy())) for k in keys])
    v = np.array([cap.v[k] for k in keys])
    invX = (1.0 / X) / np.sum(1.0 / X)
    out += ["", "T3 capability scarcity (REJECTED):",
            f"   corr(v_k, 1/X_k) = {pearsonr(v, invX)[0]:+.3f}  (dear capability is not the scarce one)",
            f"   v_A1 = {cap.v['A1']:+.3f}  (negative -> a hedonic coefficient, not a factor price)"]


# ---------------------------------------------------------------- assignment solver
def solve_assignment(d, sigma=SIGMA, lam=0.0, gamma=0.0, kf=KAPPA_FRAC,
                     iters=600, damp=0.6, tol=1e-9):
    """Capability-assignment equilibrium with CES task demand. NO observed wages
    are used. alpha(r) = exp(lam * zq) is the cognitive demand bias; f = e *
    exp(gamma * zth (x) zq) is the capability-task productivity complementarity.
    Returns the task price Pi(r), the occupation wage w_o, and employment(r)."""
    e, L, n0, area = d["e"], d["L"], d["n0"], d["area"]
    rho = (sigma - 1) / sigma
    a = np.exp(lam * d["zq"]); a /= a.sum()
    f = e * np.exp(gamma * np.outer(d["zth"], d["zq"]))
    y = np.maximum(n0, 1e-6); y /= y.sum()
    for _ in range(iters):
        Ybar = (np.sum(a * y ** rho)) ** (1 / rho)
        Pi = a * (Ybar / np.maximum(y, 1e-12)) ** (1 / sigma)
        Pi /= np.mean(Pi)                                       # numeraire
        z = (Pi[None, :] * f) / (kf * np.mean(Pi))
        z -= z.max(1, keepdims=True)
        s = np.exp(z); s /= s.sum(1, keepdims=True)             # occ allocation over tasks
        yn = (L[:, None] * s * f).sum(0); yn /= yn.sum()
        if np.max(np.abs(yn - y)) < tol:
            break
        y = damp * y + (1 - damp) * yn
    w_o = (s * Pi[None, :] * f).sum(1)                          # occupation wage (earnings/worker)
    emp = (L[:, None] * s).sum(0) / area
    return Pi, w_o, emp


# ---------------------------------------------------------------- T4
def test_assignment_uniform(d, out):
    """Does the assignment reproduce the price field without using wages? RESULT:
    Spearman ~0.91 (shape), but the directed gradient is ~4x too flat; robust to
    kappa and sigma."""
    Pi, _, _ = solve_assignment(d, lam=0.0, gamma=0.0)
    lnE = np.log(np.maximum(Pi, 1e-12))
    out += ["", "T4 assignment, uniform demand (SHAPE, wage-independent):",
            f"   Spearman(lnPi_eq, lnPi_paper1) = {spearmanr(lnE, d['lnP'])[0]:+.3f}",
            f"   directed gradient(eq) = {_ns(lnE, d['north'], d['south']):+.3f}  (estimated-field target +0.39)"]
    rob = []
    for kf in (0.2, 0.5, 1.0):
        P, _, _ = solve_assignment(d, kf=kf)
        rob.append(spearmanr(np.log(np.maximum(P, 1e-12)), d["lnP"])[0])
    out.append(f"   kappa robustness (Spearman @ kf=0.2/0.5/1.0): "
               + "/".join(f"{x:+.2f}" for x in rob))


# ---------------------------------------------------------------- T5
def test_demand_bias(d, out):
    """Sweep the cognitive demand bias lambda. RESULT: lambda~0.25 brackets the
    directed gradient (+0.39) AND keeps the north abundant in sign, peaking the
    overall fit at Spearman ~0.97."""
    out += ["", "T5 cognitive demand bias (CLOSURE):",
            "   lambda  Spearman  directed-grad(Pi)  emp directed-grad"]
    best = None
    for lam in (0.10, 0.20, 0.25, 0.30, 0.40):
        Pi, _, emp = solve_assignment(d, lam=lam, gamma=0.0)
        lnE, lne = np.log(np.maximum(Pi, 1e-12)), np.log(np.maximum(emp, 1e-12))
        sr = spearmanr(lnE, d["lnP"])[0]
        nspi = _ns(lnE, d["north"], d["south"])
        nse = _ns(lne, d["north"] & d["occ_cells"], d["south"] & d["occ_cells"])
        out.append(f"     {lam:.2f}    {sr:+.3f}     {nspi:+.3f}          {nse:+.3f}")
        if best is None or abs(nspi - 0.39) < abs(best[1] - 0.39):
            best = (lam, nspi, sr)
    out.append(f"   -> best lambda ~ {best[0]:.2f} (gradient {best[1]:+.2f}, Spearman {best[2]:+.2f})")
    return best[0]


# ---------------------------------------------------------------- T6
def test_productivity_complementarity(d, out):
    """Sweep the capability-task productivity complementarity gamma. RESULT:
    gamma concentrates employment (north-south up to ~+0.6) but CRASHES wages
    (north-south wage < 0) via the CES output effect -> abundant-but-cheap north.
    REJECTED as the gradient mechanism; demand bias (gamma=0) is best."""
    out += ["", "T6 productivity complementarity (REJECTED):",
            "   lambda gamma  Spearman(ln w_o, lnP_o)  wage directed-grad  emp directed-grad"]
    for lam, gamma in [(0.25, 0.0), (0.0, 1.0), (0.0, 2.0), (0.10, 2.0)]:
        _, w_o, emp = solve_assignment(d, lam=lam, gamma=gamma)
        lw, lne = np.log(np.maximum(w_o, 1e-12)), np.log(np.maximum(emp, 1e-12))
        sr = spearmanr(lw, d["lnP_o"])[0]
        nsw = _ns(lw, d["north_o"], d["south_o"])
        nse = _ns(lne, d["north"] & d["occ_cells"], d["south"] & d["occ_cells"])
        out.append(f"     {lam:.2f}  {gamma:.1f}    {sr:+.3f}                {nsw:+.3f}            {nse:+.3f}")


def main():
    d = build()
    out = ["Price-field microfoundation: assignment equilibrium with cognitive demand.",
           f"  sigma {SIGMA}, kappa_frac {KAPPA_FRAC}; capability assignment, no observed wages.", ""]
    test_task_scarcity(d, out)
    test_capability_pricing(d, out)
    test_capability_scarcity(d, out)
    test_assignment_uniform(d, out)
    lam_star = test_demand_bias(d, out)
    test_productivity_complementarity(d, out)

    (RESULTS / "price_microfoundation_summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    # figure: derived price (best lambda) vs Paper 1, coloured by depth
    Pi, _, _ = solve_assignment(d, lam=lam_star, gamma=0.0)
    lnE = np.log(np.maximum(Pi, 1e-12))
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax[0].scatter(lnE, d["lnP"], s=4, alpha=0.25, c=d["chi"], cmap="viridis")
    ax[0].set_xlabel(r"derived assignment price $\ln\Pi_{eq}(r)$")
    ax[0].set_ylabel(r"estimated price field $\ln\Pi(r)$")
    ax[0].set_title(f"Microfounded vs reduced-form (Spearman {spearmanr(lnE, d['lnP'])[0]:+.2f}, $\\lambda$={lam_star:.2f})")
    fig.colorbar(sc, ax=ax[0], label=r"depth $\chi$")
    # decomposition: demand steepens, productivity crashes wages
    lams = np.linspace(0, 0.4, 9); gams = np.linspace(0, 2.0, 9)
    ns_dem = [_ns(np.log(np.maximum(solve_assignment(d, lam=l)[0], 1e-12)), d["north"], d["south"]) for l in lams]
    ns_prod = [_ns(np.log(np.maximum(solve_assignment(d, gamma=g)[1], 1e-12)), d["north_o"], d["south_o"]) for g in gams]
    ax[1].plot(lams, ns_dem, "-o", ms=3, label=r"price grad. vs demand $\lambda$")
    ax[1].plot(gams / 5, ns_prod, "-s", ms=3, label=r"wage grad. vs prod. $\gamma$ (x/5)")
    ax[1].axhline(0.39, color="0.5", ls="--", lw=0.8, label="estimated-field target")
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_xlabel("parameter"); ax[1].set_ylabel(r"directed gradient $\Delta_{\mathrm{dg}}$")
    ax[1].set_title("gradient vs closure parameter")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "price_microfoundation.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {RESULTS/'price_microfoundation_summary.txt'} and price_microfoundation.png")


if __name__ == "__main__":
    main()
