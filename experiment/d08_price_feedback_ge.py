"""
d08_price_feedback_ge.py
------------------------
The static close of the AR-style price feedback (manuscript sec. 5): the sign
of the closure, the one-pass extremes, the GE fixed point on both allocation
bases, and the contraction sufficient condition that guarantees the fixed
point is reachable by damped iteration. Migrated from price_feedback_ge.py
(fixed point) and price_feedback.py (one-pass sign comparison) onto the
_interface layer; supersedes both.

The feedback law, applied to the estimated baseline (level from data, change
from the shock):

    Delta ln Pi(r) = (1/sigma) [ ln(1 - a(Pi)) - ln( nL(Pi) / nL0 ) ].

Pre-registered hypotheses (written before the migration run):
  (H1) Sign. The scarcity reading (automation as a supply shock, mean-zeroed)
       is destabilising and collapses the one-pass labour share toward 1/3;
       the ar_level reading (demand shock, level free to fall) lifts it
       toward 0.83. The fixed-price 0.58 sits between. (Frozen from the
       superseded scripts; the layer's mobility reference differs by 2%,
       so the values may shift within the 5% tolerance but not the ordering.)
  (H2) Fixed point. Damped iteration (damp 1/2) at sigma = 3 converges to an
       interior point: price level about -0.26, labour share 0.73 on the L0
       basis and 0.78 on the re-solved (paper) basis, D_o cut by about 44%.
  (H3) Contraction. The gate slope is d ln(1-a)/d ln Pi = -a (R/Pi)/tau, so
       the undamped map slope is bounded by
       (1/sigma) (sup_r a R/(Pi tau) + crowding elasticity), and damped
       iteration with weight d converges iff the stabilising slope K
       satisfies K < 2/d - 1 (= 3 at d = 1/2). Predictions: the analytic
       gate sup at the fixed point is about 1.6 and bounds the empirical
       spectral radius (power iteration on the full map, crowding included)
       from above; both sit below 3 with a factor-two margin.

Every number writes to experiment/results/ and is asserted against the
frozen baseline (0.02 for correlations, 5% relative for magnitudes).

Usage: python experiment/d08_price_feedback_ge.py   (about 2 minutes)
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
from model.regime import regime


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")

SIGMA = 3.0
DAMP, TOL, MAXIT = 0.5, 4e-3, 60
POWER_MAXIT, POWER_TOL, POWER_EPS = 40, 1e-3, 1e-3

# Frozen baseline (this machine, this calibration; layer mobility reference).
BASE = {
    "ls_fix_L0": 0.5822, "ls_fix_out": 0.6254,
    "ls_scarcity": 0.3330, "ls_arlevel_1pass": 0.8279,
    "level_star": -0.2607, "ls_star_L0": 0.7334, "ls_star_out": 0.7785,
    "Do_fix": 0.2786, "Do_star": 0.1561,
    "K_gate_sup": 1.644, "K_emp": 1.594,
}
REL = 0.05  # magnitude tolerance


def _ctx(layer):
    """Shared per-run context: baseline equilibrium, densities, masks."""
    inp, L0, grid = layer.inp, layer.L0, layer.inp.grid
    eq0 = Equilibrium(inp, layer.tech, layer.R, layer.tau, layer.gamma,
                      layer.ell, layer.beta, wedge=None, survival=True)
    eq0.L0 = L0
    rw, bw, cl, area = eq0.row_of, eq0.b_w, eq0.cell_of, eq0.area
    nL0 = np.bincount(cl, weights=L0[rw] * bw, minlength=grid.xi.size) / area
    return dict(inp=inp, L0=L0, grid=grid, eq0=eq0, rw=rw, bw=bw, cl=cl,
                area=area, nL0=nL0, occ_cell=nL0 > 0)


def _equilibrium_at(layer, ctx, d):
    """Equilibrium and re-solved allocation at the adjusted price field."""
    inp_k = replace(ctx["inp"], field=iface.AdjustedField(ctx["inp"].field, d, ctx["grid"]))
    eqk = Equilibrium(inp_k, layer.tech, layer.R, layer.tau, layer.gamma,
                      layer.ell, layer.beta, wedge=None, survival=True)
    eqk.L0 = ctx["L0"]
    Lk = eqk.solve(layer.c, layer.kappa).L
    return inp_k, eqk, Lk


def feedback_target(layer, ctx, d):
    """One undamped evaluation of the map T(d); returns (target, eqk)."""
    _, eqk, Lk = _equilibrium_at(layer, ctx, d)
    ak = np.clip(eqk.a_grid, 0.0, 0.999)
    nLk = np.bincount(ctx["cl"], weights=Lk[ctx["rw"]] * ctx["bw"],
                      minlength=ctx["grid"].xi.size) / ctx["area"]
    oc = ctx["occ_cell"]
    tgt = np.zeros(ctx["grid"].xi.size)
    ratio = (nLk[oc] + 1e-12) / (ctx["nL0"][oc] + 1e-12)
    tgt[oc] = (1.0 / SIGMA) * (np.log(1.0 - ak[oc]) - np.log(ratio))
    return tgt, eqk


def ge_fixed_point(layer, ctx, sigma, damp, tol=TOL, maxit=MAXIT):
    """Damped iteration of the feedback map at this sigma. Returns the
    adjustment field d, convergence flag, and iteration count. Divergence
    (per H3, when the stabilising slope exceeds 2/damp - 1) is reported,
    not raised: d07 uses the flag to locate the boundary."""
    global SIGMA
    sig_save, SIGMA = SIGMA, sigma
    d = np.zeros(ctx["grid"].xi.size)
    oc = ctx["occ_cell"]
    converged, iters = False, maxit
    with np.errstate(over="ignore"):
        for it in range(maxit):
            tgt, _ = feedback_target(layer, ctx, d)
            d_new = (1.0 - damp) * d + damp * tgt
            change = np.max(np.abs((d_new - d)[oc]))
            d = d_new
            if not np.isfinite(change) or change > 1e3:   # blown up: diverged
                iters = it + 1
                break
            if change < tol:
                converged, iters = True, it + 1
                break
    SIGMA = sig_save
    return d, converged, iters


def contraction_diagnostics(layer, ctx, d_star, emit, power_maxit=POWER_MAXIT,
                            power_tol=POWER_TOL):
    """H3: analytic gate slope at the fixed point, and power iteration on the
    Jacobian of the undamped map (crowding coupling included)."""
    tgt_star, eq_star = feedback_target(layer, ctx, d_star)
    oc = ctx["occ_cell"]
    a_star = np.clip(eq_star.a_grid, 0.0, 0.999)
    Kg = (a_star * (layer.R / eq_star.pi_cell) / layer.tau / SIGMA)[oc]
    w_mass = (ctx["nL0"] * ctx["area"])[oc]
    K_sup = float(Kg.max())
    K_mean = float(np.average(Kg, weights=w_mass))
    emit(f"  gate slope K_gate = (1/sigma) a R/(Pi tau) at the fixed point:")
    emit(f"    sup {K_sup:.3f}, mass-weighted mean {K_mean:.3f}, "
         f"99th pct {np.percentile(Kg, 99):.3f}")

    rng = np.random.default_rng(0)
    v = rng.standard_normal(ctx["grid"].xi.size)
    v[~oc] = 0.0
    v /= np.linalg.norm(v)
    lam = lam_prev = 0.0
    for k in range(power_maxit):
        Tv, _ = feedback_target(layer, ctx, d_star + POWER_EPS * v)
        Jv = (Tv - tgt_star) / POWER_EPS
        lam = float(np.linalg.norm(Jv))
        v = Jv / (lam if lam > 0 else 1.0)
        if k > 0 and abs(lam - lam_prev) < power_tol:
            break
        lam_prev = lam
    emit(f"  power iteration on the full map ({k + 1} its): "
         f"spectral radius K = {lam:.3f}")
    thresh = 2.0 / DAMP - 1.0
    emit(f"  sufficient condition at damp {DAMP:g}: K < 2/damp - 1 = {thresh:g}"
         f"  ->  {'satisfied' if lam < thresh else 'VIOLATED'}"
         f"  (margin x{thresh / lam:.1f}; implied max damp {2.0 / (lam + 1.0):.2f})")
    return K_sup, K_mean, lam


def main():
    lines = []

    def emit(s=""):
        lines.append(s)          # write_summary echoes once at the end

    layer = iface.load_static_layer()
    ctx = _ctx(layer)
    inp, L0, grid, eq0 = ctx["inp"], ctx["L0"], ctx["grid"], ctx["eq0"]
    oc, rw, bw, cl, area, nL0 = (ctx["occ_cell"], ctx["rw"], ctx["bw"],
                                 ctx["cl"], ctx["area"], ctx["nL0"])

    def LS(inp_x, L_x):
        return regime(inp_x, layer.tech, L_x, layer.R, layer.tau, layer.gamma,
                      layer.ell, layer.beta, wedge=None, survival=True)

    # ---- fixed-price baselines ----
    reg_fix_L0 = LS(inp, L0)
    Lpost = eq0.solve(layer.c, layer.kappa).L
    reg_fix_out = LS(inp, Lpost)
    ls_fix_L0 = reg_fix_L0["labor_share"]
    ls_fix_out = reg_fix_out["labor_share"]
    emit("d08: GE price feedback -- sign, one-pass extremes, fixed point, contraction")
    emit(f"sigma {SIGMA:g}, damp {DAMP:g}; mobility reference kappa {layer.kappa:.3f}, "
         f"c {layer.c:.3f} (layer rule, A_K = 0)")
    emit(f"fixed-Pi baseline: labour share L0 {ls_fix_L0:.4f} | out.L {ls_fix_out:.4f}; "
         f"D_o mean {reg_fix_out['D_o'].mean():.4f}")
    emit("")

    # ---- (H1) the sign: one-pass extremes ----
    a0 = np.clip(eq0.a_grid, 0.0, 0.999)
    nLpost = np.bincount(cl, weights=Lpost[rw] * bw, minlength=grid.xi.size) / area
    nHpost = np.bincount(cl, weights=Lpost[rw] * bw * (1.0 - eq0.a_task),
                         minlength=grid.xi.size) / area
    d_sc = np.zeros(grid.xi.size)
    d_sc[oc] = -(1.0 / SIGMA) * (np.log(nHpost[oc] + 1e-12) - np.log(nL0[oc] + 1e-12))
    d_sc[oc] -= d_sc[oc].mean()                       # scarcity: redistributive
    d_ar = np.zeros(grid.xi.size)
    ratio = (nLpost[oc] + 1e-12) / (nL0[oc] + 1e-12)
    d_ar[oc] = (1.0 / SIGMA) * (np.log(1.0 - a0[oc]) - np.log(ratio))
    ls_sc = LS(replace(inp, field=iface.AdjustedField(inp.field, d_sc, grid)), L0)["labor_share"]
    ls_ar = LS(replace(inp, field=iface.AdjustedField(inp.field, d_ar, grid)), L0)["labor_share"]
    emit("(H1) one-pass extremes, L0 basis:")
    emit(f"  scarcity (supply reading, destabilising)   labour share {ls_sc:.4f}")
    emit(f"  ar_level (demand reading, level falls)     labour share {ls_ar:.4f}")
    emit(f"  fixed price                                labour share {ls_fix_L0:.4f}")
    emit("")

    # ---- (H2) the fixed point ----
    d_star, converged, iters = ge_fixed_point(layer, ctx, SIGMA, DAMP)
    assert converged, "fixed point did not converge at the baseline sigma/damp"
    inp_star = replace(inp, field=iface.AdjustedField(inp.field, d_star, grid))
    reg_star_L0 = LS(inp_star, L0)
    _, eq_star, L_star = _equilibrium_at(layer, ctx, d_star)
    reg_star_out = LS(inp_star, L_star)
    level = float(d_star[oc].mean())
    a_star_grid = np.clip(eq_star.a_grid, 0.0, 0.999)
    emit(f"(H2) fixed point at sigma {SIGMA:g}: converged in {iters} damped steps")
    emit(f"  Delta ln Pi: level {level:+.4f}, sd {d_star[oc].std():.4f}; "
         f"corr with operated share a {spearmanr(d_star[oc], a_star_grid[oc])[0]:+.3f}")
    emit(f"  labour share  L0 basis   {ls_fix_L0:.4f} -> {reg_star_L0['labor_share']:.4f}")
    emit(f"  labour share  out basis  {ls_fix_out:.4f} -> {reg_star_out['labor_share']:.4f}"
         f"   (price and allocation both equilibrate; the paper basis)")
    emit(f"  D_o mean      {reg_fix_out['D_o'].mean():.4f} -> {reg_star_out['D_o'].mean():.4f}"
         f"  ({100 * (reg_star_out['D_o'].mean() / reg_fix_out['D_o'].mean() - 1):+.0f}%)")
    emit("")

    # ---- (H3) the contraction condition ----
    emit("(H3) contraction sufficient condition:")
    K_sup, K_mean, K_emp = contraction_diagnostics(layer, ctx, d_star, emit)

    # ---- asserts against the frozen baseline ----
    got = {"ls_fix_L0": ls_fix_L0, "ls_fix_out": ls_fix_out,
           "ls_scarcity": ls_sc, "ls_arlevel_1pass": ls_ar,
           "level_star": level, "ls_star_L0": reg_star_L0["labor_share"],
           "ls_star_out": reg_star_out["labor_share"],
           "Do_fix": float(reg_fix_out["D_o"].mean()),
           "Do_star": float(reg_star_out["D_o"].mean()),
           "K_gate_sup": K_sup, "K_emp": K_emp}
    for k, v in got.items():
        assert abs(v - BASE[k]) <= REL * abs(BASE[k]), \
            f"{k} drifted: {v:.4f} vs frozen {BASE[k]:.4f}"
    assert K_sup >= K_emp - 0.05, "gate sup no longer bounds the spectral radius"
    assert K_emp < 2.0 / DAMP - 1.0, "contraction condition violated at baseline"

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(iface.RESULTS / "price_feedback_ge.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in got.items():
            fh.write(f"{k},{v:.4f}\n")
    lines += ["", "all frozen-baseline asserts passed."]
    iface.write_summary("price_feedback_ge", lines)
    return got


if __name__ == "__main__":
    main()
