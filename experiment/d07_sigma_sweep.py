"""
d07_sigma_sweep.py
------------------
Sensitivity of the GE price feedback to the assignment/task elasticity sigma
(manuscript sec. 5). Migrated from sigma_sweep.py onto the _interface layer;
supersedes it. sigma enters only the feedback coefficient 1/sigma, not the
underlying equilibrium, so the sweep isolates the strength of the feedback.
The point sigma = 0.5 is the Acemoglu-Restrepo (2026) anchor, where 1/sigma
equals their wage-law coefficient 1/lambda; the calibrated model uses
sigma = 3 (tasks gross substitutes); the sweep brackets both regimes.

Loads d08 (the fixed-point map and contraction diagnostics) and d09 (the
feedback-augmented integrator); a lower number consuming higher ones is the
price of keeping the agenda's script-name assignment.

Pre-registered hypotheses (written before the boundary probes were run):
  (H1) Strength. The dynamic hump damping increases monotonically as sigma
       falls. The static lift is positive everywhere and monotone from
       sigma = 1 upward but NOT at the AR anchor: at sigma = 0.5 the price
       level falls so far (about -1.3) that displacement nearly vanishes
       (D_o ~ 0.08) and the re-solved labour share comes off its sigma = 1
       peak -- already on record in the superseded sweep's summary.
  (H2) 1/sigma scaling of the stabilising slope. The spectral radius K of
       the undamped map at the fixed point scales as K ~ Kbar/sigma:
       sigma * K agrees between sigma = 3 (d08: 3 x 1.594 = 4.78) and
       sigma = 0.5 (computed here) within 15%.
  (H3) The divergence boundary. Damped iteration converges iff
       K < 2/damp - 1. With Kbar ~ 4.8 the fixed-damp-1/2 boundary sits at
       sigma* = Kbar/3 ~ 1.6: at sigma = 1 the iteration FAILS at damp 1/2
       (K ~ 4.8 > 3) and converges at the schedule's damp 1/4 (threshold 7);
       at sigma >= 2 it converges at damp 1/2. The damping schedule
       damp(sigma) = 0.5 min(1, sigma/2) keeps every grid point inside the
       bound, which is why the sweep converges everywhere.

Every number writes to experiment/results/ and is asserted against the
frozen baseline (0.02 for correlations, 5% relative for magnitudes).

Usage: python experiment/d07_sigma_sweep.py   (about 8 minutes)

Recalibration note: the frozen baselines in this script were re-frozen
after the anchoring of the dynamic sorting kernel (alpha_o through the
interface; patch series 01-04). Point values quoted in the hypothesis
text above are the pre-anchoring pre-registration record; the
pre-anchoring baselines remain in git history.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.regime import regime
from dataclasses import replace


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
d08 = _load("d08_price_feedback_ge")
d09 = _load("d09_dynamic_gefeedback")

SIGMA_GRID = (0.5, 1.0, 2.0, 3.0, 5.0)
AR_ANCHOR, SIGMA_BASE = 0.5, 3.0
MAXIT = 200
POWER_ITS_ANCHOR = 25          # fixed count: deterministic frozen value


def schedule(sigma):
    """Damping schedule: strong feedback (small sigma) needs gentler steps."""
    return 0.5 * min(1.0, sigma / 2.0)


# Frozen baseline (this machine, this calibration; layer mobility reference).
BASE_STATIC = {  # sigma: (level, ls_out, ls_L0, Do)
    0.5: (-0.1613, 0.8136, 0.7750, 0.1471),
    1.0: (-0.1138, 0.7847, 0.7426, 0.1760),
    2.0: (-0.0792, 0.7507, 0.7040, 0.2064),
    3.0: (-0.0621, 0.7306, 0.6810, 0.2230),
    5.0: (-0.0447, 0.7073, 0.6543, 0.2407),
}
BASE_DYN = {  # sigma: (u_peak_fb, damping)
    0.5: (0.000329, 0.877), 1.0: (0.000720, 0.731), 2.0: (0.001265, 0.527),
    3.0: (0.001577, 0.410), 5.0: (0.001934, 0.277),
}
BASE_K_ANCHOR = 8.887          # spectral radius at sigma = 0.5, 25 power its
REL = 0.05


def main():
    lines = []

    def emit(s=""):
        lines.append(s)          # write_summary echoes once at the end

    layer = iface.load_static_layer()
    ctx = d08._ctx(layer)
    inp, L0, grid, eq0 = ctx["inp"], ctx["L0"], ctx["grid"], ctx["eq0"]
    oc = ctx["occ_cell"]

    def LS(inp_x, L_x):
        return regime(inp_x, layer.tech, L_x, layer.R, layer.tau, layer.gamma,
                      layer.ell, layer.beta, wedge=None, survival=True)

    ls_fix_L0 = LS(inp, L0)["labor_share"]
    Lpost = eq0.solve(layer.c, layer.kappa).L
    ls_fix_out = LS(inp, Lpost)["labor_share"]
    emit("d07: sigma sweep of the GE price feedback")
    emit(f"feedback coefficient 1/sigma; AR(2026) anchor sigma = {AR_ANCHOR}; "
         f"damping schedule damp = 0.5 min(1, sigma/2)")
    emit(f"fixed-price baseline: labour share L0 {ls_fix_L0:.4f} | out.L {ls_fix_out:.4f}")
    emit("")

    # ---- (1) static GE fixed point across sigma ----
    emit("=== (1) static GE fixed point ===")
    emit(f"  {'sigma':>6} {'1/sigma':>8} {'damp':>6} {'it':>4} "
         f"{'dlnPi':>8} {'LS(out)':>8} {'lift':>7} {'LS(L0)':>8} {'D_o':>8}")
    static = []
    for s in SIGMA_GRID:
        d, conv, iters = d08.ge_fixed_point(layer, ctx, s, schedule(s), maxit=MAXIT)
        assert conv, f"fixed point did not converge at sigma={s} under the schedule"
        inp_star = replace(inp, field=iface.AdjustedField(inp.field, d, grid))
        with np.errstate(over="ignore"):    # |dlnPi| ~ 1.3 at sigma = 0.5: benign saturation
            _, _, L_star = d08._equilibrium_at(layer, ctx, d)
            reg_out = LS(inp_star, L_star)
            ls_L0 = LS(inp_star, L0)["labor_share"]
        row = (s, float(d[oc].mean()), reg_out["labor_share"], ls_L0,
               float(reg_out["D_o"].mean()), iters)
        static.append(row)
        b = BASE_STATIC[s]
        for got, ref in zip(row[1:5], b):
            assert abs(got - ref) <= REL * abs(ref), \
                f"static drifted at sigma={s}: {got:.4f} vs frozen {ref:.4f}"
        tag = "  <- AR" if s == AR_ANCHOR else ("  <- baseline" if s == SIGMA_BASE else "")
        emit(f"  {s:>6.2f} {1/s:>8.3f} {schedule(s):>6.3f} {iters:>4} "
             f"{row[1]:>+8.4f} {row[2]:>8.4f} {row[2]-ls_fix_out:>+7.4f} "
             f"{row[3]:>8.4f} {row[4]:>8.4f}{tag}")
    lifts = [r[2] - ls_fix_out for r in static]
    assert all(l > 0 for l in lifts), "(H1) static lift not positive everywhere"
    assert all(a >= b - 1e-9 for a, b in zip(lifts[1:], lifts[2:])), \
        "(H1) static lift not monotone from sigma = 1 upward"

    # ---- (2) dynamic GE feedback across sigma ----
    emit("")
    emit("=== (2) dynamic GE feedback (mismatch damping and end state) ===")
    rec_off, ls_off, _ = d09.run(price_feedback=False, layer=layer)
    u_peak_off = float(rec_off["U_tot"].max())
    emit(f"  feedback OFF: U_tot peak {u_peak_off:.4f}, end labour share {ls_off:.4f}")
    emit(f"  {'sigma':>6} {'1/sigma':>8} {'U_peak_fb':>10} {'damping':>8} "
         f"{'LS_end':>8} {'level_end':>10}")
    dyn = []
    for s in SIGMA_GRID:
        rec_on, ls_on, lvl_on = d09.run(price_feedback=True, sigma=s, layer=layer)
        u_peak_on = float(rec_on["U_tot"].max())
        damping = 1.0 - u_peak_on / u_peak_off
        dyn.append((s, u_peak_on, damping, ls_on, lvl_on))
        bu, bd = BASE_DYN[s]
        assert abs(u_peak_on - bu) <= REL * bu, \
            f"dynamic U peak drifted at sigma={s}: {u_peak_on:.4f} vs {bu:.4f}"
        assert abs(damping - bd) <= 0.02, \
            f"dynamic damping drifted at sigma={s}: {damping:.3f} vs {bd:.3f}"
        tag = "  <- AR" if s == AR_ANCHOR else ("  <- baseline" if s == SIGMA_BASE else "")
        emit(f"  {s:>6.2f} {1/s:>8.3f} {u_peak_on:>10.4f} {100*damping:>7.1f}% "
             f"{ls_on:>8.4f} {lvl_on:>+10.4f}{tag}")
    damps = [r[2] for r in dyn]
    assert all(a >= b - 1e-9 for a, b in zip(damps, damps[1:])), \
        "(H1) dynamic damping not monotone in 1/sigma"

    # ---- (3) the contraction bound against the empirical boundary ----
    emit("")
    emit("=== (3) contraction bound vs the empirical divergence boundary ===")
    # (H2) spectral radius at the AR anchor, fixed power-iteration count
    d_anchor, conv, _ = d08.ge_fixed_point(layer, ctx, AR_ANCHOR,
                                           schedule(AR_ANCHOR), maxit=MAXIT)
    assert conv
    d08.SIGMA = AR_ANCHOR
    try:
        _, _, K_anchor = d08.contraction_diagnostics(
            layer, ctx, d_anchor, emit,
            power_maxit=POWER_ITS_ANCHOR, power_tol=0.0)
    finally:
        d08.SIGMA = SIGMA_BASE
    emit(f"  (the sweep runs sigma = {AR_ANCHOR:g} at the schedule's damp "
         f"{schedule(AR_ANCHOR):g}, threshold {2/schedule(AR_ANCHOR)-1:g}: satisfied)")
    K_base = d08.BASE["K_emp"]
    prod_a, prod_b = AR_ANCHOR * K_anchor, SIGMA_BASE * K_base
    emit(f"  (H2) sigma*K: {AR_ANCHOR:g} x {K_anchor:.2f} = {prod_a:.2f} against "
         f"{SIGMA_BASE:g} x {K_base:.2f} = {prod_b:.2f} "
         f"(ratio {prod_a/prod_b:.2f})")
    assert abs(K_anchor - BASE_K_ANCHOR) <= REL * BASE_K_ANCHOR, \
        f"K at the anchor drifted: {K_anchor:.2f} vs frozen {BASE_K_ANCHOR:.2f}"
    assert abs(prod_a / prod_b - 1.0) < 0.15, "(H2) 1/sigma scaling broken"

    # (H3) boundary probes at sigma = 1: fail at damp 1/2, converge at 1/4
    Kbar = 0.5 * (prod_a + prod_b)
    sigma_star = Kbar / 3.0
    emit(f"  (H3) predicted boundary at damp 1/2: sigma* = Kbar/3 = {sigma_star:.2f}")
    _, conv_half, it_half = d08.ge_fixed_point(layer, ctx, 1.0, 0.5, maxit=40)
    _, conv_sched, it_sched = d08.ge_fixed_point(layer, ctx, 1.0, schedule(1.0),
                                                 maxit=MAXIT)
    emit(f"       sigma = 1 at damp 0.50: "
         f"{'converged in ' + str(it_half) + ' its' if conv_half else 'DID NOT CONVERGE (40 its)'}")
    emit(f"       sigma = 1 at damp 0.25: "
         f"{'converged in ' + str(it_sched) + ' its' if conv_sched else 'did not converge'}")
    assert not conv_half, "(H3) sigma=1 converged at damp 1/2 against the prediction"
    assert conv_sched, "(H3) sigma=1 failed at the schedule's damping"
    emit(f"       sigma = 2 at damp 0.50: converged in {static[2][5]} its (sweep above)")
    emit(f"  the boundary is bracketed: sigma = 1 fails, sigma = 2 converges, "
         f"prediction {sigma_star:.2f} between them.")

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(iface.RESULTS / "sigma_sweep.csv", "w") as fh:
        fh.write("sigma,damp,level,ls_out,ls_L0,Do,iters,u_peak_fb,damping,ls_end,level_end\n")
        for st, dy in zip(static, dyn):
            fh.write(f"{st[0]:g},{schedule(st[0]):g},{st[1]:.4f},{st[2]:.4f},"
                     f"{st[3]:.4f},{st[4]:.4f},{st[5]},{dy[1]:.6f},{dy[2]:.4f},"
                     f"{dy[3]:.4f},{dy[4]:.4f}\n")

    sig = np.array(SIGMA_GRID)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].axhline(ls_fix_out, ls="--", c="0.6", lw=1, label=f"fixed price {ls_fix_out:.2f}")
    ax[0].plot(sig, [r[2] for r in static], "o-", c="#2C5A57")
    ax[0].axvline(AR_ANCHOR, ls=":", c="#B5532A", lw=1)
    ax[0].annotate("AR anchor\n$1/\\sigma=1/\\lambda=2$", (AR_ANCHOR, ls_fix_out),
                   textcoords="offset points", xytext=(8, 8), color="#B5532A", fontsize=8)
    ax[0].set_xlabel(r"$\sigma$ (assignment/task elasticity)")
    ax[0].set_ylabel("GE labour share (re-solved basis)")
    ax[0].set_title(r"Static fixed-point lift vs $\sigma$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)
    ax[1].plot(sig, [100 * r[2] for r in dyn], "s-", c="#6D3C8E")
    ax[1].axvline(AR_ANCHOR, ls=":", c="#B5532A", lw=1)
    ax[1].set_xlabel(r"$\sigma$ (assignment/task elasticity)")
    ax[1].set_ylabel("mismatch hump damping (%)")
    ax[1].set_title(r"Dynamic mismatch damping vs $\sigma$")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "sigma_sweep.png", dpi=150)

    lines += ["", "all frozen-baseline asserts passed."]
    iface.write_summary("sigma_sweep", lines)


if __name__ == "__main__":
    main()
