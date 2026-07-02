"""
d09_dynamic_gefeedback.py
-------------------------
The dynamic transition WITH the AR-style price feedback wired into the time
loop -- the GE-dynamic close of manuscript sec. 5. Migrated from
run_dynamic_gefeedback.py onto the _interface layer; supersedes it. At each
step the price responds to the current adoption and labour allocation,

    Delta ln Pi_t(r) = (1/sigma) [ ln(1 - a_t(r)) - ln( n_t(r) / n_0(r) ) ],

applied to the estimated baseline with a one-step lag and a mild relaxation.
The updated price feeds the adoption gate and the place value, so the wage
adjustment shapes the whole trajectory, not just the end state. sigma is a
parameter of run() (d07 sweeps it); the binding block mirrors run_dynamic's
match-allocated law verbatim -- run_dynamic remains the fixed-price
integrator, this module is its feedback-augmented counterpart.

Pre-registered hypotheses (written before the migration run):
  (H1) Damping. The feedback cuts the unbound-mass (mismatch) peak by about
       64% at sigma = 3 (0.0044 -> 0.0016) at unchanged timing, and the
       price level falls monotonically as adoption rises.
  (H2) Statics meet dynamics. The dynamic end state reproduces the static
       fixed point of d08 within rounding: price level about -0.285 against
       d08's -0.261, labour share 0.739 against 0.733 (both on the L0
       basis, which isolates the price channel).

Every number writes to experiment/results/ and is asserted against the
frozen baseline (5% relative for magnitudes).

Usage: python experiment/d09_dynamic_gefeedback.py   (about 2 minutes)
"""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.regime import regime


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

RELAX = 0.3

# Frozen baseline (this machine, this calibration; layer mobility reference).
BASE = {"u_peak_off": 0.004358, "u_peak_on": 0.001618,
        "level_end": -0.2852, "ls_end": 0.7386, "ls_off": 0.5822}
REL = 0.05


def run(price_feedback, sigma=3.0, layer=None, T_max=20.0, dt=0.2, theta_L=3.0,
        rho=0.5, theta_abs=3.0, lam_over=1.0, match_beta=3.0, T_shock=5.0):
    """One transition, fixed price (price_feedback=False) or with the
    feedback in the loop. Returns (rec, end labour share, end price level)."""
    if layer is None:
        layer = iface.load_static_layer()
    inp, L0, eq, ell = layer.inp, layer.L0, layer.eq, layer.ell
    g0_grid, g0_task = layer.g0_grid, layer.g0_task
    A_final = layer.tech.A_K
    layer.set_maturity(0.0)
    kappa, c = layer.kappa, layer.c
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
        GammaD = float(np.sum(dyn.L[:dyn.n0] * eq.D_o))
        dGamma = 0.0 if GammaD_prev is None else max((GammaD - GammaD_prev) / dt, 0.0)
        GammaD_prev = GammaD
        sdot = rd.GAMMA * dGamma * eq.g_hat * (1.0 - a_grid)
        dyn.U = dyn.U + dt * sdot
        # match-allocated, size-rate-limited binding (run_dynamic's law, verbatim)
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
            tgt[okc] = (1.0 / sigma) * (np.log(1.0 - aclip[okc]) - np.log(ratio))
            dlnPi = (1.0 - RELAX) * dlnPi + RELAX * tgt
            eq.pi_cell = pi_cell_0 * np.exp(dlnPi)
            eq.pi_task = pi_task_0 * np.exp(dlnPi[eq.cell_of])
            dyn.Pi = eq.pi_cell

        rec["t"].append(t); rec["A_K"].append(A_K)
        rec["U_tot"].append(float(np.sum(dyn.U * dyn.area)))
        rec["B_tot"].append(float(np.sum(dyn.B * dyn.area)))
        rec["dlnPi_mean"].append(float(dlnPi[okc].mean()))

    # restore the shared eq's price field for subsequent runs on this layer
    eq.pi_cell = pi_cell_0
    eq.pi_task = pi_task_0
    dyn.Pi = eq.pi_cell

    # end-state labour share via regime with the evolved price field (L0 basis)
    adj = iface.AdjustedField(inp.field, dlnPi, inp.grid)
    reg = regime(replace(inp, field=adj), layer.tech, L0, rd.R, rd.TAU,
                 rd.GAMMA, ell, rd.BETA, wedge=None, survival=True)
    rec = {k: np.array(v) for k, v in rec.items()}
    return rec, reg["labor_share"], float(dlnPi[okc].mean())


def main():
    lines = []

    def emit(s=""):
        lines.append(s)          # write_summary echoes once at the end

    layer = iface.load_static_layer()
    emit("d09: dynamic transition, fixed price vs AR price feedback in the loop")
    emit(f"sigma 3, relax {RELAX:g}; mobility reference kappa {layer.kappa:.3f}, "
         f"c {layer.c:.3f} (layer rule, A_K = 0)")
    emit("")
    rec0, ls0, _ = run(price_feedback=False, layer=layer)
    rec1, ls1, lvl1 = run(price_feedback=True, layer=layer)

    def at(rec, tt):
        i = int(np.argmin(np.abs(rec["t"] - tt)))
        return rec["U_tot"][i], rec["B_tot"][i], rec["dlnPi_mean"][i]

    emit(f"  {'t':>4}  {'A_K':>5} | {'U_tot fix':>9} {'U_tot fb':>9} | "
         f"{'B_tot fix':>9} {'B_tot fb':>9} | {'dlnPi_mean':>10}")
    for tt in (1, 2, 3, 5, 8, 12, 20):
        u0, b0, _ = at(rec0, tt)
        u1, b1, lv = at(rec1, tt)
        ak = rec0["A_K"][int(np.argmin(np.abs(rec0["t"] - tt)))]
        emit(f"  {tt:>4}  {ak:>5.2f} | {u0:>9.4f} {u1:>9.4f} | {b0:>9.4f} {b1:>9.4f} | {lv:>+10.4f}")

    u_peak0, u_peak1 = float(rec0["U_tot"].max()), float(rec1["U_tot"].max())
    damping = 1.0 - u_peak1 / u_peak0
    lvl_path = rec1["dlnPi_mean"]
    emit("")
    emit(f"(H1) U_tot peak: fixed {u_peak0:.4f} (t={rec0['t'][rec0['U_tot'].argmax()]:.1f}) "
         f"-> feedback {u_peak1:.4f} (t={rec1['t'][rec1['U_tot'].argmax()]:.1f}); "
         f"damping {100 * damping:.1f}%")
    emit(f"  end B_tot: fixed {rec0['B_tot'][-1]:.4f} -> feedback {rec1['B_tot'][-1]:.4f}")
    emit(f"(H2) statics meet dynamics (L0 basis):")
    emit(f"  end price level  {lvl1:+.4f}   (d08 static fixed point -0.2607)")
    emit(f"  end labour share {ls1:.4f}   (fixed-Pi {ls0:.4f}; d08 static fixed point 0.7334)")

    # ---- asserts against the frozen baseline ----
    got = {"u_peak_off": u_peak0, "u_peak_on": u_peak1,
           "level_end": lvl1, "ls_end": ls1, "ls_off": ls0}
    for k, v in got.items():
        assert abs(v - BASE[k]) <= REL * abs(BASE[k]), \
            f"{k} drifted: {v:.4f} vs frozen {BASE[k]:.4f}"
    assert np.all(np.diff(lvl_path[: int(np.argmin(lvl_path)) + 1]) <= 1e-9), \
        "price level not monotone down to its trough"
    assert abs(lvl1 - (-0.2607)) < 0.05 and abs(ls1 - 0.7334) < 0.02, \
        "dynamic end state no longer reproduces the static fixed point"

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(iface.RESULTS / "dynamic_gefeedback.csv", "w") as fh:
        fh.write("t,A_K,U_fix,U_fb,B_fix,B_fb,dlnPi_mean\n")
        for i in range(len(rec0["t"])):
            fh.write(f"{rec0['t'][i]:.2f},{rec0['A_K'][i]:.4f},"
                     f"{rec0['U_tot'][i]:.6f},{rec1['U_tot'][i]:.6f},"
                     f"{rec0['B_tot'][i]:.6f},{rec1['B_tot'][i]:.6f},"
                     f"{rec1['dlnPi_mean'][i]:.4f}\n")
    lines += ["", "all frozen-baseline asserts passed."]
    iface.write_summary("dynamic_gefeedback", lines)
    return got


if __name__ == "__main__":
    main()
