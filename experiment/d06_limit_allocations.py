"""
d06_limit_allocations.py
------------------------
The destination result as a homotopy between two proved endpoints
(manuscript secs. 3.4 and 4.3). theta_L = 1 throughout (the d03
decomposition: the destination is carried by theta_abs alone); theta_abs is
swept against the fixed T_shock = 5 year diffusion.

Proposition 1 (fast-absorption limit, theta_abs/T_shock -> 0): no cap ever
binds, each step's seeding is absorbed in full at the time-invariant claim
shares w_o(r) = e_o^beta_m / sum e^beta_m, so the final allocation is
exactly R_o = sum_r w_o(r) s(r), s the cumulative surviving seed density.
Checked here EXACTLY (machine precision), not by correlation: at
theta_abs = 0.05 the bound field B(r) is the cumulative seed, and
||R - w @ s||_1 / S < 1e-12.

Proposition 2 (slow-absorption limit, theta_abs/T_shock -> inf): every
claim share is strictly positive, so once caps are small enough they all
bind while any unbound mass remains; per-step absorption is then
proportional to M_o. Endogenous task-mass growth preserves the proportions
-- dr_o/dt = (M_o(0) + r_o)/theta_abs gives r_o proportional to M_o(0), a
common exponential factor -- so the final allocation converges to the
initial size shares, R_o -> S * M_o(0)/M_tot.

The sweep traces the homotopy: corr(R, fast endpoint) falls from +1.00 to
+0.07 and corr(R, size shares) rises from +0.03 to +0.94 as theta_abs runs
from 0.5 to 120; the crossover sits between theta_abs = 10 and 15, two to
three shock windows. The aggregate drain law dU/dt = sdot - M_tot/theta_abs
holds only where all caps bind. The capacity utilisation -- mass absorbed
per step over the aggregate bound (dt/theta_abs) M_tot, recorded in the
loop and read at the peak of the mismatch -- rises from 0.11 at
theta_abs = 2 to 0.56 at 15 and 0.99 at 120: binding is spatially local,
only the crescent's neighbours absorb, which is why the hump outlives the
naive aggregate prediction at intermediate tempos.

Births are disabled (see d02). All numbers write to experiment/results/ and
are asserted against the frozen baseline.

Usage: python experiment/d06_limit_allocations.py   (about 4 minutes)
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
from scipy.stats import pearsonr, spearmanr


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

T_SHOCK, DT, THETA_L = 5.0, 0.2, 1.0
BETA_M = 3.0
SWEEP = (0.5, 2.0, 5.0, 8.0, 10.0, 15.0, 30.0, 60.0, 120.0)

# Frozen baseline (this machine, this calibration), tolerance 0.02.
BASELINE_FAST = {0.5: 1.000, 2.0: 0.900, 5.0: 0.688, 8.0: 0.548, 10.0: 0.479,
                 15.0: 0.358, 30.0: 0.206, 60.0: 0.117, 120.0: 0.070}
BASELINE_SIZE = {0.5: 0.033, 2.0: 0.074, 5.0: 0.203, 8.0: 0.315, 10.0: 0.382,
                 15.0: 0.510, 30.0: 0.726, 60.0: 0.872, 120.0: 0.939}
BASELINE_UTIL = {0.5: 0.002, 2.0: 0.107, 5.0: 0.252, 8.0: 0.370, 10.0: 0.434,
                 15.0: 0.563, 30.0: 0.786, 60.0: 0.943, 120.0: 0.987}


def main():
    layer = iface.load_static_layer()

    def run(theta_abs, T_max):
        dyn, rec, _ = rd.main(theta_L=THETA_L, theta_abs=theta_abs,
                              T_shock=T_SHOCK, T_max=T_max, dt=DT,
                              max_births=0, verbose=False, layer=layer)
        assert rec["U_tot"][-1] < 1e-6, f"U does not drain at theta_abs={theta_abs}"
        return dyn, rec

    # ---- Proposition 1, exact check at the fast endpoint ----
    dyn_f, rec_f = run(0.05, 40.0)
    n0 = dyn_f.n0
    w = dyn_f.FIT[:n0] ** BETA_M
    w = w / w.sum(0)[None, :]
    pred_fast = w @ (dyn_f.B * dyn_f.area)
    r_fast = dyn_f.reinst[:n0]
    rel = float(np.abs(pred_fast - r_fast).sum() / r_fast.sum())
    assert rel < 1e-12, f"Prop 1 exact check failed: {rel:.2e}"
    M0 = dyn_f.original[:n0]
    size_shares = M0 / M0.sum()

    # ---- the homotopy sweep ----
    rows = []
    for ta in SWEEP:
        T_max = 120.0 if ta >= 60.0 else (80.0 if ta >= 30.0 else 60.0)
        dyn, rec = run(ta, T_max)
        r = dyn.reinst[:n0]
        cf = float(pearsonr(r, r_fast)[0])
        cs = float(pearsonr(r, M0)[0])
        css = float(spearmanr(r, M0)[0])
        l1 = float(np.abs(r - r.sum() * size_shares).sum() / r.sum())
        cu = float(rec["cap_util"][int(np.argmax(rec["U_tot"]))])
        assert abs(cf - BASELINE_FAST[ta]) < 0.02, f"fast corr drifted at {ta}: {cf:.3f}"
        assert abs(cs - BASELINE_SIZE[ta]) < 0.02, f"size corr drifted at {ta}: {cs:.3f}"
        assert abs(cu - BASELINE_UTIL[ta]) < 0.02, f"cap util drifted at {ta}: {cu:.3f}"
        rows.append((ta, cf, cs, css, l1, cu, float(np.max(rec["U_tot"]))))
    cfs = [r[1] for r in rows]
    css_ = [r[2] for r in rows]
    l1s = [r[4] for r in rows]
    assert all(a >= b - 1e-9 for a, b in zip(cfs, cfs[1:])), "fast corr not monotone"
    assert all(a <= b + 1e-9 for a, b in zip(css_, css_[1:])), "size corr not monotone"
    assert l1s[-1] < 0.30, "slow-limit L1 not converging"

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(iface.RESULTS / "limit_allocations.csv", "w") as fh:
        fh.write("theta_abs,corr_fast_endpoint,corr_size_pearson,corr_size_spearman,"
                 "L1_to_size_shares,cap_utilisation_at_peak,U_peak\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                              for x in r) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    tas = [r[0] for r in rows]
    ax1.semilogx(tas, cfs, "o-", color="#2C5A57",
                 label="corr with fast endpoint (Prop. 1)")
    ax1.semilogx(tas, css_, "s--", color="#B5532A",
                 label="corr with size shares (Prop. 2)")
    ax1.axvline(T_SHOCK, color="0.35", lw=1, ls="--")
    ax1.set_xlabel(r"$\theta_{\rm abs}$ (years, log scale)")
    ax1.set_ylabel("correlation of final allocation")
    ax1.set_title("The homotopy between the two limits")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    ax2.semilogx(tas, [r[5] for r in rows], "o-", color="#6D3C8E")
    ax2.axhline(1.0, color="0.8", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$\theta_{\rm abs}$ (years, log scale)")
    ax2.set_ylabel("capacity utilisation at the mismatch peak")
    ax2.set_title("Binding is local: the aggregate capacity bound\nis saturated only in the deep limit")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "limit_allocations.png", dpi=150)

    lines = [
        "limit_allocations -- the destination result between two proved endpoints",
        f"theta_L = {THETA_L:g} (destination carried by theta_abs, d03), "
        f"T_shock = {T_SHOCK:g} years, beta_m = {BETA_M:g}, births off",
        "",
        f"Prop 1 exact check at theta_abs = 0.05: ||R - w@s||_1 / S = {rel:.2e}",
        "",
        f"{'theta_abs':>9} {'corr fast':>10} {'corr size':>10} {'(rank)':>8} "
        f"{'L1/S to size':>13} {'cap util':>9} {'U peak':>9}",
    ]
    for r in rows:
        lines.append(f"{r[0]:>9g} {r[1]:>+10.3f} {r[2]:>+10.3f} {r[3]:>+8.3f} "
                     f"{r[4]:>13.3f} {r[5]:>9.3f} {r[6]:>9.5f}")
    lines += [
        "",
        "crossover between theta_abs = 10 and 15: two to three shock windows.",
        "cap util = mass absorbed per step over (dt/theta_abs) M_tot, at the",
        "U peak; its shortfall at intermediate tempos is the spatial locality",
        "of binding (Prop. 2's regime is cap util = 1).",
    ]
    iface.write_summary("limit_allocations", lines)


if __name__ == "__main__":
    main()
