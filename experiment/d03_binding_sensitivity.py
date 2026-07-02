"""
d03_binding_sensitivity.py
--------------------------
Sensitivity of the destination result to the two binding handles, and the
decomposition of the redistribution tempo into its two timescales.

Owns two nameable parts of the argument (manuscript sections 3.3 and 4.2):

(1) beta_m sweep. The claim sharpness beta_m sets how strongly the best match
    dominates the local claim (beta_m = 1: fit-proportional sharing;
    beta_m -> inf: strict best-match assignment). The headline cross-regime
    correlation -- corr of final reinstated mass, gradual theta = 1 vs
    congested theta = 15 -- is recomputed for beta_m in {1, 2, 3, 5, 8}.
    The divergence of destinations survives the full range (correlation stays
    far below one everywhere); the point size depends on the handle,
    rising from +0.26 at beta_m = 1 to +0.61 at beta_m = 8.

(2) Timescale decomposition at beta_m = 3. The regime sweep moves theta_L
    (mobility) and theta_abs (absorption) together; here they are slowed one
    at a time. Slowing only absorption (theta_L = 1, theta_abs = 15)
    reproduces the congested allocation (corr +0.39 with the gradual run,
    against +0.37 for the fully congested); slowing only mobility
    (theta_L = 15, theta_abs = 1) leaves the destination essentially
    unchanged (+1.00). The destination result is carried by the absorption
    timescale alone: it is about how fast occupations can take on new work,
    not how fast workers move.

Births are disabled throughout (see d02). All numbers write to
experiment/results/ and are asserted against the frozen baseline below.

Usage: python experiment/d03_binding_sensitivity.py
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

THETA_GRADUAL, THETA_CONGESTED = 1.0, 15.0
T_SHOCK, T_MAX, DT = 5.0, 60.0, 0.2
BETAS = (1.0, 2.0, 3.0, 5.0, 8.0)

# Frozen baseline (this machine, this calibration), absolute tolerance 0.02.
BASELINE_PEARSON = {1.0: 0.255, 2.0: 0.304, 3.0: 0.371, 5.0: 0.503, 8.0: 0.612}
BASELINE_ABSORB_ONLY = 0.385
BASELINE_MOBILITY_ONLY = 0.996


def main():
    layer = iface.load_static_layer()

    def run(theta_L, theta_abs, beta_m):
        dyn, rec, occ = rd.main(theta_L=theta_L, theta_abs=theta_abs,
                                match_beta=beta_m, T_shock=T_SHOCK,
                                T_max=T_MAX, dt=DT, max_births=0,
                                verbose=False, layer=layer)
        assert rec["U_tot"][-1] < 1e-8, "U does not drain"
        return dyn.reinst[:dyn.n0].copy()

    # (1) beta_m sweep of the cross-regime correlation
    pear, spear = {}, {}
    keep = {}
    for bm in BETAS:
        rg = run(THETA_GRADUAL, THETA_GRADUAL, bm)
        rc = run(THETA_CONGESTED, THETA_CONGESTED, bm)
        pear[bm] = float(pearsonr(rg, rc)[0])
        spear[bm] = float(spearmanr(rg, rc)[0])
        keep[bm] = (rg, rc)
        assert abs(pear[bm] - BASELINE_PEARSON[bm]) < 0.02, \
            f"corr drifted at beta_m={bm}: {pear[bm]:.3f}"

    # (2) timescale decomposition at the reference sharpness
    rg = keep[3.0][0]
    r_absorb = run(THETA_GRADUAL, THETA_CONGESTED, 3.0)   # slow absorption only
    r_mobil = run(THETA_CONGESTED, THETA_GRADUAL, 3.0)    # slow mobility only
    c_absorb = float(pearsonr(rg, r_absorb)[0])
    c_mobil = float(pearsonr(rg, r_mobil)[0])
    assert abs(c_absorb - BASELINE_ABSORB_ONLY) < 0.02, f"decomp drifted: {c_absorb:.3f}"
    assert abs(c_mobil - BASELINE_MOBILITY_ONLY) < 0.005, f"decomp drifted: {c_mobil:.3f}"

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(iface.RESULTS / "binding_sensitivity.csv",
               np.column_stack([BETAS, [pear[b] for b in BETAS],
                                [spear[b] for b in BETAS]]),
               delimiter=",", header="beta_m,pearson,spearman", comments="")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(BETAS, [pear[b] for b in BETAS], "o-", color="#2C5A57", label="Pearson")
    ax.plot(BETAS, [spear[b] for b in BETAS], "s--", color="#B5532A", label="Spearman")
    ax.axhline(1.0, color="0.8", lw=1)
    ax.set_xlabel(r"claim sharpness $\beta_m$")
    ax.set_ylabel("corr(gradual, congested)")
    ax.set_ylim(0, 1.05)
    ax.set_title("The tempo divergence survives the sharpness sweep")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "binding_sensitivity.png", dpi=150)

    lines = [
        "binding_sensitivity -- beta_m sweep and timescale decomposition",
        f"gradual theta = {THETA_GRADUAL:g}, congested theta = {THETA_CONGESTED:g}, "
        f"T_shock = {T_SHOCK:g} years, births off",
        "",
        "(1) cross-regime correlation of final reinstated mass, by claim sharpness:",
        f"{'beta_m':>8} {'pearson':>9} {'spearman':>9}",
    ]
    for bm in BETAS:
        lines.append(f"{bm:>8g} {pear[bm]:>+9.3f} {spear[bm]:>+9.3f}")
    lines += [
        "the divergence survives the full sweep; the point size depends on the handle.",
        "",
        "(2) timescale decomposition at beta_m = 3 (corr with the gradual run):",
        f"  congested, both slowed (theta_L = theta_abs = 15):   "
        f"{pearsonr(rg, keep[3.0][1])[0]:+.3f}",
        f"  slow absorption only (theta_L = 1, theta_abs = 15):  {c_absorb:+.3f}",
        f"  slow mobility only  (theta_L = 15, theta_abs = 1):   {c_mobil:+.3f}",
        "the destination result is carried by the absorption timescale alone.",
    ]
    iface.write_summary("binding_sensitivity", lines)


if __name__ == "__main__":
    main()
