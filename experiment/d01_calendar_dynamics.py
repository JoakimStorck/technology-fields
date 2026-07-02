"""
d01_calendar_dynamics.py
------------------------
Producer of fig:calendar in the tempo manuscript: the technology maturing as a
logistic diffusion over a T_shock = 5 year window (left), and the unbound stock
U(t) -- the transitional mismatch between destruction and creation -- for four
redistribution tempos theta in {1, 3, 8, 15} years against that fixed shock
(right). theta sets theta_L and theta_abs together ("the redistribution
tempo").

Owns one nameable part of the argument: the mismatch hump grows and lingers as
redistribution slows relative to the shock, but always drains to zero. The
result-bearing numbers are the peak heights, their timing, the fastest-to-
slowest peak ratio, and the drain check; all are written to
experiment/results/ and asserted against the frozen baseline below.

Births are disabled (max_births = 0): the birth layer belongs to the birth
extension (probe_allocation.py) and would confound the tempo comparison by
changing the occupation set across regimes.

Usage: python experiment/d01_calendar_dynamics.py
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

THETAS = (1.0, 3.0, 8.0, 15.0)
T_SHOCK, T_MAX, DT = 5.0, 60.0, 0.2

# Frozen baseline (this machine, this calibration). Asserted with 5 percent
# relative tolerance so a drifted calibration fails loudly.
BASELINE_PEAKS = {1.0: 0.000306, 3.0: 0.004358, 8.0: 0.013881, 15.0: 0.024563}
BASELINE_RATIO = 80.2


def main():
    layer = iface.load_static_layer()
    res = {}
    for th in THETAS:
        dyn, rec, occ = rd.main(theta_L=th, theta_abs=th, T_shock=T_SHOCK,
                                T_max=T_MAX, dt=DT, max_births=0,
                                verbose=False, layer=layer)
        res[th] = rec
        assert abs(rec["Lsum"][0] - rec["Lsum"][-1]) < 1e-6, "population not conserved"
        assert rec["U_tot"][-1] < 1e-8, f"U does not drain at theta={th}"

    peaks = {th: float(res[th]["U_tot"].max()) for th in THETAS}
    tpeak = {th: float(res[th]["t"][res[th]["U_tot"].argmax()]) for th in THETAS}
    ratio = peaks[THETAS[-1]] / peaks[THETAS[0]]

    for th in THETAS:
        assert abs(peaks[th] / BASELINE_PEAKS[th] - 1.0) < 0.05, \
            f"peak drifted at theta={th}: {peaks[th]:.6f} vs {BASELINE_PEAKS[th]:.6f}"
    assert abs(ratio / BASELINE_RATIO - 1.0) < 0.05, f"peak ratio drifted: {ratio:.1f}"

    # ---- csv ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    t = res[THETAS[0]]["t"]
    cols = [t, res[THETAS[0]]["A_K"]] + [res[th]["U_tot"] for th in THETAS]
    header = "t,A_K," + ",".join(f"U_tot_theta{th:g}" for th in THETAS)
    np.savetxt(iface.RESULTS / "calendar_dynamics.csv",
               np.column_stack(cols), delimiter=",", header=header, comments="")

    # ---- figure ----
    colors = {1.0: "#2C5A57", 3.0: "#B5532A", 8.0: "#6D3C8E", 15.0: "#8A8A3A"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    r0 = res[THETAS[0]]
    ax1.plot(r0["t"], r0["A_K"], lw=2, color="0.25")
    ax1.axvspan(0, T_SHOCK, color="0.9", zorder=0)
    ax1.set_xlim(0, 15)
    ax1.set_xlabel("years")
    ax1.set_ylabel(r"technology maturity $A_K(t)$")
    ax1.set_title(f"Logistic diffusion over a {T_SHOCK:g}-year window")
    ax1.grid(alpha=0.25)
    for th in THETAS:
        ax2.plot(res[th]["t"], res[th]["U_tot"], lw=2, color=colors[th],
                 label=rf"$\theta = {th:g}$")
    ax2.axvspan(0, T_SHOCK, color="0.9", zorder=0)
    ax2.set_xlim(0, 20)
    ax2.set_xlabel("years")
    ax2.set_ylabel(r"unbound stock $U(t)$")
    ax2.set_title("The mismatch hump grows as redistribution slows")
    ax2.legend(fontsize=9, title=r"redistribution tempo")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "calendar_dynamics.png", dpi=150)

    lines = [
        "calendar_dynamics -- U(t) mismatch hump over redistribution tempos",
        f"T_shock = {T_SHOCK:g} years (logistic 5%->95%), dt = {DT}, T_max = {T_MAX:g}, births off",
        f"theta = theta_L = theta_abs, swept over {THETAS}",
        "",
        f"{'theta':>6} {'peak U_tot':>11} {'t(peak)':>8} {'U_tot(end)':>11}",
    ]
    for th in THETAS:
        lines.append(f"{th:>6g} {peaks[th]:>11.6f} {tpeak[th]:>8.1f} "
                     f"{res[th]['U_tot'][-1]:>11.2e}")
    lines += [
        "",
        f"peak ratio slowest/fastest (theta {THETAS[-1]:g} / {THETAS[0]:g}): {ratio:.1f}",
        "population conserved and U drains to zero in every run (asserted).",
    ]
    iface.write_summary("calendar_dynamics", lines)


if __name__ == "__main__":
    main()
