"""
d04_survival_gate_robustness.py
-------------------------------
Survival-gate robustness of the destination result (manuscript secs. 4.2 and
6): both tempo regimes rerun with the survival gate off, and with a
comparative-advantage gate replacing the uniform-R absolute gate.

Pre-registered hypothesis (written before the runs, following the d03
decomposition): the downgrading is carried by theta_abs, which governs HOW
FAST seed binds, while the gate governs WHERE seed lands. The tempo
divergence -- corr(gradual, congested) far below one -- should therefore
survive both gate variants; what may move is the skill character of the
gainers, especially where the comparative-advantage gate lets seed survive
higher up the price gradient.

Variants:
  baseline   the companion's absolute gate, seeding by (1 - a)
  gate_off   seeding on the full gradient ring, capital-dominated core included
  ca_L       comparative advantage: h(r) = exp(L*(1 - phihat(r))), phihat the
             unit-amplitude technology shape, so human productivity is high
             where the machine is weak; capital operates where
             s_K phi_K > h R / Pi. L in {0.5, 1, 2}. Note h is uniform on the
             seeding ring (phihat constant there), so the variant moves the
             survival margin up the price gradient rather than around the ring.

Outcome (frozen below): the divergence survives every variant (corr +0.35 to
+0.46) and the size mechanism is intact (congested size rank correlation
+0.91 to +0.96). Conditional on the gate is only the gainers' skill
character: gate_off seeds the core and the congested absorbers become large
health and administrative occupations instead of food service; the
comparative-advantage gate raises the mass-weighted gainer price with L.

Births are disabled (see d02). All numbers write to experiment/results/ and
are asserted against the frozen baseline.

Usage: python experiment/d04_survival_gate_robustness.py

Recalibration note: the frozen baselines in this script were re-frozen
after the anchoring of the dynamic sorting kernel (alpha_o through the
interface; patch series 01-04). Point values quoted in the hypothesis
text above are the pre-anchoring pre-registration record; the
pre-anchoring baselines remain in git history.
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

VARIANTS = {
    "baseline": {},
    "gate_off": {"survival_gate": False},
    "ca_0.5": {"ca_lambda": 0.5},
    "ca_1": {"ca_lambda": 1.0},
    "ca_2": {"ca_lambda": 2.0},
}

# Frozen baseline (this machine, this calibration), tolerance 0.02.
BASELINE = {                # corr(grad, cong), congested size-Spearman
    "baseline": (0.391, 0.941),
    "gate_off": (0.360, 0.956),
    "ca_0.5": (0.410, 0.933),
    "ca_1": (0.430, 0.922),
    "ca_2": (0.466, 0.895),
}


def main():
    layer = iface.load_static_layer()
    occ = layer.occ
    Pi_mu = layer.inp.field.pi(occ["xi"].to_numpy(), occ["chi"].to_numpy())

    def run(theta, **kw):
        dyn, rec, _ = rd.main(theta_L=theta, theta_abs=theta, T_shock=T_SHOCK,
                              T_max=T_MAX, dt=DT, max_births=0,
                              verbose=False, layer=layer, **kw)
        assert rec["U_tot"][-1] < 1e-6, "U does not drain"
        return dyn

    def char(r):
        return float(np.sum(r * Pi_mu) / r.sum())

    size = None
    rows = []
    tops = {}
    for name, kw in VARIANTS.items():
        dg = run(THETA_GRADUAL, **kw)
        dc = run(THETA_CONGESTED, **kw)
        rg, rc = dg.reinst[:dg.n0], dc.reinst[:dc.n0]
        if size is None:
            size = dg.original[:dg.n0]
        corr = float(pearsonr(rg, rc)[0])
        ssz = float(spearmanr(rc, size)[0])
        b = BASELINE[name]
        assert abs(corr - b[0]) < 0.02, f"{name}: corr drifted {corr:.3f}"
        assert abs(ssz - b[1]) < 0.02, f"{name}: size mech drifted {ssz:.3f}"
        rows.append((name, corr, ssz, char(rg), char(rc), rg.sum(), rc.sum()))
        tops[name] = (
            [occ["Title"].iloc[i] for i in np.argsort(rg)[::-1][:5]],
            [occ["Title"].iloc[i] for i in np.argsort(rc)[::-1][:5]])

    # ---- outputs ----
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(iface.RESULTS / "survival_gate_robustness.csv", "w") as fh:
        fh.write("variant,corr_grad_cong,congested_size_spearman,"
                 "gainer_meanPi_gradual,gainer_meanPi_congested,"
                 "reinst_gradual,reinst_congested\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.2f},{r[4]:.2f},"
                     f"{r[5]:.4f},{r[6]:.4f}\n")

    names = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax1.bar(names, [r[1] for r in rows], color="#2C5A57")
    ax1.axhline(1.0, color="0.8", lw=1)
    ax1.set_ylabel("corr(gradual, congested)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("The tempo divergence survives the gate variants")
    ax2.plot(names, [r[3] for r in rows], "o-", color="#2C5A57", label="gradual")
    ax2.plot(names, [r[4] for r in rows], "s--", color="#B5532A", label="congested")
    ax2.set_ylabel(r"mass-weighted gainer price $\bar\Pi$")
    ax2.set_title("The gainers' price level is what the gate moves")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "survival_gate_robustness.png", dpi=150)

    lines = [
        "survival_gate_robustness -- the destination result across gate variants",
        f"gradual theta = {THETA_GRADUAL:g}, congested theta = {THETA_CONGESTED:g}, "
        f"T_shock = {T_SHOCK:g} years, births off",
        "",
        f"{'variant':>9} {'corr(g,c)':>10} {'cong size-rank':>14} "
        f"{'meanPi grad':>12} {'meanPi cong':>12}",
    ]
    for r in rows:
        lines.append(f"{r[0]:>9} {r[1]:>+10.3f} {r[2]:>+14.3f} "
                     f"{r[3]:>12.1f} {r[4]:>12.1f}")
    lines += ["",
              "the divergence and the size mechanism survive every variant;",
              "the gate moves the gainers' price level, not the tempo dependence.",
              ""]
    for name in VARIANTS:
        g, c = tops[name]
        lines.append(f"top gainers, {name}:")
        lines.append("  gradual:   " + "; ".join(t[:34] for t in g[:3]))
        lines.append("  congested: " + "; ".join(t[:34] for t in c[:3]))
    iface.write_summary("survival_gate_robustness", lines)


if __name__ == "__main__":
    main()
