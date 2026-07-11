"""
d05_binding_law_comparison.py
-----------------------------
Producer of the binding-law comparison quoted in manuscript sec. 3.3: the
conflated alternative -- absorption proportional to size times match,
iota_o ~ M_o FIT_o, no per-occupation cap -- against the match-allocated,
size-rate-capped law of eq. claim. Both run at the reference tempo
(theta = 3) through the same shock.

Owns the sec. 3.3 numbers: under the conflated form absorption tracks size
(corr +0.99) and ignores the claim (+0.07); under the match-allocated form
absorption tracks the unconstrained claim (+0.73) and the size correlation
falls to +0.13 at the reference tempo (and to +0.04 in the gradual regime,
d02). Claim and size are the d02 metrics: the seed-weighted claim share at
beta_m = 3 with no cap, and the pre-shock task mass.

Births are disabled (see d02). All numbers write to experiment/results/ and
are asserted against the frozen baseline.

Usage: python experiment/d05_binding_law_comparison.py

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
from scipy.stats import pearsonr, spearmanr


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

THETA_REF, T_SHOCK, T_MAX, DT = 3.0, 5.0, 60.0, 0.2
BETA_M = 3.0

# Frozen baseline (this machine, this calibration), tolerance 0.02.
BASELINE = {  # (corr size, corr claim), Pearson
    "size_multiplies": (0.995, 0.074),
    "match_allocated": (0.107, 0.750),
}


def main():
    layer = iface.load_static_layer()
    res = {}
    for law in ("match_allocated", "size_multiplies"):
        dyn, rec, occ = rd.main(theta_L=THETA_REF, theta_abs=THETA_REF,
                                T_shock=T_SHOCK, T_max=T_MAX, dt=DT,
                                max_births=0, verbose=False, layer=layer,
                                binding_law=law)
        assert rec["U_tot"][-1] < 1e-6, "U does not drain"
        res[law] = dyn.reinst[:dyn.n0].copy()
        n0 = dyn.n0

    dyn_ref = rd.Dyn(layer.eq, layer.inp, layer.L0, layer.ell,
                     layer.rho, lam_over=layer.lam_over)
    size = dyn_ref.original[:n0]
    a_m = layer.set_maturity(layer.tech.A_K)
    sw = layer.eq.g_hat * (1.0 - a_m) * layer.eq.area
    sw = sw / sw.sum()
    Wb = dyn_ref.FIT[:n0] ** BETA_M
    claim = (Wb / Wb.sum(0)[None, :]) @ sw

    lines = [
        "binding_law_comparison -- size-multiplies-match vs match-allocated",
        f"reference tempo theta = {THETA_REF:g}, T_shock = {T_SHOCK:g} years, births off",
        "claim = unconstrained seed-weighted claim share at beta_m = "
        f"{BETA_M:g}; size = pre-shock task mass (d02 metrics)",
        "",
        f"{'law':>16} {'corr(absorb,size)':>18} {'corr(absorb,claim)':>19}",
    ]
    for law in ("size_multiplies", "match_allocated"):
        r = res[law]
        cs = float(pearsonr(r, size)[0])
        cc = float(pearsonr(r, claim)[0])
        b = BASELINE[law]
        assert abs(cs - b[0]) < 0.02, f"{law}: size corr drifted {cs:.3f}"
        assert abs(cc - b[1]) < 0.02, f"{law}: claim corr drifted {cc:.3f}"
        lines.append(f"{law:>16} {cs:>+18.3f} {cc:>+19.3f}   "
                     f"(rank: size {spearmanr(r, size)[0]:+.3f}, "
                     f"claim {spearmanr(r, claim)[0]:+.3f})")
    lines += [
        "",
        "under the conflated form size decides the destination and the claim is",
        "ignored; the match-allocated, size-rate-capped form inverts this.",
    ]
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    iface.write_summary("binding_law_comparison", lines)


if __name__ == "__main__":
    main()
