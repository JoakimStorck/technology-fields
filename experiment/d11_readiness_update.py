"""
d11_readiness_update.py
-----------------------
The readiness-update variant of the referee response (M2, item P3): is
frozen readiness -- time-invariant e_o, required by Proposition 1's proof --
an innocuous simplification or a load-bearing assumption? The variant lets
bound work update the occupation's bundle and re-derives readiness from the
grown bundle each step; the comparison against the frozen baseline answers
whether occupations move materially when allowed to learn from what they
absorb.

Design (fixed in the action plan). The update set is bound mass only: the
binding flow is folded, as exact running moments over the grid, into a mixed
measure -- original mass M_o(0) at the measured centroid and capability
levels, plus bound mass at its binding locations. Mass-as-relevance
renormalisation, b_o' = (M_o(0) b_o + rho_o)/(M_o(0) + r_o), then gives the
drifted centroid mu_o' and capability levels q_{o,k}' as mass-weighted
means, from which E, LOC, and FIT are rebuilt every step. No counterfactual
tasks are evaluated, no new parameter enters, and the wage-bearing bundle is
untouched (dyn.grain and the values() channel are not modified):
displacement takes the payment, not the capability. The drifted mu_o is THE
centroid -- locality, claims, and the mobility distances all read it. With
the flag off the code path is bit-identical to the baseline (verified).

PRE-REGISTERED HYPOTHESES (written before any full run):

  H1 (drift law). Centroid drift |mu_T - mu_0| is proportional to the
      bound share x_o = r_o/(M_o(0) + r_o): rank correlation above 0.8
      in both regimes. Small specialists that bind much relative to size
      move strongly toward the crescent; large absorbers barely move.

  H2 (regime asymmetry). Entrenchment dominates the gradual regime and
      the congested regime is nearly unaffected: the allocation
      displacement (relative L1 between update-on and update-off final
      allocations) is larger in the gradual regime than in the congested,
      and the congested size-rank correlation stays at its frozen +0.95.

  H3 (widening). Updating widens the divergence: the cross-regime
      correlation under updating falls below the frozen +0.371.

  Open magnitude question (the decision criterion): whether the drifts
      are small against the readiness scale ell = 0.133 and the locality
      rho = 0.5. If the employment-weighted mean drift is well below ell,
      frozen readiness is innocuous and P4's paragraph records it as the
      conservative choice. If drifts are comparable to ell for a material
      share of employment, the assumption is load-bearing and the
      manuscript must say so -- and the static-centroid premise is up for
      rethinking.

OUTCOME (first accepted run, frozen below). H1 CONFIRMED in the gradual
regime (Spearman +0.909) and FELL in the congested at the registered
threshold (+0.551 against the registered > 0.8): under capacity
allocation the bound share compresses toward a common value, so drift is
driven by the distance to the bound mass rather than by the share. H2
CONFIRMED: allocation displacement rel L1 0.218 gradual against 0.034
congested; congested corr(on, off) +0.998 and size rank +0.940. H3
CONFIRMED: the cross-regime correlation falls from +0.371 to +0.092 --
updating WIDENS the divergence. Magnitudes: employment-weighted mean
drift 0.012 (gradual) and 0.024 (congested), an order below ell; but
r-weighted drift in the gradual regime is 0.120, comparable to ell, and
17.4 percent of occupations drift beyond ell. The movers are the gradual
winners, and they move toward the crescent: entrenchment. Frozen
readiness is therefore the conservative choice -- the frozen headline
numbers UNDERSTATE the path-dependence that updating produces -- and the
static centroid stands, with the bias direction now measured.

Baseline tie-in: the update-off runs must reproduce d02's frozen numbers
(cross-regime Pearson 0.371, congested size rank 0.950). d11's own numbers
are frozen after the first accepted run.

Regimes theta = 1 / 15 (pooled theta_L = theta_abs), T_shock = 5,
T_max = 60, births off, baseline binding law, p = 1.

Usage: python experiment/d11_readiness_update.py   (about 5 minutes)

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
ELL, RHO = 0.133, 0.5

# Frozen d02 tie-in (tolerance 0.02).
D02_PEARSON, D02_SIZE_RANK = 0.391, 0.941

# Frozen d11 baselines (tolerance 0.02).
BASE = {"gradual":   dict(h1=0.914, l1=0.205, corr=0.889, drift_rw=0.108),
        "congested": dict(h1=0.571, l1=0.033, corr=0.998, drift_rw=0.043)}
BASE_CROSS_ON = 0.120


def main():
    layer = iface.load_static_layer()

    def run(theta, update):
        dyn, rec, _ = rd.main(theta_L=theta, theta_abs=theta,
                              T_shock=T_SHOCK, T_max=T_MAX, dt=DT,
                              max_births=0, verbose=False, layer=layer,
                              readiness_update=update)
        assert rec["U_tot"][-1] < 1e-6, f"U does not drain at theta={theta}"
        return dyn

    out = {}
    for name, theta in (("gradual", THETA_GRADUAL), ("congested", THETA_CONGESTED)):
        d_off = run(theta, False)
        d_on = run(theta, True)
        n0 = d_off.n0
        out[name] = (d_off, d_on, n0)

    # Baseline tie-in on the update-off runs.
    g_off, c_off = out["gradual"][0], out["congested"][0]
    n0 = out["gradual"][2]
    M0 = g_off.original[:n0]
    pear_off = float(pearsonr(g_off.reinst[:n0], c_off.reinst[:n0])[0])
    size_off = float(spearmanr(c_off.reinst[:n0], M0)[0])
    assert abs(pear_off - D02_PEARSON) < 0.02, f"d02 tie-in drifted: {pear_off:.3f}"
    assert abs(size_off - D02_SIZE_RANK) < 0.02, f"d02 size rank drifted: {size_off:.3f}"

    L0 = layer.L0[:n0]/layer.L0[:n0].sum()
    lines = [
        "readiness_update -- bound mass updates the bundle (P3/M2, d11)",
        "mixed-measure renormalisation; mu, q, FIT re-integrated each step;",
        f"regimes theta = {THETA_GRADUAL:g} / {THETA_CONGESTED:g}, T_shock = {T_SHOCK:g}, births off.",
        "",
    ]
    stats = {}
    for name in ("gradual", "congested"):
        d_off, d_on, n0 = out[name]
        r_on, r_off = d_on.reinst[:n0], d_off.reinst[:n0]
        drift = np.hypot(d_on.mu[:n0, 0]-d_on.mu0[:n0, 0],
                         d_on.mu[:n0, 1]-d_on.mu0[:n0, 1])
        x = r_on/(d_on.original[:n0] + r_on)
        h1 = float(spearmanr(drift, x)[0])
        l1 = float(np.abs(r_on - r_off).sum()/r_off.sum())
        corr_onoff = float(pearsonr(r_on, r_off)[0])
        size_on = float(spearmanr(r_on, M0)[0])
        dq = np.zeros(n0)
        for k in d_on._keys:
            Mtot = d_on.original[:n0] + r_on
            qk_on = (d_on.original[:n0]*d_on.q0[k][:n0] + d_on.Sq[k][:n0])/Mtot
            dq += d_on._cap.v[k]*np.abs(qk_on - d_on.q0[k][:n0])
        b = BASE[name]
        assert abs(h1 - b["h1"]) < 0.02, f"H1 drifted in {name}: {h1:.3f}"
        assert abs(l1 - b["l1"]) < 0.02, f"L1 drifted in {name}: {l1:.3f}"
        assert abs(corr_onoff - b["corr"]) < 0.02, f"corr drifted in {name}"
        assert abs(float(drift@(r_on/r_on.sum())) - b["drift_rw"]) < 0.02, \
            f"r-weighted drift drifted in {name}"
        stats[name] = dict(drift=drift, x=x, h1=h1, l1=l1,
                           corr_onoff=corr_onoff, size_on=size_on, dq=dq)
        lines += [
            f"{name} regime (theta = {THETA_GRADUAL if name=='gradual' else THETA_CONGESTED:g}):",
            f"   drift vs bound share r/(M0+r): Spearman {h1:+.3f}   (H1: > +0.8)",
            f"   drift: mean {drift.mean():.4f}   L0-weighted {float(drift@L0):.4f}"
            f"   r-weighted {float(drift@(r_on/r_on.sum())):.4f}   max {drift.max():.4f}",
            f"   share of occupations with drift > ell: {float((drift > ELL).mean()):.1%}"
            f"   (> rho: {float((drift > RHO).mean()):.1%});  ell = {ELL}, rho = {RHO}",
            f"   v-weighted capability shift |dq|: mean {dq.mean():.4f}"
            f"   r-weighted {float(dq@(r_on/r_on.sum())):.4f}   (ell = {ELL})",
            f"   allocation displacement rel L1(on, off) = {l1:.3f}"
            f"   corr(on, off) = {corr_onoff:+.3f}",
            f"   size rank of the allocation, update on: {size_on:+.3f}",
            "",
        ]

    g_on, c_on = out["gradual"][1], out["congested"][1]
    pear_on = float(pearsonr(g_on.reinst[:n0], c_on.reinst[:n0])[0])
    assert abs(pear_on - BASE_CROSS_ON) < 0.02, f"cross-regime on drifted: {pear_on:.3f}" 
    lines += [
        f"cross-regime Pearson: frozen {pear_off:+.3f}  ->  updating {pear_on:+.3f}"
        f"   (H3: falls below the frozen d02 value +0.391)",
    ]

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for name, col in (("gradual", "#1f3d7a"), ("congested", "#b3401f")):
        st = stats[name]
        ax.scatter(st["x"], st["drift"], s=8, alpha=0.4, c=col, label=name)
    ax.axhline(ELL, color="0.4", ls="--", lw=0.8)
    ax.text(0.02, ELL*1.05, r"$\ell$", fontsize=9, color="0.3")
    ax.set_xlabel(r"bound share $r_o/(M_o(0)+r_o)$")
    ax.set_ylabel(r"centroid drift $|\mu_T-\mu_0|$")
    ax.legend(fontsize=8); ax.set_title("drift law (H1)")
    ax = axes[1]
    for name, col in (("gradual", "#1f3d7a"), ("congested", "#b3401f")):
        d_off, d_on, _ = out[name]
        ax.scatter(d_off.reinst[:n0], d_on.reinst[:n0], s=8, alpha=0.4, c=col,
                   label=f"{name}: rel $L_1$ = {stats[name]['l1']:.2f}")
    lim = max(g_off.reinst[:n0].max(), g_on.reinst[:n0].max())
    ax.plot([0, lim], [0, lim], color="0.5", lw=0.7)
    ax.set_xlabel("final allocation, frozen"), ax.set_ylabel("final allocation, updating")
    ax.legend(fontsize=8); ax.set_title("allocation displacement (H2)")
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "readiness_update.png", dpi=150)
    plt.close(fig)

    with open(iface.RESULTS / "readiness_update.csv", "w") as fh:
        fh.write("regime,h1_spearman,drift_mean,drift_L0w,drift_rw,drift_max,"
                 "share_gt_ell,relL1_on_off,corr_on_off,size_rank_on\n")
        for name in ("gradual", "congested"):
            st = stats[name]
            d_off, d_on, _ = out[name]
            r_on = d_on.reinst[:n0]
            fh.write(f"{name},{st['h1']:.4f},{st['drift'].mean():.4f},"
                     f"{float(st['drift']@L0):.4f},"
                     f"{float(st['drift']@(r_on/r_on.sum())):.4f},"
                     f"{st['drift'].max():.4f},"
                     f"{float((st['drift'] > ELL).mean()):.4f},"
                     f"{st['l1']:.4f},{st['corr_onoff']:.4f},{st['size_on']:.4f}\n")
        fh.write(f"cross_regime,{pear_off:.4f},{pear_on:.4f},,,,,,,\n")
    iface.write_summary("readiness_update", lines)


if __name__ == "__main__":
    main()
