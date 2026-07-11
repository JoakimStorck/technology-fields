"""
d10_capacity_exponent.py
------------------------
The capacity-scaling sweep of the referee response (M2, item P2): is the
path-dependence of the destination a property of the linear capacity law
c_o = M_o/theta_abs, or does it survive sublinear scaling?

Design. Capacity generalises to the level-neutral sublinear law

    c_o = (dt/theta_abs) * M_tot * M_o^p / sum_o M_o^p,   p in {0.5, 0.7, 1.0},

which preserves the aggregate absorption rate (dt/theta_abs) M_tot for every
p, so the exponent moves only the cross-sectional allocation of capacity and
not the effective tempo; p = 1 is exactly the baseline law. The plan's
literal form c_o = M_o^p/theta_abs would raise aggregate capacity by the
factor sum(M^p)/sum(M) (large, since the M_o are small shares), mechanically
de-congesting the theta = 15 regime and confounding the scaling assumption
with the tempo. Under full saturation the normalisation is a common time
change ds = k(t) dt, so the generalised slow limit is unaffected: dM_o/ds =
M_o^p is separable, M_o^{1-p} is linear in s, and the final allocation is

    R_o = [M_o(0)^{1-p} + (1-p) s*]^{1/(1-p)} - M_o(0),    p < 1,

with s* the unique scalar solving sum_o R_o = S (generalised Proposition 2;
p = 1 recovers the size shares R_o = S M_o(0)/M_tot, p -> 0 the uniform
absolute allocation R_o = S/n).

PRE-REGISTERED HYPOTHESES (written before any run):

  H1 (survival of the divergence). The cross-regime Pearson correlation
      corr(gradual theta = 1, congested theta = 15) for p in {0.5, 0.7}
      stays in the range of the existing gate variants, ~0.35-0.46
      (baseline p = 1: +0.37). FALSIFIER: if the correlation for p < 1
      leaves this range and rises toward the gradual run, the
      path-dependence is a property of the linear law and M2 is conceded
      in that form in the manuscript.

  H2 (character of the congested endpoint). Sublinear capacity relieves
      small occupations: in the congested regime the size-rank (Spearman)
      correlation falls monotonically below the p = 1 value +0.95, and the
      top size-quartile share of reinstated mass falls from 57 percent
      toward the uniform 25, while the bottom quartile recovers from 4
      percent. The quantitative size of the downgrading at p < 1 is the
      open question the sweep settles.

  H3 (theory verification at the deep endpoint). At theta_abs = 120
      (theta_L = 1, as in d06) the realised allocation matches the
      generalised closed form above with relative L1 comparable to the
      p = 1 slow-limit check (d06: L1 to size shares 0.279 at 120), and
      the closed form at p = 1 IS the size shares, reproducing that
      number.

OUTCOME (first accepted run, frozen below). H1 FELL in its registered
form: the cross-regime Pearson correlation leaves the gate-variant range
upward, +0.530 at p = 0.7 and +0.684 at p = 0.5. H2 CONFIRMED: congested
size rank falls monotonically 0.950 -> 0.784 -> 0.491, top size-quartile
share 57 -> 43 -> 35 percent, bottom quartile recovers 4 -> 10 -> 16.
H3 CONFIRMED: deep-limit relative L1 to the generalised closed form is
0.108 (p = 0.5) and 0.156 (p = 0.7) against 0.279 for the size shares at
p = 1, corr >= +0.97. Additionally the deep limit stays uncorrelated with
the fast endpoint for every p (+0.140, +0.103, +0.070), so the
theorem-level divergence of the limits survives all p; what the linear
law owns is the severity of the downgrading at a given tempo, not the
path-dependence itself. The manuscript states both.

Baseline tie-in: the p = 1.0 column must reproduce d02's frozen numbers
(cross-regime Pearson 0.371, congested size rank 0.950) -- asserted here, so
this producer cannot drift from the headline producer. d10's own numbers are
frozen after the first accepted run.

Births are disabled; theta_L = theta_abs = theta in the regime grid (the
pooled tempo), theta_L = 1 in the deep-limit runs (the d03/d06 convention:
the destination is carried by theta_abs alone).

Usage: python experiment/d10_capacity_exponent.py   (about 7 minutes)

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
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

P_GRID = (0.5, 0.7, 1.0)
THETA_GRADUAL, THETA_CONGESTED = 1.0, 15.0
T_SHOCK, T_MAX, DT = 5.0, 60.0, 0.2
THETA_DEEP, T_MAX_DEEP = 120.0, 120.0

# Frozen d02 tie-in (tolerance 0.02).
D02_PEARSON, D02_SIZE_RANK = 0.391, 0.941

# Frozen d10 baselines (tolerance 0.02).
BASE_CROSS = {0.5: 0.716, 0.7: 0.563, 1.0: 0.391}
BASE_SIZE_RANK = {0.5: 0.437, 0.7: 0.749, 1.0: 0.941}
BASE_TOPQ = {0.5: 0.345, 0.7: 0.413, 1.0: 0.559}
BASE_DEEP_L1 = {0.5: 0.121, 0.7: 0.176, 1.0: 0.309}
BASE_FAST_CORR = {0.5: 0.162, 0.7: 0.122, 1.0: 0.086}


def closed_form_slow(M0, S, p):
    """Generalised Prop 2 allocation: R_o(s*) with sum R_o = S."""
    if p == 1.0:
        return S * M0 / M0.sum()

    def total(s):
        return ((M0 ** (1.0 - p) + (1.0 - p) * s) ** (1.0 / (1.0 - p)) - M0).sum() - S

    s_hi = 1.0
    while total(s_hi) < 0:
        s_hi *= 2.0
    s_star = brentq(total, 0.0, s_hi, xtol=1e-14)
    return (M0 ** (1.0 - p) + (1.0 - p) * s_star) ** (1.0 / (1.0 - p)) - M0


def quartile_shares(r, key):
    q = np.quantile(key, [0.25, 0.5, 0.75])
    idx = np.digitize(key, q)
    return [float(r[idx == k].sum() / r.sum()) for k in range(4)]


def main():
    layer = iface.load_static_layer()

    def run(theta_L, theta_abs, p, T_max):
        dyn, rec, _ = rd.main(theta_L=theta_L, theta_abs=theta_abs,
                              T_shock=T_SHOCK, T_max=T_max, dt=DT,
                              max_births=0, verbose=False, layer=layer,
                              cap_exponent=p)
        assert rec["U_tot"][-1] < 1e-6, \
            f"U does not drain at p={p}, theta_abs={theta_abs}"
        return dyn

    # ---- regime grid ----
    rows = []
    grid = {}
    for p in P_GRID:
        rg = run(THETA_GRADUAL, THETA_GRADUAL, p, T_MAX)
        rc = run(THETA_CONGESTED, THETA_CONGESTED, p, T_MAX)
        n0 = rg.n0
        g, c = rg.reinst[:n0], rc.reinst[:n0]
        M0 = rg.original[:n0]
        pear = float(pearsonr(g, c)[0])
        spear = float(spearmanr(g, c)[0])
        size_s = float(spearmanr(c, M0)[0])
        qs = quartile_shares(c, M0)
        assert abs(pear - BASE_CROSS[p]) < 0.02, f"cross corr drifted at p={p}: {pear:.3f}"
        assert abs(size_s - BASE_SIZE_RANK[p]) < 0.02, f"size rank drifted at p={p}: {size_s:.3f}"
        assert abs(qs[3] - BASE_TOPQ[p]) < 0.02, f"top quartile drifted at p={p}: {qs[3]:.3f}"
        rows.append((p, pear, spear, size_s, qs[3], qs[0]))
        grid[p] = (g, c, M0)

    # p = 1 must reproduce the headline producer.
    p1 = [r for r in rows if r[0] == 1.0][0]
    assert abs(p1[1] - D02_PEARSON) < 0.02, f"d02 tie-in drifted: {p1[1]:.3f}"
    assert abs(p1[3] - D02_SIZE_RANK) < 0.02, f"d02 size rank drifted: {p1[3]:.3f}"

    # ---- deep-limit verification of the generalised closed form ----
    # Fast endpoint (p-independent: no cap ever binds in the fast limit).
    dyn_f = run(1.0, 0.05, 1.0, 40.0)
    r_fast = dyn_f.reinst[:dyn_f.n0]
    deep = []
    for p in P_GRID:
        dyn = run(1.0, THETA_DEEP, p, T_MAX_DEEP)
        n0 = dyn.n0
        r = dyn.reinst[:n0]
        M0 = dyn.original[:n0]
        pred = closed_form_slow(M0, float(r.sum()), p)
        l1 = float(np.abs(r - pred).sum() / r.sum())
        cfast = float(pearsonr(r, r_fast)[0])
        assert abs(l1 - BASE_DEEP_L1[p]) < 0.02, f"deep L1 drifted at p={p}: {l1:.3f}"
        assert abs(cfast - BASE_FAST_CORR[p]) < 0.02, f"fast corr drifted at p={p}: {cfast:.3f}"
        deep.append((p, l1, float(pearsonr(r, pred)[0]), cfast))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ps = [r[0] for r in rows]
    ax.plot(ps, [r[1] for r in rows], "o-", label="cross-regime Pearson")
    ax.plot(ps, [r[3] for r in rows], "s-", label="congested size rank")
    ax.plot(ps, [r[4] for r in rows], "^-", label="top size-quartile share")
    ax.axhspan(0.35, 0.46, color="0.85", zorder=0,
               label="gate-variant range (H1)")
    ax.set_xlabel("capacity exponent $p$"), ax.set_ylabel("value")
    ax.legend(fontsize=8), ax.set_title("regime grid")
    ax = axes[1]
    for p, l1, cf, cfast in deep:
        ax.bar(str(p), l1)
    ax.set_xlabel("$p$"), ax.set_ylabel("rel. $L_1$ to closed form")
    ax.set_title(r"deep limit $\theta_{\rm abs}=120$ vs generalised Prop 2")
    fig.tight_layout()
    fig.savefig(iface.RESULTS / "capacity_exponent.png", dpi=150)
    plt.close(fig)

    # ---- outputs ----
    with open(iface.RESULTS / "capacity_exponent.csv", "w") as fh:
        fh.write("p,cross_regime_pearson,cross_regime_spearman,"
                 "congested_size_rank,top_size_quartile,bottom_size_quartile\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},"
                     f"{r[4]:.4f},{r[5]:.4f}\n")

    lines = [
        "capacity_exponent -- level-neutral sublinear capacity sweep (P2/M2)",
        f"c_o = (dt/theta_abs) M_tot M_o^p / sum M^p, p in {P_GRID};",
        "regimes theta = 1 / 15 (pooled), T_shock = 5, births off.",
        "",
        f"{'p':>4} {'cross Pearson':>14} {'cross rank':>11} "
        f"{'size rank (cong)':>17} {'top size Q':>11} {'bottom Q':>9}",
    ]
    for r in rows:
        lines.append(f"{r[0]:>4} {r[1]:>+14.3f} {r[2]:>+11.3f} "
                     f"{r[3]:>+17.3f} {r[4]:>11.1%} {r[5]:>9.1%}")
    lines += [
        "",
        "deep limit (theta_L = 1, theta_abs = 120): realised allocation vs the",
        "generalised Prop 2 closed form R_o = [M0^(1-p) + (1-p)s*]^(1/(1-p)) - M0:",
    ]
    for p, l1, cf, cfast in deep:
        lines.append(f"   p = {p}:  rel L1 = {l1:.3f}   corr = {cf:+.4f}"
                     f"   corr with fast endpoint = {cfast:+.3f}")
    iface.write_summary("capacity_exponent", lines)


if __name__ == "__main__":
    main()
