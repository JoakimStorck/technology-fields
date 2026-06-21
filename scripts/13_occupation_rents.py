"""
13_occupation_rents.py
----------------------
Extension to occupation-level rents, making Acemoglu & Restrepo's within-group
wage-compression prediction (Prop 4 of "Automation and Rent Dissipation",
NBER w32536) testable in the geometry. The static paper uses a FAMILY-level rent
wedge eta_g (the family mean of the wage residual), which has no within-family
variation and therefore cannot speak to within-group compression. Here we recover
the occupation-level rent eta_o -- the within-family residual the family wedge
averages away -- and use it as the wedge.

RENT MEASURE (sharpened). The rent is the occupation's wage above what its
priced capability content commands -- AR's "wage above opportunity cost." We
residualise the log wage on the FULL capability vector {S1,S2,A1,A2}
(default, "capabilities"), not on the single S1-dominated price composite. This
addresses the concern that an unmodelled capability dimension is misread as rent.
A robustness panel reports three controls of increasing richness:
    price        : lnw ~ ln Pi(mu_o)            (one composite; R^2 ~ 0.57)
    capabilities : lnw ~ S1+S2+A1+A2            (default; R^2 ~ 0.66)
    cap+position : lnw ~ S1+S2+A1+A2+xi+chi     (R^2 ~ 0.67)
Position is nearly collinear with the capabilities (it adds ~1.5pp of R^2) and is
NOT used as the default, since spatially patterned rents would be absorbed into
the position coefficients (over-control). The result is robust across all three.

MECHANISM (AR, reproduced here):
  - automation targets high-rent tasks WITHIN a group (Prop 3 at occupation level)
  - displaced high-rent workers fall back toward their base (capability) wage,
    losing the rent -> within-group wage dispersion COMPRESSES (Prop 4), with a
    flat-then-falling profile across within-group wage percentiles (AR Fig 1).

TWO MEASUREMENT POINTS:
  1. The wage change is the BOUNDED rent loss dlnw_o = ln[(1-D)e^{eta}+D] - eta in
     [-eta_o, 0] (a displaced share D falls to base, losing the rent), NOT the
     bundle value change dW_bundle, which is not a log wage change and explodes.
     The base log wage is the OBSERVED lnw_o.
  2. The full U (an up-turn at the very top) requires AR Assumption 2(i): the
     highest-rent tasks are non-automatable. With CAP_TOP_FRAC > 0 we make the top
     rent quantile non-automatable and bend the top up; with 0 we get the hook
     (AR's other case, Delta ln w_g(1) < Delta ln w_g).

RESULT (capabilities, CAP_TOP_FRAC=0): wage R^2 0.66; within-family
corr(D,eta)=+0.45, corr(dlnw,eta)=-0.77; within-family wage sd 0.285 -> 0.251
(compression in 82% of families). Robust to the price and cap+position controls.

Usage:
    python scripts/13_occupation_rents.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.regime import regime

_spec = importlib.util.spec_from_file_location("_setup", Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)
_s11 = importlib.util.spec_from_file_location("s11", Path(__file__).parent / "11_centroid_shift_test.py")
s11 = importlib.util.module_from_spec(_s11)
_s11.loader.exec_module(s11)

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
MIN_FAM = 6
N_BINS = 8
DEFAULT_CONTROL = "capabilities"


def _observed_logwage(occ):
    w = occ["OCC_CODE"].map(s11.oews_median(2019)).to_numpy().astype(float)
    ok = np.isfinite(w)
    lnw = np.zeros(len(w))
    lnw[ok] = np.log(w[ok])
    return lnw, ok


def occupation_rent(inp, occ, control=DEFAULT_CONTROL):
    """eta_o = residual of the observed log wage off priced capability content.
    `control` selects the regressors: 'price' (ln Pi composite), 'capabilities'
    (S1..A2), or 'cap+position' (S1..A2 + xi + chi). Returns (eta, ok, lnw, R2)."""
    lnw, ok = _observed_logwage(occ)
    field = inp.field
    cols = {
        "price": [np.log(field.pi(occ["xi"].to_numpy(), occ["chi"].to_numpy()))],
        "capabilities": [occ[k].to_numpy() for k in ("S1", "S2", "A1", "A2")],
        "cap+position": [occ[k].to_numpy() for k in ("S1", "S2", "A1", "A2", "xi", "chi")],
    }[control]
    X = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in cols])
    coef = np.linalg.lstsq(X, lnw[ok], rcond=None)[0]
    pred = X @ coef
    r2 = 1 - np.sum((lnw[ok] - pred) ** 2) / np.sum((lnw[ok] - lnw[ok].mean()) ** 2)
    eta = np.zeros(len(lnw))
    eta[ok] = lnw[ok] - pred
    return eta, ok, lnw, r2


def run(inp, L0, occ, tech, ell, eta, ok, lnw, cap_top_frac=0.0):
    """Displacement with the occupation rent as the wedge; optionally make the top
    rent quantile non-automatable (AR Assumption 2(i)). Bounded rent-loss wage
    change off the OBSERVED base log wage."""
    out = regime(inp, tech, L0, R, TAU, GAMMA, ell, BETA, wedge=eta, survival=True)
    D = np.clip(out["D_o"], 0.0, 1.0)
    if cap_top_frac > 0:
        thr = np.quantile(eta[ok], 1.0 - cap_top_frac)
        D = np.where(eta > thr, 0.0, D)
    dlnw = np.log((1 - D) * np.exp(eta) + D) - eta            # in [-eta, 0]
    df = pd.DataFrame({"fam": occ["Job Family"].to_numpy(), "eta": eta, "D": D,
                       "dlnw": dlnw, "lnw0": lnw, "ok": ok})
    df = df[df.ok].copy()
    df["lnw1"] = df.lnw0 + df.dlnw
    return df


def prop4_stats(df):
    def wf(g):
        if len(g) < MIN_FAM:
            return None
        return pd.Series({"cDe": spearmanr(g.D, g.eta)[0], "cWe": spearmanr(g.dlnw, g.eta)[0],
                          "sd0": g.lnw0.std(), "sd1": g.lnw1.std()})
    res = df.groupby("fam").apply(wf, include_groups=False).dropna()
    big = df.groupby("fam").filter(lambda g: len(g) >= MIN_FAM).copy()
    big["pct"] = big.groupby("fam")["lnw0"].rank(pct=True)
    edges = np.linspace(0, 1, N_BINS + 1)
    big["bin"] = np.clip(np.digitize(big["pct"], edges) - 1, 0, N_BINS - 1)
    profile = big.groupby("bin")["dlnw"].mean().reindex(range(N_BINS))
    return res, profile, edges


def main():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    lines = ["Occupation-level rents and AR (w32536) Prop 4: within-group compression.", ""]

    # ---- robustness panel across rent controls (hook, no cap) -------------
    lines.append("Robustness of the rent measure (CAP_TOP_FRAC=0):")
    lines.append(f"{'control':<14} {'wageR2':>7} {'eta_sd':>7} {'wf_sd':>6} {'corr(D,eta)':>12} "
                 f"{'corr(dlnw,eta)':>15} {'sd0->sd1':>12} {'%compress':>10}")
    for control in ("price", "capabilities", "cap+position"):
        eta, ok, lnw, r2 = occupation_rent(inp, occ, control)
        df = run(inp, L0, occ, tech, ell, eta, ok, lnw, cap_top_frac=0.0)
        res, _, _ = prop4_stats(df)
        wf_sd = pd.Series(eta[ok]).groupby(occ["Job Family"].to_numpy()[ok]).std().mean()
        lines.append(f"{control:<14} {r2:>7.3f} {eta[ok].std():>7.3f} {wf_sd:>6.3f} "
                     f"{res.cDe.mean():>+12.3f} {res.cWe.mean():>+15.3f} "
                     f"{res.sd0.mean():>5.3f}->{res.sd1.mean():.3f} "
                     f"{100*(res.sd1<res.sd0).mean():>9.0f}%")
    lines.append("")

    # ---- default measure: hook vs U profile -------------------------------
    eta, ok, lnw, r2 = occupation_rent(inp, occ, DEFAULT_CONTROL)
    profiles = {}
    for cap, tag in [(0.0, "hook (no cap)"), (0.05, "U (top 5% non-automatable)")]:
        df = run(inp, L0, occ, tech, ell, eta, ok, lnw, cap_top_frac=cap)
        res, profile, edges = prop4_stats(df)
        profiles[tag] = (profile, edges)
        lines += [f"[default='{DEFAULT_CONTROL}', {tag}]  n={len(res)} families, wage R^2={r2:.3f}",
                  f"   within-family corr(D,eta)={res.cDe.mean():+.3f}  corr(dlnw,eta)={res.cWe.mean():+.3f}  "
                  f"sd {res.sd0.mean():.3f}->{res.sd1.mean():.3f} ({100*(res.sd1<res.sd0).mean():.0f}% compress)",
                  "   profile dlnw by within-group wage percentile: "
                  + " ".join(f"{profile[i]:+.3f}" for i in range(N_BINS))]
    (RESULTS / "occupation_rents_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tag, (profile, edges) in profiles.items():
        centers = (edges[:-1] + edges[1:]) / 2 * 100
        ax.plot(centers, profile.values, "-o", ms=4, label=tag)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("within-group wage percentile")
    ax.set_ylabel(r"$\Delta \ln w$ (rent loss)")
    ax.set_title("Within-group wage change (AR w32536 Prop 4 / Fig 1) in the geometry")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "occupation_rents_prop4.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {RESULTS/'occupation_rents_summary.txt'} and occupation_rents_prop4.png")


if __name__ == "__main__":
    main()
