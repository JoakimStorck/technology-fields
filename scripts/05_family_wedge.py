"""
05_family_wedge.py
------------------
The measurement layer for the non-spatial wage component. 04 showed
that ~48% of the wage cross-section is residual to the price field,
that about a third of it is family/capability composition, and that
none of it is smooth spatial structure. The theory layer stays
family-free; this script measures the family-level wage wedge once and
exports it, so simulations can be run with and without it.

Definition. The occupation wedge is the log gap between the observed
wage and the model's bundle-priced wage,

    eta_o = ln w_obs,o - ln( sum_t b_t Pi(r_t) ),

and the family wedge eta_g is the unweighted mean of eta_o over the
occupations of Job Family g in the estimation sample (employment-
weighted means are reported alongside as a robustness column). The
wedge is measured ONCE on the cross-section and held fixed under
technology shocks - the same discipline previously applied to the
structural distortion parameter: validated against institutional
composition, never re-optimized against wage statistics.

Sensitivity demonstration. The wedge enters the operated regime through
the effective price of labor, Pi_eff = exp(eta_g) * Pi: dearer families
cross the takeover margin sooner. With the (still illustrative)
technology of script 02 we run displacement with and without the wedge
and report how much the incidence ordering moves - the quantitative
content of the baseline-vs-precision design.

Outputs:
    results/family_wage_wedge.csv        eta_g, sd, n, employment-weighted
    results/family_wedge_summary.txt
    results/family_wedge.png             eta_g by family, sorted
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.data import load_bundles, load_family_map, load_mincer_sample
from model.price_field import PriceField
from model.technology import Technology

RESULTS = REPO_ROOT / "results"

# Same illustrative technology and economy parameters as script 02
DEMO_TECH = Technology(xi_K=np.deg2rad(75), chi_K=0.45, z_K=0.35,
                       A_K=1.0, s_K=1.0)
DEMO_R = 18.0
DEMO_TAU = 0.08


def main() -> None:
    field = PriceField.from_results()
    bundles = load_bundles()
    sample = load_mincer_sample()
    lines: list[str] = [f"Sample: N = {len(sample)}", ""]

    # ── estimate the wedge ────────────────────────────────────────
    w_bundle = field.bundle_wage(bundles)
    df = sample.merge(w_bundle, left_on="onet_code", right_index=True,
                      how="inner")
    df["eta_o"] = df["ln_wage"] - np.log(df["w_bundle"])

    def wmean(d):
        wt = d["TOT_EMP"].fillna(0).clip(lower=0)
        return (np.average(d["eta_o"], weights=wt) if wt.sum() > 0
                else np.nan)

    g = df.groupby("Job Family")
    wedge = pd.DataFrame({
        "eta_g": g["eta_o"].mean(),
        "eta_g_sd": g["eta_o"].std(),
        "n": g.size(),
        "eta_g_empw": g.apply(wmean, include_groups=False),
    }).sort_values("eta_g")
    wedge.to_csv(RESULTS / "family_wage_wedge.csv")

    share = 1 - (df["eta_o"]
                 - df["Job Family"].map(wedge["eta_g"])).var() / \
        df["eta_o"].var()
    lines += [
        "Family wage wedge eta_g (log points vs bundle-priced wage):",
        f"  range {wedge['eta_g'].min():+.3f} "
        f"({wedge['eta_g'].idxmin()}) to {wedge['eta_g'].max():+.3f} "
        f"({wedge['eta_g'].idxmax()})",
        f"  share of occupation-level wedge variance absorbed by "
        f"family means: {share:.3f}",
        f"  corr(eta_g, eta_g_empw) = "
        f"{wedge[['eta_g', 'eta_g_empw']].corr().iloc[0, 1]:.3f}",
        "",
    ]

    # ── sensitivity: regime with and without the wedge ────────────
    eta_o = df.set_index("onet_code")["Job Family"].map(
        wedge["eta_g"]).rename("eta")
    t = DEMO_TECH
    d0 = t.displacement(bundles, field, DEMO_R, DEMO_TAU)
    d1 = t.displacement(bundles, field, DEMO_R, DEMO_TAU, wedge=eta_o)
    both = d0[["D_o"]].join(d1[["D_o"]], lsuffix="_base", rsuffix="_wedge",
                            how="inner")
    both = both.loc[both.index.isin(df["onet_code"])]
    dD = both["D_o_wedge"] - both["D_o_base"]
    rho = both.corr(method="spearman").iloc[0, 1]
    fam = load_family_map().reindex(both.index)
    by_fam = dD.groupby(fam).mean().sort_values()
    lines += [
        "Regime sensitivity (illustrative technology of script 02):",
        f"  mean displaced share: baseline {both['D_o_base'].mean():.4f}, "
        f"wedged {both['D_o_wedge'].mean():.4f}",
        f"  Spearman rho(D_o baseline, wedged) = {rho:.4f}",
        f"  mean |Delta D_o| = {dD.abs().mean():.4f}, "
        f"max |Delta D_o| = {dD.abs().max():.4f}",
        "  largest family-mean shifts in D_o:",
    ]
    show = pd.concat([by_fam.head(3), by_fam.tail(3)])
    for name, v in show.items():
        lines.append(f"    {name[:46]:46s} {v:+.4f}")
    lines += ["", "  Interpretation: identical orderings (rho ~ 1) mean the",
              "  parsimonious wedge-free baseline is cheap; large shifts",
              "  mean institutional wage setting steers automation",
              "  incidence and the wedge variant must be reported."]

    # ── figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["C3" if v > 0 else "C0" for v in wedge["eta_g"]]
    ax.barh(np.arange(len(wedge)), wedge["eta_g"], color=colors, alpha=0.85)
    ax.errorbar(wedge["eta_g"], np.arange(len(wedge)),
                xerr=wedge["eta_g_sd"] / np.sqrt(wedge["n"]),
                fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
    ax.set_yticks(np.arange(len(wedge)))
    ax.set_yticklabels([f"{i} (n={n})" for i, n in
                        zip(wedge.index, wedge["n"])], fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"family wage wedge $\eta_g$ "
                  "(log points vs bundle-priced wage; bars: $\\pm$ s.e.)")
    fig.tight_layout()
    fig.savefig(RESULTS / "family_wedge.png", dpi=150)
    plt.close(fig)

    (RESULTS / "family_wedge_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
