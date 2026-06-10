"""
03_field_vs_paper1.py
---------------------
Compares the estimated price-of-skill field Pi(r) with what Paper 1
currently reports, in the two forms Paper 1 uses:

(A) The directional return to depth. Paper 1, Figure 10A, shows the
    unconditional within-sector slope beta_chi for eight 45-degree
    sectors (ln w ~ const + chi, HC3). The field replaces the eight bars
    with the continuous first harmonic

        beta_chi(xi) = m3 + m4 cos xi + m5 sin xi.

    We re-estimate the sectoral slopes from the data (same specification
    as Paper 1, with HC3 confidence intervals) and overlay the field curve
    with a delta-method pointwise 95% band. For reference, the BALANCED
    second-harmonic variant is added (level AND interaction second
    harmonics); the unbalanced variant shown in an earlier version was a
    specification artifact (chi*cos 2xi proxies the omitted level term
    cos 2xi, corr ~ 0.93) and is no longer drawn. See 01_wage_field.py
    for the harmonic-order test battery.

(B) The wage map. Paper 1, Figure 8A, plots observed median wages in the
    geometry. We plot the field Pi(r) as a continuous surface with the observed
    occupation wages scattered on the SAME color scale, plus the
    residual map ln w_obs - ln Pi(centroid), with mean residuals per
    sector (the spatial signature of the omitted second harmonic).

Outputs:
    results/field_vs_sectoral_beta_chi.png
    results/field_vs_observed_wages.png
    results/field_vs_paper1_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.data import load_mincer_sample

RESULTS = REPO_ROOT / "results"

SECTOR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
SECTOR_CENTERS = np.deg2rad(np.arange(0, 360, 45))

# Published unconditional sectoral beta_chi, Paper 1 Figure 10A
PAPER1_FIG10A = {"E": -0.15, "NE": 0.81, "N": 0.93, "NW": 0.28,
                 "W": -0.28, "SW": -0.80, "S": -1.04, "SE": -1.18}


def assign_sector(xi: np.ndarray) -> np.ndarray:
    """Paper 1 sector assignment: 45-degree wedges centered on the
    cardinal and intercardinal directions."""
    width = np.pi / 4
    return (np.mod(xi + width / 2, 2 * np.pi) // width).astype(int)


def fit_field(df: pd.DataFrame, second_harmonic: bool):
    """second_harmonic=True fits the BALANCED extension: second harmonics
    in both the level and the chi interaction."""
    cols = ["cos_xi", "sin_xi", "chi", "chi_cos", "chi_sin"]
    if second_harmonic:
        cols += ["cos2", "sin2", "chi_cos2", "chi_sin2"]
    X = sm.add_constant(df[cols].astype(float))
    return sm.OLS(df["ln_wage"].astype(float), X).fit(cov_type="HC3")


def beta_chi_curve(model, xi: np.ndarray, second_harmonic: bool):
    """beta_chi(xi) and its delta-method standard error."""
    names = ["chi", "chi_cos", "chi_sin"]
    V = [np.ones_like(xi), np.cos(xi), np.sin(xi)]
    if second_harmonic:
        names += ["chi_cos2", "chi_sin2"]
        V += [np.cos(2 * xi), np.sin(2 * xi)]
    V = np.column_stack(V)
    b = model.params[names].to_numpy()
    S = model.cov_params().loc[names, names].to_numpy()
    mean = V @ b
    se = np.sqrt(np.einsum("ij,jk,ik->i", V, S, V))
    return mean, se


def main() -> None:
    df = load_mincer_sample()
    df["cos2"] = np.cos(2 * df["xi"])
    df["sin2"] = np.sin(2 * df["xi"])
    lines: list[str] = [f"Sample: N = {len(df)}", ""]

    # ── sectoral slopes, Paper 1 specification ────────────────────
    df["sector_idx"] = assign_sector(df["xi"].to_numpy())
    rows = []
    for k, lab in enumerate(SECTOR_LABELS):
        d = df.loc[df["sector_idx"] == k]
        X = sm.add_constant(d[["chi"]].astype(float))
        m = sm.OLS(d["ln_wage"].astype(float), X).fit(cov_type="HC3")
        ci = m.conf_int().loc["chi"]
        rows.append(dict(sector=lab, center=np.degrees(SECTOR_CENTERS[k]),
                         n=len(d), beta=m.params["chi"],
                         lo=ci[0], hi=ci[1],
                         published=PAPER1_FIG10A[lab]))
    sec = pd.DataFrame(rows)
    lines.append("(A) Sectoral beta_chi: re-estimated (HC3 95% CI) vs "
                 "Paper 1 Fig. 10A")
    for _, r in sec.iterrows():
        lines.append(f"  {r['sector']:2s} (n={r['n']:3.0f}): "
                     f"{r['beta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
                     f"| published {r['published']:+.2f}")
    max_dev = (sec["beta"] - sec["published"]).abs().max()
    lines += [f"  max |re-estimated - published| = {max_dev:.3f}", ""]

    # ── field curves ──────────────────────────────────────────────
    s1 = fit_field(df, second_harmonic=False)
    s2 = fit_field(df, second_harmonic=True)
    xi_g = np.linspace(0, 2 * np.pi, 721)
    c1, se1 = beta_chi_curve(s1, xi_g, second_harmonic=False)
    c2, _ = beta_chi_curve(s2, xi_g, second_harmonic=True)

    # ── figure A: the decision figure ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.2))
    deg = np.degrees(xi_g)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.fill_between(deg, c1 - 1.96 * se1, c1 + 1.96 * se1,
                    color="C0", alpha=0.18,
                    label="first harmonic, 95% band")
    ax.plot(deg, c1, color="C0", lw=2,
            label=r"field: $\beta_\chi(\xi)=m_3+m_4\cos\xi+m_5\sin\xi$")
    ax.plot(deg, c2, color="0.5", lw=1.2, ls="--",
            label="balanced 2nd-harmonic ref. (n.s., p = 0.44)")
    ax.errorbar(sec["center"], sec["beta"],
                yerr=[sec["beta"] - sec["lo"], sec["hi"] - sec["beta"]],
                fmt="o", color="k", ms=5, capsize=3, lw=1.2,
                label="sectoral slopes (Paper 1 spec., HC3 95% CI)")
    # The disk is closed: repeat the E point at 360 deg (open marker) so
    # the figure shows that the periodic curve is bound at both ends.
    e = sec.loc[sec["sector"] == "E"].iloc[0]
    ax.errorbar([360.0], [e["beta"]],
                yerr=[[e["beta"] - e["lo"]], [e["hi"] - e["beta"]]],
                fmt="o", mfc="white", color="k", ms=5, capsize=3, lw=1.2)
    for _, r in sec.iterrows():
        ax.annotate(r["sector"], (r["center"], r["hi"] + 0.07),
                    ha="center", fontsize=9)
    ax.annotate("E", (360.0, e["hi"] + 0.07), ha="center", fontsize=9,
                color="0.4")
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel(r"task-space direction $\xi$ (deg)")
    ax.set_ylabel(r"return to depth $\beta_\chi$")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "field_vs_sectoral_beta_chi.png", dpi=150)
    plt.close(fig)

    # ── figure B: field vs observed wages, residual map ───────────
    from model.price_field import PriceField
    field = PriceField.from_results()
    grid = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(grid, grid)
    inside = np.hypot(X, Y) <= 1.0
    PI = np.where(inside, field.pi(np.arctan2(Y, X), np.hypot(X, Y)), np.nan)

    df["ln_pi_c"] = field.log_pi(df["xi"].to_numpy(), df["chi"].to_numpy())
    df["resid"] = df["ln_wage"] - df["ln_pi_c"]
    xs = df["chi"] * np.cos(df["xi"])
    ys = df["chi"] * np.sin(df["xi"])

    vmin, vmax = df["H_MEDIAN"].quantile([0.02, 0.98])
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].pcolormesh(X, Y, np.clip(PI, vmin, vmax), cmap="viridis",
                       norm=norm, shading="auto", rasterized=True)
    sc = axes[0].scatter(xs, ys, c=df["H_MEDIAN"], cmap="viridis", norm=norm,
                         s=14, ec="k", lw=0.25)
    axes[0].set_title(r"$\Pi(\mathbf{r})$ (field) and observed median "
                      "wages (points), shared scale")
    fig.colorbar(sc, ax=axes[0], shrink=0.85, label="USD/h (log scale)")

    lim = float(df["resid"].abs().quantile(0.98))
    sc2 = axes[1].scatter(xs, ys, c=df["resid"], cmap="RdBu_r",
                          vmin=-lim, vmax=lim, s=14, ec="k", lw=0.25)
    axes[1].set_title(r"residual $\ln w_{obs} - \ln\Pi(\mu_o)$")
    fig.colorbar(sc2, ax=axes[1], shrink=0.85)
    for ax in axes:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="0.4", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS / "field_vs_observed_wages.png", dpi=150)
    plt.close(fig)

    res_sec = df.groupby("sector_idx")["resid"].mean()
    lines.append("(B) Mean residual ln w_obs - ln Pi(centroid) per sector "
                 "(first-harmonic field):")
    for k, lab in enumerate(SECTOR_LABELS):
        lines.append(f"  {lab:2s}: {res_sec.get(k, np.nan):+.3f}")

    (RESULTS / "field_vs_paper1_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
