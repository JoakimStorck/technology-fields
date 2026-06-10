"""
02_price_field.py
-----------------
Builds the price-of-skill field Pi(r) from the coefficients estimated in
scripts/01_wage_field.py and runs two checks:

(A) Bundle wage validation (eq. wage-bundle).
    The model prices an occupation as its bundle, w_o = sum_t b_t Pi(r_t),
    not as Pi evaluated at the centroid. The coefficients were estimated at
    centroids, so Pi(centroid) ~ ln w is in-sample by construction; the
    bundle-integrated wage is a genuinely different predictor. We compare
    both against observed BLS log wages (N = 785) and report the Jensen gap
    ln w_bundle - ln Pi(centroid), which measures how much bundle dispersion
    matters for pricing.

(B) Regime machinery demonstration (eqs. regime-shift, soft-switch,
    displaced). With an ILLUSTRATIVE technology centered in the cognitive
    arc (parameters are placeholders until the Gmyrek calibration in
    script 03), we compute the operated share a(r), verify that takeover is
    ordered by price (capital moves first against the dear work), and rank
    occupations by displaced mass D_o.

Reads exclusively from data/ and results/wage_field_coefficients.csv.

Outputs:
    results/bundle_wage_validation.csv
    results/price_field_summary.txt
    results/price_field_map.png          Pi over the disk + occupations
    results/operated_share_demo.png      a(r) for the illustrative technology
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

from model.data import load_bundles, load_mincer_sample
from model.price_field import PriceField
from model.technology import Technology

RESULTS = REPO_ROOT / "results"

# Illustrative technology for the regime demo (placeholder until script 03
# calibrates against Gmyrek et al. exposure scores): replacing character,
# centered in the cognitive arc, moderate reach.
DEMO_TECH = Technology(xi_K=np.deg2rad(75), chi_K=0.45, z_K=0.35,
                       A_K=1.0, s_K=1.0)
DEMO_R = 18.0     # capital rental, same units as Pi (USD/hour-equivalent)
DEMO_TAU = 0.08   # within-cell margin width


def r2(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss = np.sum((y - yhat) ** 2)
    return 1 - ss / np.sum((y - y.mean()) ** 2)


def main() -> None:
    field = PriceField.from_results()
    sample = load_mincer_sample()
    bundles = load_bundles()
    lines: list[str] = []

    dep_dir, dep_amp = field.depth_return_direction()
    lines += [
        "Price field Pi(r), coefficients from S1_field:",
        f"  m0..m5 = {field.m0:+.4f} {field.m1:+.4f} {field.m2:+.4f} "
        f"{field.m3:+.4f} {field.m4:+.4f} {field.m5:+.4f}",
        f"  depth-return direction {dep_dir:.1f} deg, amplitude {dep_amp:.3f}",
        "",
    ]

    # ── (A) bundle wage validation ────────────────────────────────
    w_bundle = field.bundle_wage(bundles)
    df = sample[["onet_code", "Title", "xi", "chi", "ln_wage",
                 "H_MEDIAN", "TOT_EMP"]].copy()
    df["pi_centroid"] = field.pi(df["xi"].to_numpy(), df["chi"].to_numpy())
    df = df.merge(w_bundle, left_on="onet_code", right_index=True, how="inner")
    df["ln_pi_centroid"] = np.log(df["pi_centroid"])
    df["ln_w_bundle"] = np.log(df["w_bundle"])
    df["jensen_gap"] = df["ln_w_bundle"] - df["ln_pi_centroid"]

    r2_c = r2(df["ln_wage"], df["ln_pi_centroid"])
    r2_b = r2(df["ln_wage"], df["ln_w_bundle"])
    corr_b = float(np.corrcoef(df["ln_wage"], df["ln_w_bundle"])[0, 1])
    lines += [
        f"(A) Bundle wage validation, N = {len(df)}",
        f"  R2(ln w_obs ~ ln Pi(centroid))   = {r2_c:.4f}   "
        "(in-sample reference: estimation used centroids)",
        f"  R2(ln w_obs ~ ln w_bundle)       = {r2_b:.4f}   "
        f"(out-of-spec predictor; corr = {corr_b:.4f})",
        f"  Jensen gap ln w_bundle - ln Pi(mu): "
        f"mean {df['jensen_gap'].mean():+.4f}, "
        f"sd {df['jensen_gap'].std():.4f}, "
        f"min {df['jensen_gap'].min():+.4f}, "
        f"max {df['jensen_gap'].max():+.4f}",
        "",
    ]
    df.to_csv(RESULTS / "bundle_wage_validation.csv", index=False)

    # ── (B) regime machinery demo ─────────────────────────────────
    t = DEMO_TECH
    lines += [
        "(B) Regime demo - ILLUSTRATIVE technology (not calibrated):",
        f"  center (xi, chi) = ({np.rad2deg(t.xi_K):.0f} deg, {t.chi_K}), "
        f"z_K = {t.z_K}, A_K = {t.A_K}, s_K = {t.s_K}, "
        f"R = {DEMO_R}, tau = {DEMO_TAU}",
    ]

    # Price-ordering check: among tasks with equal phi, a(r) must increase
    # with Pi. Verified on the bundle tasks by rank correlation of a with Pi
    # within narrow phi bands.
    xi_t = bundles["xi"].to_numpy()
    chi_t = bundles["chi"].to_numpy()
    a_t = t.operated_share(xi_t, chi_t, field, DEMO_R, DEMO_TAU)
    pi_t = field.pi(xi_t, chi_t)
    phi_t = t.phi(xi_t, chi_t)
    bands = pd.qcut(phi_t, 20, duplicates="drop")
    rho = (pd.DataFrame({"a": a_t, "pi": pi_t, "band": bands})
           .groupby("band", observed=True)
           .apply(lambda d: d["a"].corr(d["pi"], method="spearman"),
                  include_groups=False))
    lines += [
        f"  price ordering within phi-bands: "
        f"median Spearman rho(a, Pi) = {rho.median():.3f} "
        f"(min {rho.min():.3f}) - capital moves first against the dear work",
    ]

    disp = t.displacement(bundles, field, DEMO_R, DEMO_TAU)
    disp = disp.merge(sample[["onet_code", "Title"]].set_index("onet_code"),
                      left_index=True, right_index=True, how="inner")
    disp["wage_loss_share"] = 1 - disp["w_retained"] / disp["w_pre"]
    disp = disp.sort_values("D_o", ascending=False)
    lines += ["", "  Top 10 by displaced mass D_o:"]
    for _, row in disp.head(10).iterrows():
        lines.append(f"    {row['Title'][:42]:42s}  D_o = {row['D_o']:.3f}  "
                     f"wage loss = {row['wage_loss_share']:.3f}")
    agg = float((disp["D_o"]).mean())
    lines += ["", f"  Mean displaced bundle share across occupations: {agg:.4f}"]

    # ── figures ───────────────────────────────────────────────────
    grid = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(grid, grid)
    inside = np.hypot(X, Y) <= 1.0
    XI = np.arctan2(Y, X)
    CHI = np.hypot(X, Y)
    PI = np.where(inside, field.pi(XI, CHI), np.nan)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    cf = ax.contourf(X, Y, PI, levels=24, cmap="viridis")
    ax.contour(X, Y, PI, levels=[DEMO_R / t.A_K], colors="white",
               linewidths=1.0, linestyles="--")
    occ = sample
    ax.scatter(occ["chi"] * np.cos(occ["xi"]), occ["chi"] * np.sin(occ["xi"]),
               s=4, c="white", alpha=0.45, lw=0)
    ax.set_aspect("equal")
    ax.set_title(r"Price of skill $\Pi(\mathbf{r})$ (USD/h); "
                 r"dashed: $\Pi = R/A_K$")
    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS / "price_field_map.png", dpi=150)
    plt.close(fig)

    A = np.where(inside,
                 t.operated_share(XI, CHI, field, DEMO_R, DEMO_TAU), np.nan)
    G = np.where(inside, t.grad_phi_norm(XI, CHI), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    cf0 = axes[0].contourf(X, Y, A, levels=24, cmap="magma")
    axes[0].set_title(r"Operated share $a(\mathbf{r})$ (illustrative tech.)")
    fig.colorbar(cf0, ax=axes[0], shrink=0.8)
    cf1 = axes[1].contourf(X, Y, G, levels=24, cmap="cividis")
    axes[1].set_title(r"$\|\nabla\phi_K\|$ - seeding ring at $z_K$")
    fig.colorbar(cf1, ax=axes[1], shrink=0.8)
    for ax in axes:
        px, py = t.p_K
        ax.plot(px, py, "w+", ms=10)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS / "operated_share_demo.png", dpi=150)
    plt.close(fig)

    (RESULTS / "price_field_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
