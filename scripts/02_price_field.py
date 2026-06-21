"""
02_price_field.py
-----------------
Builds the price-of-skill field Pi(r) from the coefficients estimated in
scripts/01_wage_field.py and validates it against observed wages.
The operated-regime illustration that previously lived here now uses the
calibrated technology in scripts/09_equilibrium_regime.py.

(A) Bundle wage validation (eq. wage-bundle).
    The model prices an occupation as its bundle, w_o = sum_t b_t Pi(r_t),
    not as Pi evaluated at the centroid. The coefficients were estimated at
    centroids, so Pi(centroid) ~ ln w is in-sample by construction; the
    bundle-integrated wage is a genuinely different predictor. We compare
    both against observed BLS log wages (N = 785) and report the Jensen gap
    ln w_bundle - ln Pi(centroid), which measures how much bundle dispersion
    matters for pricing.

Reads exclusively from data/ and results/wage_field_coefficients.csv.

Outputs:
    results/bundle_wage_validation.csv
    results/price_field_summary.txt
    results/price_field_map.png          Pi over the disk + occupations
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

RESULTS = REPO_ROOT / "results"

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

    # ── (C) domain diagnostics: extrapolation beyond occupational
    #        support (technology-free) ──────────────────────────────
    # The field is identified on occupational centroids (chi <= chi_max);
    # bundle pricing evaluates it at task locations, which extend to
    # chi = 1. Quantify the extrapolated region and its weight in bundle
    # wages. The model uses the extrapolated field (decision 2026-06-11:
    # the task layer is defined to the rim); the clipped variant is a
    # sensitivity, not the baseline.
    chi_max = float(sample["chi"].max())
    out_mass = (bundles.assign(out=bundles["chi"] > chi_max)
                .groupby("onet_code")
                .apply(lambda g: g.loc[g["out"], "b"].sum(),
                       include_groups=False))
    task_share_out = float((bundles["chi"] > chi_max).mean())

    chi_clip = np.minimum(bundles["chi"].to_numpy(), chi_max)
    pi_clip = field.pi(bundles["xi"].to_numpy(), chi_clip)
    w_clip = (pd.Series(bundles["b"].to_numpy() * pi_clip,
                        index=bundles["onet_code"].to_numpy())
              .groupby(level=0).sum().rename("w_bundle_clip"))
    df = df.merge(w_clip, left_on="onet_code", right_index=True, how="inner")
    df["ln_w_bundle_clip"] = np.log(df["w_bundle_clip"])
    d_ln = df["ln_w_bundle"] - df["ln_w_bundle_clip"]
    r2_b_clip = r2(df["ln_wage"], df["ln_w_bundle_clip"])
    pi_rim = float(field.pi(np.pi / 2, 1.0))
    pi_edge = float(field.pi(np.pi / 2, chi_max))
    lines += [
        f"(C) Domain diagnostics: occupational support chi <= {chi_max:.4f}, "
        "tasks to chi = 1",
        f"  bundle mass beyond support: mean {out_mass.mean():.4f}, "
        f"median {out_mass.median():.4f}, max {out_mass.max():.4f} per "
        f"occupation; unweighted task share {task_share_out:.4f}",
        f"  ln w_bundle (extrapolated) - ln w_bundle (clipped at chi_max): "
        f"mean {d_ln.mean():+.4f}, sd {d_ln.std():.4f}, "
        f"max {d_ln.max():+.4f}",
        f"  R2(ln w_obs ~ ln w_bundle): extrapolated {r2_b:.4f}, "
        f"clipped {r2_b_clip:.4f}",
        f"  Pi(90 deg, chi): {pi_edge:.1f} at chi_max -> {pi_rim:.1f} "
        f"at the rim (+{100 * (pi_rim / pi_edge - 1):.0f}%)",
        "",
    ]
    df.to_csv(RESULTS / "bundle_wage_validation.csv", index=False)

    # ── figures ───────────────────────────────────────────────────
    grid = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(grid, grid)
    inside = np.hypot(X, Y) <= 1.0
    XI = np.arctan2(Y, X)
    CHI = np.hypot(X, Y)
    PI = np.where(inside, field.pi(XI, CHI), np.nan)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    cf = ax.contourf(X, Y, PI, levels=24, cmap="viridis")
    occ = sample
    ax.scatter(occ["chi"] * np.cos(occ["xi"]), occ["chi"] * np.sin(occ["xi"]),
               s=4, c="white", alpha=0.45, lw=0)
    ax.set_aspect("equal")
    ax.set_title(r"Price of skill $\Pi(\mathbf{r})$ (USD/h)")
    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS / "price_field_map.png", dpi=150)
    plt.close(fig)

    (RESULTS / "price_field_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
