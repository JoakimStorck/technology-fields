"""
04_residual_structure.py
------------------------
The price field explains R^2 ~ 0.52 of the occupation wage cross-section,
so half the variance is residual. Visually (03, residual map) the
occupation-level picture looks far more complex than the sectoral
aggregates. This script asks WHICH KIND of complexity it is, by testing
two hypotheses with opposite implications:

(H1) SPATIAL structure: nearby occupations share residual sign. Then the
     field is missing systematic spatial wage structure, and Pi(r) is an
     inadequate summary of the location price.

(H2) COMPOSITIONAL structure: the residual lives WITHIN locations -
     occupations at (nearly) the same position differ in capability
     content and institutions. Then Pi(r) is a valid reduced form of the
     spatial component, and the residual is exactly what Paper 1's
     mediation analysis predicts: the market prices capabilities (S1
     foremost), and position proxies them imperfectly.

Tests:
  1. Moran's I of the field residuals e_o = ln w_o - ln Pi(mu_o), with
     k-nearest-neighbour weights in task space and a permutation null.
  2. A spatial correlogram: mean pair residual correlation by task-space
     distance bin, with a permutation band (positions shuffled jointly,
     as in Paper 1's pair-level tests).
  3. Compositional regressions: e_o on (a) cluster intensities S1, S2,
     A1, A2, (b) Job Family fixed effects, (c) both. For each: the share
     of residual variance absorbed and Moran's I of what remains. Note
     on (a): adjusting for a spatially SMOOTH covariate like S1 can
     RAISE Moran's I - it removes smooth variance from the denominator
     faster than locally correlated variance from the numerator - so the
     family battery is the informative one for the local clustering.

Reads exclusively from data/ (occupation_cluster_intensity.csv requires
re-running scripts/00_freeze_inputs.py against a geometry-of-work
checkout if not yet present).

Outputs:
    results/residual_structure_summary.txt
    results/residual_correlogram.png
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

from model.data import DATA, load_mincer_sample
from model.price_field import PriceField

RESULTS = REPO_ROOT / "results"
RNG = np.random.default_rng(42)
N_PERM = 999
K_NN = 10


def morans_i(e: np.ndarray, W: np.ndarray) -> float:
    """Moran's I with row-standardized weight matrix W (zero diagonal)."""
    z = e - e.mean()
    return float(len(e) / W.sum() * (z @ W @ z) / (z @ z))


def knn_weights(xy: np.ndarray, k: int) -> np.ndarray:
    """Row-standardized k-nearest-neighbour weights in task space."""
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    nn = np.argsort(d, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(d)), k)
    W[rows, nn.ravel()] = 1.0 / k
    return W


def moran_perm_p(e: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    obs = morans_i(e, W)
    null = np.array([morans_i(RNG.permutation(e), W) for _ in range(N_PERM)])
    p = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    return obs, p


def correlogram(e: np.ndarray, xy: np.ndarray, edges: np.ndarray):
    """Mean standardized residual cross-product by pair-distance bin."""
    z = (e - e.mean()) / e.std()
    iu = np.triu_indices(len(e), 1)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)[iu]
    prod = (z[:, None] * z[None, :])[iu]
    which = np.digitize(d, edges) - 1
    obs = np.array([prod[which == b].mean() if np.any(which == b) else np.nan
                    for b in range(len(edges) - 1)])
    # permutation band: shuffle residuals over occupations
    null = np.empty((N_PERM, len(obs)))
    for i in range(N_PERM):
        zp = RNG.permutation(z)
        pp = (zp[:, None] * zp[None, :])[iu]
        null[i] = [pp[which == b].mean() if np.any(which == b) else np.nan
                   for b in range(len(edges) - 1)]
    lo, hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
    return obs, lo, hi


def main() -> None:
    field = PriceField.from_results()
    df = load_mincer_sample()
    ci = pd.read_csv(DATA / "occupation_cluster_intensity.csv")
    df = df.merge(ci, on="onet_code", how="inner")
    lines: list[str] = [f"Sample: N = {len(df)}", ""]

    df["e"] = df["ln_wage"] - field.log_pi(df["xi"].to_numpy(),
                                           df["chi"].to_numpy())
    xy = np.column_stack([df["chi"] * np.cos(df["xi"]),
                          df["chi"] * np.sin(df["xi"])])
    e = df["e"].to_numpy()
    W = knn_weights(xy, K_NN)

    # 1) Moran's I of raw field residuals
    I_raw, p_raw = moran_perm_p(e, W)
    lines += [f"(1) Moran's I, field residuals (kNN k={K_NN}): "
              f"I = {I_raw:+.4f}, perm. p = {p_raw:.4f}",
              f"    residual variance share: {np.var(e)/np.var(df['ln_wage']):.3f}",
              ""]

    # 3) compositional regressions (run before plotting)
    X = sm.add_constant(df[["S1", "S2", "A1", "A2"]].astype(float))
    m = sm.OLS(df["e"].astype(float), X).fit(cov_type="HC3")
    e2 = m.resid.to_numpy()
    I_cap, p_cap = moran_perm_p(e2, W)
    lines += ["(3a) e ~ S1 + S2 + A1 + A2:", f"    R2 = {m.rsquared:.4f}"]
    for v in ["S1", "S2", "A1", "A2"]:
        lines.append(f"    {v}: {m.params[v]:+.4f} "
                     f"(se {m.bse[v]:.4f}, p {m.pvalues[v]:.4f})")
    lines += [f"    Moran's I after: I = {I_cap:+.4f}, p = {p_cap:.4f}",
              "    (I rises: S1 is spatially smooth; see docstring)", ""]

    fam = pd.get_dummies(df["Job Family"], drop_first=True).astype(float)
    m_f = sm.OLS(df["e"].astype(float), sm.add_constant(fam)).fit()
    I_fam, p_fam = moran_perm_p(m_f.resid.to_numpy(), W)
    lines += [f"(3b) e ~ Job Family FE ({df['Job Family'].nunique()} "
              f"families): R2 = {m_f.rsquared:.4f}",
              f"    Moran's I after: I = {I_fam:+.4f}, p = {p_fam:.4f}", ""]

    Xb = sm.add_constant(pd.concat(
        [fam, df[["S1", "S2", "A1", "A2"]].astype(float)], axis=1))
    m_b = sm.OLS(df["e"].astype(float), Xb).fit()
    e2 = m_b.resid.to_numpy()
    I_b, p_b = moran_perm_p(e2, W)
    tot = 1 - m_b.resid.var() / df["ln_wage"].var()
    lines += [f"(3c) e ~ Job Family FE + S1..A2: R2 = {m_b.rsquared:.4f}",
              f"    Moran's I after: I = {I_b:+.4f}, p = {p_b:.4f}",
              f"    total R2 vs ln w (field + family + capabilities): "
              f"{tot:.3f}", ""]

    # 2) correlograms, raw and capability-adjusted
    edges = np.linspace(0, 1.0, 11)
    mids = 0.5 * (edges[:-1] + edges[1:])
    obs1, lo1, hi1 = correlogram(e, xy, edges)
    obs2, lo2, hi2 = correlogram(e2, xy, edges)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, obs, lo, hi, title in [
            (axes[0], obs1, lo1, hi1, "field residuals"),
            (axes[1], obs2, lo2, hi2,
             "after family FE + capability adjustment")]:
        ax.fill_between(mids, lo, hi, color="0.85",
                        label="95% permutation band")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.plot(mids, obs, "o-", color="C0", lw=1.6, ms=4)
        ax.set_xlabel("task-space distance")
        ax.set_title(title)
    axes[0].set_ylabel("mean pair residual correlation")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "residual_correlogram.png", dpi=150)
    plt.close(fig)

    lines += ["(2) Correlogram saved to results/residual_correlogram.png",
              "    bins (distance, raw, adjusted):"]
    for d, a, b in zip(mids, obs1, obs2):
        lines.append(f"    {d:.2f}: {a:+.4f}  ->  {b:+.4f}")

    (RESULTS / "residual_structure_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
