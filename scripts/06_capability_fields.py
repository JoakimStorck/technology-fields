"""
06_capability_fields.py
-----------------------
The capability plane: the structural postulate of the paper tested and
measured. The postulate (Sec. 3 of the paper) is that the capability
content of work is, to first order, a two-dimensional plane,

    v_o  =  vbar + chi_o (cos xi_o  u1 + sin xi_o  u2) + eps_o,

with the task disk as its polar map: chi is amplitude, xi is phase.
This is the structure that jointly generates the Paper 1 findings
(the cosine law for deviation similarity, radial intensification
without profile-shape change, the absence of an isotropic depth
premium, and the first-harmonic price field).

(A) Rank-2 test (the postulate tested directly). Per descriptor family
    (35 Skills, 52 Abilities, N = 878): PCA of the deviations
    v_o - vbar; report the variance shares of the leading components,
    and the alignment of the leading two-dimensional score plane with
    the disk coordinates (x, y) = (chi cos xi, chi sin xi): canonical
    correlations and R2 of x and y on the two scores. The disk
    coordinates are built from task text alone, capabilities enter no
    step of their construction, so alignment is a genuine test.

(B) Capability requirement fields q_k. Two variants per cluster
    k in {S1, S2, A1, A2}:

      Q_plane  (theory):   q_k = vbar_k + u1_k x + u2_k y
                           (3 parameters; the exact plane form)
      Q1_field (measured): full first-harmonic fit with level harmonics
                           and isotropic depth term (6 parameters)

    plus the texture ledger: delta R2 of plane -> measured -> second
    harmonics, per cluster. Exported Q_plane rows carry a1 = a2 = a3
    = 0 so that model.capability_field evaluates both specs with one
    formula.

(C) Cluster weights v_k. Replicates the global mediation regression of
    Paper 1 on the frozen inputs:

        ln w ~ cos xi + sin xi + chi + rle_mean + S1 + S2 + A1 + A2,
        N = 785, HC3.

    The direction terms are part of the replicated specification (the
    variant without them shifts chi to -0.03 and S2 to +0.09 and does
    not replicate). Replication targets from Paper 1: beta_chi = -0.08
    (p = 0.24), S1 = +0.33 (p < 0.001), S2 = +0.05 (p = 0.02), A1/A2
    not significant. The deficit gate of the model is defined over the
    PRICED clusters {S1, S2}: unpriced capabilities lie in the kernel
    of the price functional and cannot gate, and the sign restriction
    v_k >= 0 excludes carrying noise estimates (A1 = -0.03) as theory.
    All four estimates are exported; the gate reads the priced two.

Reads exclusively from data/ (frozen by 00_freeze_inputs.py).
Writes:
    results/capability_field_coefficients.csv
    results/capability_field_summary.txt

Usage:
    python scripts/06_capability_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.data import load_mincer_sample  # noqa: E402

RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

CLUSTERS = ["S1", "S2", "A1", "A2"]
NAME_MAP = {
    "const": "a0", "cos_xi": "a1", "sin_xi": "a2",
    "chi": "a3", "chi_cos": "a4", "chi_sin": "a5",
    "x": "a4", "y": "a5",
}
FULL_COLS = ["cos_xi", "sin_xi", "chi", "chi_cos", "chi_sin"]


def add_regressors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cos_xi"] = np.cos(df["xi"])
    df["sin_xi"] = np.sin(df["xi"])
    df["x"] = df["chi"] * df["cos_xi"]
    df["y"] = df["chi"] * df["sin_xi"]
    df["chi_cos"] = df["x"]
    df["chi_sin"] = df["y"]
    df["cos2"] = np.cos(2 * df["xi"])
    df["sin2"] = np.sin(2 * df["xi"])
    df["chi_cos2"] = df["chi"] * df["cos2"]
    df["chi_sin2"] = df["chi"] * df["sin2"]
    return df


def fit(df: pd.DataFrame, y: str, cols: list[str]):
    X = sm.add_constant(df[cols].astype(float))
    return sm.OLS(df[y].astype(float), X).fit(cov_type="HC3")


def canonical_correlations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Canonical correlations between column spaces of A and B
    (both mean-centered, n x p)."""
    Qa, _ = np.linalg.qr(A - A.mean(axis=0))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)


def rank2_test(wide: pd.DataFrame, coords: pd.DataFrame,
               family: str, lines: list[str]) -> None:
    df = coords.merge(wide, on="onet_code", how="inner")
    desc_cols = [c for c in wide.columns if c != "onet_code"]
    V = df[desc_cols].to_numpy(dtype=float)
    D = V - V.mean(axis=0)
    s = np.linalg.svd(D, compute_uv=False)
    share = s**2 / np.sum(s**2)
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    scores = U[:, :2] * sv[:2]
    XY = df[["x", "y"]].to_numpy(dtype=float)
    cc = canonical_correlations(scores, XY)
    r2 = {}
    for j, name in enumerate(["x", "y"]):
        m = sm.OLS(XY[:, j], sm.add_constant(scores)).fit()
        r2[name] = m.rsquared
    lines += [
        f"== Rank-2 test, {family} ({len(desc_cols)} descriptors, "
        f"N = {len(df)})",
        f"   variance share of deviations: PC1 {share[0]:.3f}, "
        f"PC2 {share[1]:.3f}, PC1+PC2 {share[0] + share[1]:.3f} "
        f"(PC3 {share[2]:.3f}, PC4 {share[3]:.3f})",
        f"   alignment of the leading score plane with the disk (x, y): "
        f"canonical correlations {cc[0]:.3f}, {cc[1]:.3f}",
        f"   R2(x ~ scores) = {r2['x']:.3f}, R2(y ~ scores) = {r2['y']:.3f}",
        "",
    ]


def main() -> None:
    occ = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv")
    ci = pd.read_csv(DATA / "occupation_cluster_intensity.csv")
    full = add_regressors(
        occ.merge(ci, on="onet_code", how="inner")
           .dropna(subset=["xi", "chi"] + CLUSTERS)
           .reset_index(drop=True))
    mincer = add_regressors(
        load_mincer_sample().merge(ci, on="onet_code", how="inner")
                            .dropna(subset=CLUSTERS))

    lines: list[str] = [
        f"Capability plane: estimation sample N = {len(full)} "
        f"(coordinates + intensities; wages not required)",
        f"Mincer subsample for v_k: N = {len(mincer)}",
        "",
    ]
    rows: list[dict] = []

    # ── (A) rank-2 test ───────────────────────────────────────────
    coords = full[["onet_code", "x", "y"]]
    for fname, family in [("occupation_skills_levels.csv", "Skills"),
                          ("occupation_abilities_levels.csv", "Abilities")]:
        rank2_test(pd.read_csv(DATA / fname), coords, family, lines)

    # ── (B) capability fields: plane and measured variants ───────
    for k in CLUSTERS:
        mp = fit(full, k, ["x", "y"])
        mf = fit(full, k, FULL_COLS)
        m_lvl = fit(full, k, FULL_COLS + ["cos2", "sin2"])
        m2b = fit(full, k, FULL_COLS + ["cos2", "sin2",
                                        "chi_cos2", "chi_sin2"])
        w_bal = m2b.wald_test("(chi_cos2 = 0), (chi_sin2 = 0)", scalar=True)

        lines.append(f"== {k}")
        lines.append(f"   Q_plane (theory, 3 params):  R2 = {mp.rsquared:.4f}")
        for var in mp.params.index:
            rows.append(dict(spec="Q_plane", cluster=k,
                             param=NAME_MAP[var], variable=var,
                             coef=mp.params[var], se=mp.bse[var],
                             t=mp.tvalues[var], p=mp.pvalues[var]))
            lines.append(f"      {NAME_MAP[var]:3s} ({var:9s}) "
                         f"{mp.params[var]:+8.4f}  (se {mp.bse[var]:.4f}, "
                         f"p {mp.pvalues[var]:.4f})")
        for var in ["cos_xi", "sin_xi", "chi"]:
            rows.append(dict(spec="Q_plane", cluster=k,
                             param=NAME_MAP[var], variable=var,
                             coef=0.0, se=np.nan, t=np.nan, p=np.nan))
        pole = float(np.degrees(np.arctan2(
            mp.params["y"], mp.params["x"])) % 360)
        amp = float(np.hypot(mp.params["x"], mp.params["y"]))
        lines.append(f"      pole direction {pole:.1f} deg, "
                     f"amplitude {amp:.3f}")

        lines.append(f"   Q1_field (measured, 6 params): "
                     f"R2 = {mf.rsquared:.4f}")
        for var in mf.params.index:
            rows.append(dict(spec="Q1_field", cluster=k,
                             param=NAME_MAP[var], variable=var,
                             coef=mf.params[var], se=mf.bse[var],
                             t=mf.tvalues[var], p=mf.pvalues[var]))
            lines.append(f"      {NAME_MAP[var]:3s} ({var:9s}) "
                         f"{mf.params[var]:+8.4f}  (se {mf.bse[var]:.4f}, "
                         f"p {mf.pvalues[var]:.4f})")

        lines += [
            "   texture ledger (delta R2):",
            f"      plane -> measured (level harmonics + isotropic depth): "
            f"{mf.rsquared - mp.rsquared:+.4f}",
            f"      measured -> + level 2nd harmonic: "
            f"{m_lvl.rsquared - mf.rsquared:+.4f}",
            f"      + interaction 2nd harmonic: "
            f"{m2b.rsquared - m_lvl.rsquared:+.4f}   "
            f"(balanced test p = {float(w_bal.pvalue):.4f})",
            "",
        ]

    # ── (C) cluster weights v_k ───────────────────────────────────
    med_cols = ["cos_xi", "sin_xi", "chi", "rle_mean"] + CLUSTERS
    med = fit(mincer, "ln_wage", med_cols)
    lines += [
        "== V_mediation: ln w ~ cos xi + sin xi + chi + rle_mean "
        f"+ S1 + S2 + A1 + A2 (N = {int(med.nobs)}, HC3)   "
        f"R2 = {med.rsquared:.4f}",
        "   Replication targets (Paper 1): chi -0.08 (p 0.24), "
        "S1 +0.33 (p < 0.001), S2 +0.05 (p 0.02), A1/A2 n.s.",
    ]
    for var in med.params.index:
        lines.append(f"   {var:9s}  {med.params[var]:+8.4f}  "
                     f"(se {med.bse[var]:.4f}, p {med.pvalues[var]:.4f})")
        if var in CLUSTERS:
            rows.append(dict(spec="V_mediation", cluster=var, param="v_k",
                             variable=var, coef=med.params[var],
                             se=med.bse[var], t=med.tvalues[var],
                             p=med.pvalues[var]))
    lines += [
        "   Gate weights (theory): priced clusters {S1, S2}; A1/A2 lie in "
        "the kernel of the price functional and do not gate.",
        "",
    ]

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(
        RESULTS / "capability_field_coefficients.csv", index=False)
    (RESULTS / "capability_field_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))

    # smoke test: the model class must load and evaluate what was written
    from model.capability_field import CapabilityField  # noqa: E402
    cf = CapabilityField.from_results()
    print("Smoke test, q_k(90 deg, 0.5), Q_plane:",
          {k: round(float(cf.q(k, np.pi / 2, 0.5)), 3) for k in CLUSTERS},
          "| gate weights:", {k: round(v, 3) for k, v in cf.v_gate.items()})


if __name__ == "__main__":
    main()
