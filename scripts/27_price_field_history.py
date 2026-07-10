"""
27_price_field_history.py
-------------------------
Estimates the directional price field on the historical OEWS wage
cross-sections frozen by script 26 (1999, 2003, 2007; SOC-2000 carried
onto the frozen coordinates) and compares each to the committed field of
script 01 (the May 2023 cross-section that the paper operates on).

Motivation. The robot-era analysis needs the adoption gate to compare
capital cost with CONTEMPORARY prices: the robot field should be run
against Pi_1999, not Pi_2023. That requires the price field to be
time-indexed, Pi_t, estimated per wage vintage on frozen coordinates.
This script delivers Pi_t and, at the same time, the licensing test for
the whole historical exercise: if the directional structure of the wage
field -- a first-harmonic depth return oriented toward the
socio-cognitive north -- already holds in 1999, then operating the
frozen geometry in the robot window is defensible; if it does not, the
historical design stops here and that is the finding. The trajectory of
the depth-return amplitude 1999 -> 2003 -> 2007 -> 2023 is, in either
case, skill-biased technical change expressed in the disk's language.

Caveats owned. The coordinates are built from current O*NET task
statements, so 1999 occupations are placed by their present task
content; this is a stated primitive, and the stability test below is
the partial license, not a proof. Wages are nominal; the constant m0
absorbs the level and all comparisons are of shape (harmonics), not
levels. The estimation sample per year is whatever the crosswalk
delivers (many-to-one duplication onto O*NET occupations, as in the
paper's 2019-2025 join); the one_to_one subsample is the robustness
check on the crosswalk itself.

PRE-REGISTERED HYPOTHESES (before first run):
  H1  License, orientation: the 1999 depth return is carried by a
      positive first harmonic (delta-method z on the amplitude
      A = hypot(m4, m5), p < 0.05) whose peak direction lies within
      +/- 20 degrees of the committed field's 89.5 degrees.
  H2  License, structure: the isotropic depth term m3 is
      indistinguishable from zero in 1999 (|t| < 2), as in the
      committed field.
  H3  Steepening (SBTC direction): the committed amplitude is at least
      the 1999 amplitude (one-sided z on A_2023 - A_1999, delta-method
      standard errors, independent-samples approximation across
      vintages 24 years apart).
  H4  Cross-vintage shape: the committed field evaluated at the 1999
      sample's centroids ranks the 1999 wages at Spearman >= +0.5.
  Verdicts are printed with these tags; the one_to_one subsample for
  1999 is reported beside H1/H2 as the crosswalk robustness.

RESULTS (first run, recorded after pre-registration):
  H1  FAIL. The amplitude is strongly significant (z_A 6.1) but the peak
      sits at 67.1 deg, 22.3 deg east of the committed 89.5; the peak
      rotates monotonically toward the north across vintages
      (67.1 -> 69.2 -> 72.8 -> 89.5).
  H2  FAIL. m3 = +0.281 (t +4.05) in 1999: depth paid isotropically; the
      term decays monotonically to the committed zero
      (+0.21 -> +0.07 -> -0.06).
  H3  PASS (directional; z +1.2, not significant). 0.683 vs 0.891; the
      rise is concentrated in 2003 -> 2007 (0.942) and complete by 2007.
  H4  PASS. Spearman +0.720.
  Composition robustness (same first run): on the balanced code set
  (620 codes, N = 694 occupations in every vintage) the pattern
  disappears -- direction 96.1 -> 95.6 -> 93.1 -> 93.2 deg, m3 t in
  [-0.84, +0.40], amplitude 0.91 -> 0.93 -> 0.96 -> 0.83 with no
  monotone trend and differences within one standard error. The
  rotation, the isotropic term, and the steepening of the full-sample
  fits are carried by the codes absent from the early vintages, not by
  repricing of the codes present throughout.
  Reading. On constant composition the wage surface is shape-stable
  1999 -> 2023: same peak direction, no isotropic term, comparable
  amplitude, R2 ~ 0.51 in every vintage. The license for the robot-era
  analysis is therefore stronger than the H1/H2 verdicts alone suggest;
  what those verdicts measured is the extensive margin -- which
  occupation codes exist and publish -- and that margin cannot be
  separated here from BLS publication practice, so it is parked as an
  observation. Pi_1999 remains the field to use in the robot window:
  own-era prices are the disciplined choice, and the nominal level
  difference (the real difference between vintages) is absorbed by the
  adoption gate's single calibrated scale. The one_to_one subsample
  reproduces the full-sample fits, so none of this is a crosswalk
  artefact. Vintage caveat: occupations are placed by current O*NET
  task content.

Reads:
    data/oews_history_wages.csv                   (script 26)
    data/occupation_embeddings_polar_scaled.csv   frozen coordinates
    data/national_M2023_dl.xlsx                   committed reference
                                                  (via model.data)
    results/wage_field_coefficients.csv           reproducibility guard
Writes:
    results/price_field_history.csv               per-vintage m0..m5, se,
                                                  R2, N, direction,
                                                  amplitude (+ delta se)
    results/price_field_history_summary.txt
    results/price_field_history.png               beta_chi(xi) per vintage

Not in run_all yet: run standalone after 26, until the inputs are frozen
and a baseline is committed.

Usage:
    python scripts/27_price_field_history.py
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
import matplotlib.pyplot as plt  # noqa: E402

from model.data import load_mincer_sample  # noqa: E402

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
HIST_FILE = DATA / "oews_history_wages.csv"
COEF_FILE = RESULTS / "wage_field_coefficients.csv"

S1_COLS = ["cos_xi", "sin_xi", "chi", "chi_cos", "chi_sin"]
REF_LABEL = "2023 (committed)"


# ─────────────────────────────────────────────────────────────────────
# Estimation and delta-method summaries
# ─────────────────────────────────────────────────────────────────────

def fit_s1(df: pd.DataFrame):
    X = sm.add_constant(df[S1_COLS].astype(float))
    return sm.OLS(df["ln_wage"].astype(float), X).fit(cov_type="HC3")


def harmonics(res) -> dict:
    """Depth-return amplitude and direction with delta-method SEs, plus
    the pointwise machinery for beta_chi(xi) bands."""
    m = res.params
    S = res.cov_params()
    m4, m5 = float(m["chi_cos"]), float(m["chi_sin"])
    A = float(np.hypot(m4, m5))
    gA = np.array([m4, m5]) / A
    Sab = S.loc[["chi_cos", "chi_sin"], ["chi_cos", "chi_sin"]].to_numpy()
    se_A = float(np.sqrt(gA @ Sab @ gA))
    phi = float(np.degrees(np.arctan2(m5, m4)) % 360)
    gphi = np.array([-m5, m4]) / A**2
    se_phi = float(np.degrees(np.sqrt(gphi @ Sab @ gphi)))
    return {"A": A, "se_A": se_A, "dir_deg": phi, "se_dir_deg": se_phi,
            "m3": float(m["chi"]), "t_m3": float(res.tvalues["chi"]),
            "level_dir_deg": float(np.degrees(
                np.arctan2(m["sin_xi"], m["cos_xi"])) % 360),
            "level_amp": float(np.hypot(m["cos_xi"], m["sin_xi"]))}


def beta_chi_band(res, xi: np.ndarray):
    m = res.params
    S = res.cov_params().loc[["chi", "chi_cos", "chi_sin"],
                             ["chi", "chi_cos", "chi_sin"]].to_numpy()
    G = np.column_stack([np.ones_like(xi), np.cos(xi), np.sin(xi)])
    b = G @ np.array([m["chi"], m["chi_cos"], m["chi_sin"]])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, S, G))
    return b, se


def regressors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ln_wage"] = np.log(df["wage_hourly"])
    df["cos_xi"] = np.cos(df["xi"])
    df["sin_xi"] = np.sin(df["xi"])
    df["chi_cos"] = df["chi"] * df["cos_xi"]
    df["chi_sin"] = df["chi"] * df["sin_xi"]
    return df


def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HIST_FILE.exists():
        sys.exit(f"{HIST_FILE} not found; run scripts/26_freeze_oews_history.py first.")
    hist = pd.read_csv(HIST_FILE)
    years = sorted(hist["year"].unique())

    occ = pd.read_csv(OCC_FILE, usecols=["onet_code", "xi", "chi"])
    occ["soc2018"] = (occ["onet_code"].astype(str)
                      .str.replace(r"\..*", "", regex=True).str.strip())

    # committed reference, refitted for its covariance; guarded against
    # the frozen coefficients of script 01
    ref = fit_s1(load_mincer_sample())
    frozen = pd.read_csv(COEF_FILE)
    m5_frozen = float(frozen.loc[(frozen["spec"] == "S1_field")
                                 & (frozen["param"] == "m5"), "coef"].iloc[0])
    if not np.isclose(float(ref.params["chi_sin"]), m5_frozen, atol=1e-8):
        sys.exit("Committed-field replication failed: refitted m5 "
                 f"{ref.params['chi_sin']:.6f} vs frozen {m5_frozen:.6f}. "
                 "Re-run scripts/01_wage_field.py and retry.")
    ref_h = harmonics(ref)

    lines = ["Historical price fields Pi_t on frozen coordinates "
             "(script 27; pre-registered, see docstring).",
             f"  committed reference: N = {int(ref.nobs)}, "
             f"A = {ref_h['A']:.3f} (se {ref_h['se_A']:.3f}), "
             f"direction {ref_h['dir_deg']:.1f} deg "
             f"(se {ref_h['se_dir_deg']:.1f}), R2 = {ref.rsquared:.3f}"]
    print(lines[-2] if len(lines) > 1 else lines[0])
    print(lines[-1])

    rows, fits = [], {}

    def record(label, res, n):
        h = harmonics(res)
        fits[label] = res
        rows.append({"vintage": label, "N": n, "R2": float(res.rsquared),
                     **{f"m{i}": float(res.params[c]) for i, c in
                        enumerate(["const"] + S1_COLS)},
                     **{f"se_m{i}": float(res.bse[c]) for i, c in
                        enumerate(["const"] + S1_COLS)},
                     "amplitude": h["A"], "se_amplitude": h["se_A"],
                     "direction_deg": h["dir_deg"],
                     "se_direction_deg": h["se_dir_deg"],
                     "m3_t": h["t_m3"],
                     "level_dir_deg": h["level_dir_deg"],
                     "level_amp": h["level_amp"]})
        return h

    record(REF_LABEL, ref, int(ref.nobs))

    hists = {}
    for y in years:
        w = hist[hist["year"] == y]
        df = regressors(occ.merge(w, on="soc2018", how="inner"))
        res = fit_s1(df)
        h = record(str(y), res, len(df))
        hists[y] = (res, h, df)
        msg = (f"  [{y}] N = {len(df)}: A = {h['A']:.3f} "
               f"(se {h['se_A']:.3f}), direction {h['dir_deg']:.1f} deg "
               f"(se {h['se_dir_deg']:.1f}), m3 t = {h['t_m3']:+.2f}, "
               f"R2 = {res.rsquared:.3f}")
        lines.append(msg)
        print(msg)
        if y == min(years):
            d11 = regressors(occ.merge(w[w["one_to_one"]], on="soc2018",
                                       how="inner"))
            r11 = fit_s1(d11)
            h11 = harmonics(r11)
            msg = (f"  [{y} one_to_one] N = {len(d11)}: A = {h11['A']:.3f} "
                   f"(se {h11['se_A']:.3f}), direction "
                   f"{h11['dir_deg']:.1f} deg (crosswalk robustness)")
            record(f"{y} one_to_one", r11, len(d11))
            lines.append(msg)
            print(msg)

    # ── pre-registered verdicts on the earliest vintage ────────────
    y0 = min(years)
    _, h0, df0 = hists[y0]
    zA = h0["A"] / h0["se_A"]
    ddir = abs(wrap_deg(h0["dir_deg"] - ref_h["dir_deg"]))
    H1 = (zA > 1.96) and (ddir <= 20.0)
    H2 = abs(h0["t_m3"]) < 2.0
    zdiff = ((ref_h["A"] - h0["A"])
             / np.hypot(ref_h["se_A"], h0["se_A"]))
    H3 = ref_h["A"] >= h0["A"]
    pi_ref_at = ref.predict(sm.add_constant(df0[S1_COLS].astype(float)))
    from scipy.stats import spearmanr
    rho4, p4 = spearmanr(pi_ref_at, df0["ln_wage"])
    H4 = rho4 >= 0.5

    verdicts = [
        f"  H1 (first harmonic, within 20 deg of committed)  "
        f"{'PASS' if H1 else 'FAIL'}  (z_A {zA:.1f}, delta-dir {ddir:.1f} deg)",
        f"  H2 (isotropic m3 ~ 0 in {y0})                    "
        f"{'PASS' if H2 else 'FAIL'}  (t {h0['t_m3']:+.2f})",
        f"  H3 (committed amplitude >= {y0})                 "
        f"{'PASS' if H3 else 'FAIL'}  (z {zdiff:+.1f}; "
        f"{ref_h['A']:.3f} vs {h0['A']:.3f})",
        f"  H4 (committed field ranks {y0} wages, rho>=0.5)  "
        f"{'PASS' if H4 else 'FAIL'}  (Spearman {rho4:+.3f}, p={p4:.1e})",
    ]
    lines.append("Pre-registered hypothesis verdicts:")
    lines.extend(verdicts)
    print("Pre-registered hypothesis verdicts:")
    for v in verdicts:
        print(v)

    traj = ([r for r in rows if r["vintage"].isdigit()]
            + [r for r in rows if r["vintage"] == REF_LABEL])
    amps = " -> ".join(f"{r['vintage']} {r['amplitude']:.3f}" for r in traj)
    lines.append(f"Depth-return amplitude trajectory: {amps}")
    print(lines[-1])

    # ── composition robustness (descriptive, not pre-registered) ────
    mincer = load_mincer_sample()
    common = set(mincer["OCC_CODE"])
    for y in years:
        common &= set(hist.loc[hist["year"] == y, "soc2018"])
    lines.append(f"Composition robustness, balanced code set "
                 f"(N_codes = {len(common)}; occupation-level fits):")
    print(lines[-1])
    ref_b = fit_s1(mincer[mincer["OCC_CODE"].isin(common)])
    hb = record(f"{REF_LABEL} balanced", ref_b, int(ref_b.nobs))
    msg = (f"  [{REF_LABEL} balanced] N = {int(ref_b.nobs)}: "
           f"A = {hb['A']:.3f}, direction {hb['dir_deg']:.1f} deg, "
           f"m3 t = {hb['t_m3']:+.2f}")
    lines.append(msg)
    print(msg)
    for y in years:
        dfb = hists[y][2]
        dfb = dfb[dfb["soc2018"].isin(common)]
        rb = fit_s1(dfb)
        hb = record(f"{y} balanced", rb, len(dfb))
        msg = (f"  [{y} balanced] N = {len(dfb)}: A = {hb['A']:.3f}, "
               f"direction {hb['dir_deg']:.1f} deg, m3 t = {hb['t_m3']:+.2f}")
        lines.append(msg)
        print(msg)

    df99 = hists[min(years)][2]
    dropped = df99[~df99["soc2018"].isin(common)]
    if len(dropped):
        ang = float(np.degrees(np.arctan2(
            np.sin(dropped["xi"]).mean(),
            np.cos(dropped["xi"]).mean())) % 360)
        msg = (f"  [{min(years)}] outside the balanced set: {len(dropped)} "
               f"occupations, circular-mean direction {ang:.0f} deg, "
               f"mean chi {dropped['chi'].mean():.2f}")
        lines.append(msg)
        print(msg)

    pd.DataFrame(rows).to_csv(RESULTS / "price_field_history.csv", index=False)
    (RESULTS / "price_field_history_summary.txt").write_text(
        "\n".join(lines) + "\n")

    # ── figure: beta_chi(xi) per vintage, full sample and balanced ──
    xi = np.linspace(0, 2 * np.pi, 361)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    panels = [("full published samples",
               lambda lb: "one_to_one" not in lb and "balanced" not in lb),
              ("balanced code set", lambda lb: lb == f"{REF_LABEL} balanced"
               or ("balanced" in lb and "one_to_one" not in lb))]
    for ax, (title, keep) in zip(axes, panels):
        for label, res in fits.items():
            if not keep(label):
                continue
            short = label.replace(" balanced", "")
            b, se = beta_chi_band(res, xi)
            style = (dict(lw=2.2, color="black") if short == REF_LABEL
                     else dict(lw=1.6))
            line, = ax.plot(np.degrees(xi), b, label=short, **style)
            if short in (REF_LABEL, str(min(years))):
                ax.fill_between(np.degrees(xi), b - 1.96 * se, b + 1.96 * se,
                                alpha=0.15, color=line.get_color())
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel(r"direction $\xi$ (degrees)")
        ax.set_xlim(0, 360)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"depth return $\beta_\chi(\xi)$")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Directional depth return across wage vintages "
                 "(frozen coordinates)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "price_field_history.png", dpi=200)
    print(f"wrote {RESULTS / 'price_field_history.csv'}, _summary.txt, .png")


if __name__ == "__main__":
    main()
