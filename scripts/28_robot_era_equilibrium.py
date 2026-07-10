"""
28_robot_era_equilibrium.py
---------------------------
Runs the industrial-robot field through the full worker-layer equilibrium
in its own era: the price field Pi_1999 (script 27), the 1999 employment
weights (script 26), and the Webb-located field (script 23), with the
adoption-gate scale calibrated to an external Acemoglu-Restrepo moment
instead of chosen. This replaces the manuscript's read-off-the-geometry
implication ("weak binding in the west") with a computed result, and runs
the collinearity pre-check that gates the 1999-2007 directional test of
script 29.

Design ("each wave meets its own window"). Everything era-consistent:
Pi_1999 estimated on 1999 wages over frozen coordinates; L0 from 1999
OEWS employment; ell by the committed SD rule on the era occupation set;
mobility (c, kappa) by the committed rule on era baseline values; R, tau,
beta, gamma at the committed economy constants. The cognitive
configuration of script 09 is re-run in the same script as a
reproduction guard (frozen unbound share 0.68, labour share 0.626), so
the robot numbers sit beside their comparator apples to apples.

Calibration. Only the gate margin s_K phi_K - R/Pi is identified, so the
field has one free scale; we calibrate A_K by bisection so that the
aggregate displaced task-mass share sum_o L_o D_o equals the moment

    MOMENT = 0.0030

from Acemoglu and Restrepo (2020): roughly 400,000 jobs on a ~135M
employment base by 2007 (one more robot per thousand workers reduces the
employment-to-population ratio by about 0.2 pp; the US stock rose by
about one per thousand over 1993-2007). Task mass is equated to jobs at
the aggregate; A&R's estimate is a net local-equilibrium employment
effect while the model's object is gross stripped mass, so matching
gross to net makes the calibrated field conservative. Sensitivity at
half and double the moment (0.0015, 0.0060) is reported.

PRE-REGISTERED HYPOTHESES (before first run):
  H1  Calibration: aggregate displaced mass is monotone in A_K and the
      bisection hits the moment within 1e-6 relative on A_K in
      [0.02, 50].
  H2  Binding (the Section 8.2 claim, now at risk): the robot unbound
      share of seeded mass EXCEEDS the cognitive field's committed 0.68.
      Honest note: the deficit gate is one-sided in requirements and the
      symmetric FIT kernel is local, so nearby western occupations may
      bind the seed; failure is a live possibility and would force the
      manuscript sentence to be rewritten, which is what the test is
      for.
  H3  Scale structure: the robot unbound share moves by fewer than 5
      percentage points between the half and double moment (structural
      property of binding, not a calibration artefact; mirrors script
      17).
  H4  Pre-check gate for script 29: |Spearman(proj_robot at Pi_1999,
      ln w_1999)| < 0.35, leaving identifying variation for the
      directional wage test (the cognitive case's collinearity in its
      own window was -0.57, which destroyed identification).
  Reported without hypotheses: labour share pre -> automation ->
  reinstatement (its size is itself informative about the aggregate
  scale of the robot shock), re-sorted mass, manual-arc employment
  change, bound-refill value per seeded unit beside the cognitive
  comparator, top gainers and losers.

RESULTS (first run, recorded after pre-registration):
  H1  PASS. A_K = 1.477 at the 0.0030 moment (close to the manuscript's
      illustrative 1.2, an unintended sanity stamp on that rule).
  H2  PASS. Robot unbound share 93 percent against the cognitive 68;
      only 7 percent of seeded mass binds, and it binds on the
      technician arc (mechanical and automotive engineering
      technicians, machinists, equipment repairers) -- the integrators.
  H3  PASS. 93.4 vs 92.9 percent between half and double moment.
  H4  PASS. Collinearity -0.148 (proj), +0.025 (dW_bundle): the robot
      window has the identifying variation the AI window lacked.
  Task-layer incidence is face-valid: displacement concentrates 21x in
  the western arc (mean D_o 0.0136 vs 0.00065 outside), led by welders,
  machine setters, rebar workers, dredge operators at 2.7-3.5 percent
  of bundle mass and -0.3 to -0.4 log points of bundle wage pressure.
  DISCOVERY, first run: the equilibrium dL is NOT the technology's --
  57 percent of mass relocates under a 0.3 percent shock, and dL is
  uncorrelated with the shock's own objects (corr with D_o +0.07, with
  B_o -0.05). The observed L0 is not a fixed point of the sorting
  layer, so the solved equilibrium carries a large baseline drift. The
  zero-field reference below was added in response: the technology's
  employment effect is L*(tech) - L*(0), reported separately from the
  drift L*(0) - L0, for BOTH configurations -- the cognitive case's
  re-sorting and congestion narrative is subject to the same audit.
  Zero-field decomposition (recorded, full grid). Cognitive: drift
  57.9 percent, technology component 20.0 percent; corr(dL_shock, D_o)
  -0.90, corr(dL_shock, B_o) +0.65; manual-arc technology component
  +0.104 of the total +0.150. The technology's true employment losers
  are the exposed analytical occupations (industrial ecologists,
  environmental scientists, biostatisticians, chief sustainability
  officers) and its true gainers low-exposure interpersonal and manual
  work (maintenance and repair workers, substitute teachers, clergy);
  the naive dL lists of script 09 carry drift, not the shock. Robot:
  drift 56.9 percent, technology component 0.13 percent;
  corr(dL_shock, D_o) -0.95; concentrated losses among western
  operatives (boilermakers, aircraft assemblers, millwrights), diffuse
  absorption into services, manual-arc technology component -0.0012;
  unbound share 92.8 percent also at observed L0. The audit of scripts
  09 and 19 (re-sorting and congestion measured L0 -> L*) is opened as
  its own task; no manuscript numbers are changed by this script.

Reads:
    results/price_field_history.csv      Pi_1999 coefficients (script 27)
    data/oews_history_wages.csv          1999 wages and employment (26)
    results/webb_calibration.csv         robot centre and reach (23)
    data/occupation_embeddings_polar_scaled.csv, occupation_cluster_intensity.csv
    results/wage_field_coefficients.csv  committed field (comparator)
Writes:
    results/robot_era_equilibrium.csv
    results/robot_era_equilibrium_summary.txt

Not in run_all yet: run standalone after 26-27. SMOKE=1 runs a coarse
grid without multistart (mechanics check only, not a result).

Usage:
    python scripts/28_robot_era_equilibrium.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.capability_field import CapabilityField          # noqa: E402
from model.data import load_bundles                        # noqa: E402
from model.equilibrium import Equilibrium                  # noqa: E402
from model.price_field import PriceField                   # noqa: E402
from model.regime import DiskGrid, RegimeInputs, regime    # noqa: E402
from model.technology import Technology                    # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_setup = _load("_setup.py")
cst = _load("11_centroid_shift_test.py")

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
MOMENT = 0.0030            # A&R displaced share; see docstring
MOMENT_BAND = (0.0015, 0.0060)
FROZEN_COG_UNBOUND = 0.68  # script 09 committed corner
SMOKE = os.environ.get("SMOKE", "") == "1"
GRID = dict(n_ang=60, n_rad=20) if SMOKE else dict(n_ang=120, n_rad=40)
VINTAGE = 1999


# ─────────────────────────────────────────────────────────────────────
# Era inputs
# ─────────────────────────────────────────────────────────────────────

def era_price_field(vintage: int) -> PriceField:
    df = pd.read_csv(RESULTS / "price_field_history.csv")
    row = df.loc[df["vintage"] == str(vintage)]
    if len(row) != 1:
        sys.exit(f"price_field_history.csv: expected one row for vintage "
                 f"{vintage}, found {len(row)}; run scripts/27 first.")
    r = row.iloc[0]
    return PriceField(*(float(r[f"m{i}"]) for i in range(6)))


def build_era_inputs(vintage: int):
    """RegimeInputs with Pi_vintage and era employment shares, mirroring
    _setup.build_inputs occupation for occupation but with the frozen
    historical window supplying wages and employment."""
    cap = CapabilityField.from_results()
    field = era_price_field(vintage)
    bundles = load_bundles()

    occ = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv",
                      usecols=["onet_code", "xi", "chi", "Job Family",
                               "Title"])
    occ = occ.merge(pd.read_csv(DATA / "occupation_cluster_intensity.csv"),
                    on="onet_code", how="inner")

    hist = pd.read_csv(DATA / "oews_history_wages.csv")
    h = hist[hist["year"] == vintage].set_index("soc2018")
    occ["OCC_CODE"] = (occ["onet_code"].astype(str)
                       .str.replace(r"\..*", "", regex=True).str.strip())
    occ["L0_emp"] = occ["OCC_CODE"].map(h["tot_emp"])
    occ["w_era"] = occ["OCC_CODE"].map(h["wage_hourly"])
    occ = occ.dropna(subset=["L0_emp", "w_era", "xi", "chi",
                             "S1", "S2", "A1", "A2"])
    occ = occ[occ["L0_emp"] > 0].copy()
    occ["L0"] = occ["L0_emp"] / occ["L0_emp"].sum()

    bundles = bundles[bundles["onet_code"].isin(occ["onet_code"])].copy()
    occ = occ.set_index("onet_code")
    grid = DiskGrid.build(**GRID)
    inp = RegimeInputs(bundles=bundles, occ=occ, field=field, cap=cap,
                       grid=grid)
    return inp, occ["L0"].to_numpy(), occ


def robot_tech(A_K: float) -> Technology:
    w = pd.read_csv(RESULTS / "webb_calibration.csv")
    r = w.loc[w["field"] == "webb_robot"].iloc[0]
    return Technology(xi_K=float(np.radians(r["xi_K_deg"])),
                      chi_K=float(r["chi_K"]), z_K=float(r["z_K"]),
                      A_K=A_K, s_K=1.0)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────

def displaced_share(inp, L0, A_K: float) -> float:
    tech = robot_tech(A_K)
    bx = inp.bundles
    a = tech.operated_share(bx["xi"].to_numpy(), bx["chi"].to_numpy(),
                            inp.field, R, TAU)
    row_of = pd.Index(inp.occ_codes()).get_indexer(
        bx["onet_code"].to_numpy())
    D_o = np.bincount(row_of, weights=bx["b"].to_numpy() * a,
                      minlength=len(L0))
    return float(np.sum(L0 * D_o))


def calibrate_A(inp, L0, target: float, lo=0.02, hi=50.0):
    f_lo, f_hi = (displaced_share(inp, L0, lo),
                  displaced_share(inp, L0, hi))
    monotone = f_hi > f_lo
    if not (f_lo < target < f_hi):
        sys.exit(f"Calibration bracket failed: displaced share at A_K="
                 f"{lo} is {f_lo:.2e}, at A_K={hi} is {f_hi:.2e}, target "
                 f"{target:.2e}.")
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        f = displaced_share(inp, L0, mid)
        if abs(f - target) / target < 1e-6:
            return mid, f, monotone
        if f < target:
            lo = mid
        else:
            hi = mid
    return mid, f, monotone


# ─────────────────────────────────────────────────────────────────────
# One full configuration
# ─────────────────────────────────────────────────────────────────────

def run_config(name, inp, L0, tech, ell, lines):
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                      survival=True)
    eq0.L0 = L0
    _, _, W0 = eq0.density_and_value(L0)
    c, kappa, dmed = _setup.mobility_reference(W0, eq0.d)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    if SMOKE:
        out = eq.solve(c, kappa)
        spread, allconv = np.nan, out.converged
    else:
        _, spread, allconv = eq.multistart(c, kappa, n_random=3)
        out = eq.solve(c, kappa)

    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA,
                  survival=True)
    diag0 = regime(inp, tech, out.L, R, TAU, 0.0, ell, BETA,
                   survival=True)
    dL = out.L - L0

    # zero-field reference: the same worker layer with no technology.
    # The observed L0 need not be a fixed point of the sorting model, so
    # the equilibrium drifts even with no shock; the technology's own
    # employment effect is the difference between the two equilibria.
    tech0 = Technology(xi_K=tech.xi_K, chi_K=tech.chi_K, z_K=tech.z_K,
                       A_K=1e-9, s_K=tech.s_K)
    eq_ref = Equilibrium(inp, tech0, R, TAU, GAMMA, ell, BETA, wedge=None,
                         survival=True)
    eq_ref.L0 = L0
    ref = eq_ref.solve(c, kappa)
    drift = ref.L - L0
    dL_shock = out.L - ref.L
    xi_o = inp.occ["xi"].to_numpy()
    manual = (xi_o > np.radians(135)) & (xi_o < np.radians(225))
    titles = inp.occ["Title"].to_numpy()
    unbound_share = diag["unbound_mass"] / diag["M"] if diag["M"] > 0 else np.nan
    refill_per_seed = ((diag["M"] - diag["unbound_mass"]) / diag["M"]
                       if diag["M"] > 0 else np.nan)
    # Pi-value of bound reinstatement per unit of seeded mass
    refill_value = float(np.sum(diag["B_o"])) / diag["M"] if diag["M"] > 0 else np.nan

    lines += [
        f"=== {name} ===",
        f"  occupations {len(L0)}, tasks {len(inp.bundles)}, grid "
        f"{inp.grid.xi.size} cells; ell {ell:.4f}; "
        f"kappa {kappa:.3f}, c {c:.3f}, median move {dmed:.3f}",
        f"  converged {out.converged} in {out.iters} iters (residual "
        f"{out.residual:.1e}); multistart spread {spread:.1e}, all "
        f"converged {allconv}",
        f"  displaced mass sum L*D {diag['Delta_Gamma_D']:.6f}; seeded "
        f"M {diag['M']:.6f}",
        f"  labor share: pre 1.000 -> automation {diag0['labor_share']:.4f}"
        f" -> reinstatement {diag['labor_share']:.4f}",
        f"  unbound share of seeded mass: {100 * unbound_share:.0f}%  "
        f"(bound {100 * refill_per_seed:.0f}%)",
        f"  re-sorting vs L0: sum|dL| {np.abs(dL).sum():.4f} "
        f"({100 * np.abs(dL).sum() / 2:.1f}% of mass relocated)",
        f"  zero-field reference (converged {ref.converged}): baseline "
        f"drift sum|L*0 - L0| {np.abs(drift).sum():.4f} "
        f"({100 * np.abs(drift).sum() / 2:.1f}%); technology component "
        f"sum|L* - L*0| {np.abs(dL_shock).sum():.4f} "
        f"({100 * np.abs(dL_shock).sum() / 2:.2f}%)",
        f"  shock-component alignment: corr(dL_shock, D_o) "
        f"{np.corrcoef(dL_shock, diag['D_o'])[0, 1]:+.2f}, "
        f"corr(dL_shock, B_o) "
        f"{np.corrcoef(dL_shock, diag['B_o'])[0, 1]:+.2f}",
        f"  manual-arc (135-225 deg): total dL {dL[manual].sum():+.5f}, "
        f"technology component {dL_shock[manual].sum():+.5f}",
        "  top gainers (technology component L* - L*0):",
    ]
    for i in np.argsort(dL_shock)[::-1][:5]:
        lines.append(f"    {titles[i][:44]:44s} dL {dL_shock[i]:+.4e}")
    lines.append("  top losers (technology component):")
    for i in np.argsort(dL_shock)[:5]:
        lines.append(f"    {titles[i][:44]:44s} dL {dL_shock[i]:+.4e}")
    lines.append("")
    return dict(out=out, diag=diag, diag0=diag0, dL=dL,
                dL_shock=dL_shock, drift=drift,
                unbound_share=float(unbound_share),
                refill_value=refill_value)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines = ["Robot field through the 1999 equilibrium (script 28; "
             "pre-registered, see docstring)."]
    if SMOKE:
        lines.append("SMOKE MODE: coarse grid, no multistart -- "
                     "mechanics check, NOT A RESULT.")

    # era inputs and calibration
    inp99, L0_99, occ99 = build_era_inputs(VINTAGE)
    ell99 = _setup.interpretable_ell(inp99)
    A_star, share_at, monotone = calibrate_A(inp99, L0_99, MOMENT)
    H1 = monotone and abs(share_at - MOMENT) / MOMENT < 1e-6
    lines += [
        f"Calibration to the A&R moment: target displaced share "
        f"{MOMENT:.4f}",
        f"  calibrated A_K = {A_star:.4f}  (displaced share "
        f"{share_at:.6f}); Webb-fit exposure-scale A was 0.289, the "
        f"manuscript's illustrative choice 1.2",
        "",
    ]

    # reproduction guard: the committed cognitive configuration
    inp23, L0_23, _ = _setup.build_inputs(**GRID)
    ell23 = _setup.interpretable_ell(inp23)
    cog = run_config("cognitive @ Pi_2023 (reproduction guard)",
                     inp23, L0_23, _setup.load_tech(), ell23, lines)
    guard_ok = abs(cog["unbound_share"] - FROZEN_COG_UNBOUND) < 0.02
    lines.append(f"  reproduction guard: cognitive unbound share "
                 f"{cog['unbound_share']:.3f} vs frozen "
                 f"{FROZEN_COG_UNBOUND} -> "
                 f"{'OK' if guard_ok else 'MISMATCH'}")
    lines.append("")

    # the robot era run
    rob = run_config(f"robot @ Pi_{VINTAGE} (A&R-calibrated)",
                     inp99, L0_99, robot_tech(A_star), ell99, lines)

    # H3: scale sensitivity of the unbound share
    shares = {}
    for m in MOMENT_BAND:
        A_m, _, _ = calibrate_A(inp99, L0_99, m)
        d_m = regime(inp99, robot_tech(A_m), rob["out"].L, R, TAU, GAMMA,
                     ell99, BETA, survival=True)
        shares[m] = d_m["unbound_mass"] / d_m["M"]
    swing = 100 * abs(shares[MOMENT_BAND[1]] - shares[MOMENT_BAND[0]])
    d_L0 = regime(inp99, robot_tech(A_star), L0_99, R, TAU, GAMMA, ell99,
                  BETA, survival=True)
    share_L0 = d_L0["unbound_mass"] / d_L0["M"]
    lines += [f"Scale sensitivity (at the primary equilibrium L): "
              f"unbound share {100 * shares[MOMENT_BAND[0]]:.1f}% at "
              f"half moment, {100 * shares[MOMENT_BAND[1]]:.1f}% at "
              f"double (swing {swing:.1f} points); at observed L0 "
              f"{100 * share_L0:.1f}%", ""]

    # H4: collinearity pre-check for script 29
    mu_pre, mu_post, _, _ = cst.post_centroids(inp99, robot_tech(A_star),
                                               L0_99, ell99)
    dmu = mu_post - mu_pre
    xi_o, chi_o = occ99["xi"].to_numpy(), occ99["chi"].to_numpy()
    g_r, g_ang = inp99.field.grad_log_pi(xi_o, chi_o)
    gx = g_r * np.cos(xi_o) - g_ang * np.sin(xi_o)
    gy = g_r * np.sin(xi_o) + g_ang * np.cos(xi_o)
    proj = dmu[:, 0] * gx + dmu[:, 1] * gy
    lnw99 = np.log(occ99["w_era"].to_numpy())
    coll, coll_p = spearmanr(proj, lnw99)
    coll_dw, _ = spearmanr(rob["diag"]["dW_bundle"], lnw99)
    lines += [f"Collinearity pre-check for script 29 (at Pi_{VINTAGE}):",
              f"  Spearman(proj_robot, ln w_{VINTAGE})      = "
              f"{coll:+.3f} (p={coll_p:.1e})",
              f"  Spearman(dW_bundle_robot, ln w_{VINTAGE}) = "
              f"{coll_dw:+.3f}",
              f"  cognitive-case collinearity in its own window was "
              f"-0.57 (identification destroyed)", ""]

    # verdicts
    H2 = rob["unbound_share"] > FROZEN_COG_UNBOUND
    H3 = swing < 5.0
    H4 = abs(coll) < 0.35
    verdicts = [
        f"  H1 (calibration monotone, hits moment)      "
        f"{'PASS' if H1 else 'FAIL'}  (A_K {A_star:.3f})",
        f"  H2 (robot unbound share > cognitive 0.68)   "
        f"{'PASS' if H2 else 'FAIL'}  ({100 * rob['unbound_share']:.0f}%)",
        f"  H3 (unbound share scale-stable, <5 points)  "
        f"{'PASS' if H3 else 'FAIL'}  (swing {swing:.1f})",
        f"  H4 (|collinearity| < 0.35, 29 identified)   "
        f"{'PASS' if H4 else 'FAIL'}  ({coll:+.3f})",
    ]
    lines.append("Pre-registered hypothesis verdicts:")
    lines += verdicts

    out = pd.DataFrame({
        "onet_code": occ99.index, "Title": occ99["Title"].to_numpy(),
        "xi": xi_o, "chi": chi_o, "L0": L0_99, "L_eq": rob["out"].L,
        "dL": rob["dL"], "dL_shock": rob["dL_shock"],
        "drift": rob["drift"], "D_o": rob["diag"]["D_o"],
        "B_o": rob["diag"]["B_o"],
        "dW_bundle": rob["diag"]["dW_bundle"], "W_o": rob["out"].W,
        "proj": proj, "ln_w_era": lnw99,
    })
    out.to_csv(RESULTS / "robot_era_equilibrium.csv", index=False)
    (RESULTS / "robot_era_equilibrium_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS / 'robot_era_equilibrium.csv'} and _summary.txt")


if __name__ == "__main__":
    main()
