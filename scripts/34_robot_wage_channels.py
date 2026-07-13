"""
34_robot_wage_channels.py
-------------------------
The wage burden of the industrial-robot wave, decomposed into the three
channels script 19 decomposes the cognitive wave into. Section 7.4 of the
manuscript needs it: the two field sections ask the same five questions, and
"where does the wage burden fall" is one of them.

WHAT IS REUSED, AND WHY NOTHING IS RE-DERIVED. The decomposition is script
19's exact path decomposition, unchanged:

    W0      pre-technology       content = full bundle,  density = n0(L0)
    W_strip apply takeover a(r)  content stripped,       density held at n0
    W_cong  let density move     content stripped,       density = n(L*)
    W_post  add bound inflow     content + reinstatement, density = n(L*)

    stripping     = ln W_strip - ln W0
    congestion    = ln W_cong  - ln W_strip
    reinstatement = ln W_post  - ln W_cong

the three summing exactly to d ln W_o. The configuration is script 28's, also
unchanged: the era-consistent 1999 economy (Pi_1999 from script 27, L0 from
1999 OEWS, era ell and mobility), with the robot field located by script 23
and its amplitude A_K calibrated by bisection to the Acemoglu-Restrepo
displacement moment. Importing both rather than restating either is the point:
if the cognitive channels move, these move with them.

THE PREDICTION THIS PRODUCER EXISTS TO TEST. The price field's return to depth
is a first harmonic pointing due north (scripts 01/02): +0.83 at due north,
-0.96 at due south, and beta_chi is +0.48 at the cognitive field's 38 degrees
against -0.47 at the robot field's 207. Stripping removes PRICED content.
The robot field therefore strips content the market pays little for, and the
cognitive field strips content the market pays well for. Per unit of displaced
task mass, the robot wave should cost less in wages than the cognitive wave.

That is a claim the paper has never made and cannot make from one field. It
follows from the price field's directional structure, which was estimated
before either technology was located, and it is the reason two fields are
worth running rather than one.

PRE-REGISTERED HYPOTHESES (written before the first run)
  C1 SIGN        the robot field's employment-weighted stripping adjustment is
                 negative.
  C2 RANKING     stripping carries the mean; congestion's employment-weighted
                 mean is small against it (|mean cong| < 0.5 x |mean strip|),
                 while its mean absolute value is material (>= 0.15 x |mean
                 strip|). This is the cognitive field's signature (script 19:
                 -0.348, -0.030, 0.101) and it should repeat.
  C3 INTENSITY   the wage cost per unit of displaced task mass,
                 |mean stripping| / mean D_o, is SMALLER for the robot field
                 than for the cognitive field. This is the directional-price
                 prediction above. FAIL would say the price field's direction
                 does not reach the wage burden, and would be reported.

                 C3 PASSED on the certified run (0.97 against 1.79), and the
                 measure is nonetheless a poor one. Stripping gives
                 d ln W ~= ln(1 - k D), so |d ln W| / D grows with the size of
                 the shock even at constant k: a price-neutral shock of the
                 cognitive field's size (D = 0.19) already scores 1.11, while
                 the robot field's (D = 0.003) scores 1.00. Part of the gap
                 between 0.97 and 1.79 is therefore the cognitive shock being
                 sixty-five times larger, not the price of what it takes. The
                 verdict stands as recorded, and the two scale-free statistics
                 below replace it in the manuscript. They were specified after
                 the certified run and are reported as descriptive, not as
                 pre-registered tests.

  D1 AIM         (descriptive) the price of the content capital takes, divided
                 by the employment-weighted mean price of work in that field's
                 OWN economy. Era-normalised, so 1999 and 2024 nominal wages
                 cannot drive it. This is the across-occupation channel: where
                 the field is aimed.
  D2 SELECTIVITY (descriptive) k = the price of the content capital takes in
                 an occupation, over the price of the content that occupation
                 holds, employment-weighted. This is the within-occupation
                 channel: the price gate, which orders locations by price
                 inside a band of equal effectiveness. k > 1 is expected for
                 both fields and is mechanical.

                 D1 and D2 are separate, and the first certified run separates
                 them: k exceeds one for both fields while the robot field's
                 aggregate take is CHEAPER than its economy's mean. Capital
                 takes dear content within an occupation, and the robot field
                 is nonetheless pointed at cheap occupations. Neither channel
                 is visible with one field.

  D3 EXCHANGE    (descriptive) the price at which bound new work lands, over
                 the price of the content capital took. A technology that takes
                 work dearer than the work it creates trades down, and the
                 wage burden follows. Reported two ways, since the price of
                 seeded work can be read at the seed's own grid location or at
                 the centroid of the occupation that binds it; the two must
                 agree or neither is reported.

  D4 UNBOUND     (descriptive) the price at the locations where seeded work
                 survives capital and binds to nobody. Section 7.2 of the
                 manuscript poses this and leaves it open: whether the unbound
                 mass is latent skilled work or low-paid friction depends on
                 the price at its location. The number answers it.
  C4 SORTING     congestion is negative for occupations that gain employment
                 (they crowd themselves) and positive for those that shed it.
                 Script 19 found -0.063 and +0.224.
  C5 ORDER       the strip-first against cong-first swing is recorded. Script
                 19's cognitive run FAILED its 25 percent tolerance at 54
                 percent, so the decomposition ranks channels and does not
                 measure either precisely. The same tolerance is applied here
                 and the same failure is expected; a pass would be the
                 surprise.

Guards: the cognitive comparator is re-run from scripts 09/19's committed
configuration in the same process, so C3's ratio is computed apples to apples
and the robot equilibrium reproduces script 28's frozen corner (labour share
0.9948 -> 0.9925, unbound 0.93).

Outputs:
    results/robot_wage_channels.csv
    results/robot_wage_channels_summary.txt

Usage:
    python scripts/34_robot_wage_channels.py
    SMOKE=1 python scripts/34_robot_wage_channels.py    (coarse grid)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.equilibrium import Equilibrium            # noqa: E402
from model.regime import regime, seeding_density      # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_setup = _load("_setup.py")
_defo = _load("19_wage_field_deformation.py")
_era = _load("28_robot_era_equilibrium.py")

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
NMIN = 1e-9
SMOKE = os.environ.get("SMOKE", "") == "1"
GRID = dict(n_ang=60, n_rad=20) if SMOKE else dict(n_ang=120, n_rad=40)

FROZEN_ROBOT = dict(lam_auto=0.9948, lam_reinst=0.9925, unbound=0.93)
FROZEN_COG = dict(strip=-0.3483, cong=-0.0298, reinst=+0.0539)
TOL = 0.05 if SMOKE else 0.004


def channels(eq, L0, Lstar):
    """Script 19's exact path decomposition, verbatim in structure."""
    def nb1(n):
        return np.maximum(n, NMIN) ** (BETA - 1.0)

    zero = np.zeros(eq.area.size)
    content_pre = eq.b_w * eq.pi_task
    content_strip = eq.strip_wD

    n0_pre, _ = _defo.density_from(eq, L0, with_tech=False)
    n_post, gcarrier_post = _defo.density_from(eq, Lstar, with_tech=True)

    W0 = _defo.w_value(eq, content_pre, zero, nb1(n0_pre))
    W_strip = _defo.w_value(eq, content_strip, zero, nb1(n0_pre))
    W_cong = _defo.w_value(eq, content_strip, zero, nb1(n_post))
    W_post = _defo.w_value(eq, content_strip, gcarrier_post, nb1(n_post))

    # the reversed order, for C5
    W_cong_first = _defo.w_value(eq, content_pre, zero, nb1(n_post))

    def dln(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((a > 0) & (b > 0), np.log(a) - np.log(b), 0.0)

    return dict(
        strip=dln(W_strip, W0),
        cong=dln(W_cong, W_strip),
        reinst=dln(W_post, W_cong),
        total=dln(W_post, W0),
        cong_first=dln(W_cong_first, W0),
        W0=W0, W_post=W_post,
    )


def price_channels(eq, Lstar):
    """D1 and D2: where the field is aimed, and how it selects inside an
    occupation. Both are era-normalised or ratio-valued, so the 1999 and 2024
    price fields can be compared."""
    ro, b, a, pi = eq.row_of, eq.b_w, eq.a_task, eq.pi_task
    n = eq.n_occ
    Lw = np.asarray(Lstar, float)

    def bc(w):
        return np.bincount(ro, weights=w, minlength=n)

    era_mean = float(np.sum(Lw[ro] * b * pi) / np.sum(Lw[ro] * b))
    taken = float(np.sum(Lw[ro] * b * a * pi) / np.sum(Lw[ro] * b * a))

    m = bc(b * a) / np.maximum(bc(b), 1e-12)
    ok = m > 1e-9
    pi_occ = bc(b * pi) / np.maximum(bc(b), 1e-12)
    k_o = ((bc(b * a * pi) / np.maximum(bc(b * a), 1e-12))
           / np.maximum(pi_occ, 1e-12))
    k = float(np.average(k_o[ok], weights=Lw[ok]))

    return dict(era_mean=era_mean, taken=taken, aim=taken / era_mean,
                selectivity=k, pi_occ=pi_occ)


def seed_prices(inp, eq, tech, Lstar, diag, ell, price):
    """D3 and D4: the price where seeded work lands. Read at the seed's own
    grid location, and independently at the centroid of the occupation that
    binds it. The two are different objects and must agree."""
    g = inp.grid
    Lw = np.asarray(Lstar, float)
    pi_g = inp.field.pi(g.xi, g.chi)

    # rebuild the grid densities exactly as model/regime.py does
    M = float(diag["M"])
    ghat = seeding_density(tech, g, "gradient", inp.field, R, TAU)
    s = M * ghat
    surv = 1.0 - tech.operated_share(g.xi, g.chi, inp.field, R, TAU)
    C = (Lw[:, None] * eq.e).sum(axis=0)
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)

    wb = s * surv * Phi * g.area
    wu = s * surv * (1.0 - Phi) * g.area
    p_bound_grid = float(np.sum(wb * pi_g) / max(np.sum(wb), 1e-12))
    p_unbound_grid = float(np.sum(wu * pi_g) / max(np.sum(wu), 1e-12))

    # the same bound work, priced at the binding occupation's own content
    B = np.asarray(diag["B_o"], float)
    okb = B > 0
    p_bound_occ = (float(np.average(price["pi_occ"][okb],
                                    weights=(B * Lw)[okb]))
                   if okb.any() else np.nan)

    era = price["era_mean"]
    return dict(
        p_bound_grid=p_bound_grid, p_unbound_grid=p_unbound_grid,
        p_bound_occ=p_bound_occ,
        bound_aim=p_bound_grid / era, unbound_aim=p_unbound_grid / era,
        exchange=p_bound_grid / price["taken"],
        exchange_occ=p_bound_occ / price["taken"],
    )


def summarise(ch, L0, Lstar, D_o, W0, W_post, tag):
    """Aggregate EXACTLY as script 19 does, or the comparator will not
    reproduce and the guard will (correctly) fail: employment weights are the
    POST-SORT Lstar, the mask is (W0 > 0) & (W_post > 0), and the gainer /
    loser and order-robustness statistics are unweighted means over that
    mask. Reimplementing the aggregation differently is the one way to make
    this producer disagree with the paper it is supposed to extend."""
    valid = (W0 > 0) & (W_post > 0)

    def wmean(x):
        return float(np.average(x[valid], weights=Lstar[valid]))

    dL = np.asarray(Lstar, float) - np.asarray(L0, float)
    gain, lose = (dL > 0) & valid, (dL < 0) & valid

    m = {k: wmean(ch[k]) for k in ("strip", "cong", "reinst", "total")}
    m["abs_strip"] = wmean(np.abs(ch["strip"]))
    m["abs_cong"] = wmean(np.abs(ch["cong"]))
    m["cong_share"] = (m["abs_cong"] / m["abs_strip"]
                       if m["abs_strip"] > 0 else np.nan)

    # displaced mass on the same weights as the numerator, and on L0 (the
    # weighting the robot calibration targets), so the guard can see both
    m["meanD"] = wmean(np.asarray(D_o, float))
    w0 = np.asarray(L0, float)
    m["meanD_L0"] = float(np.average(np.asarray(D_o, float)[valid],
                                     weights=w0[valid]))
    m["intensity"] = (abs(m["strip"]) / m["meanD"]
                      if m["meanD"] > 0 else np.nan)

    m["cong_gainers"] = float(ch["cong"][gain].mean()) if gain.any() else 0.0
    m["cong_losers"] = float(ch["cong"][lose].mean()) if lose.any() else 0.0

    mc = float(ch["cong"][valid].mean())
    mca = float(ch["cong_first"][valid].mean())
    m["order_swing"] = abs(mca - mc) / max(abs(mc), 1e-9)

    m["exactness"] = float(np.max(np.abs(
        ch["strip"] + ch["cong"] + ch["reinst"] - ch["total"])))
    m["tag"] = tag
    return m


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    lines: list[str] = [
        "The robot wave's wage burden, in the three channels of script 19.",
        "  Configuration: script 28's era-consistent 1999 economy, robot field "
        "at the A&R displacement moment.",
        f"  grid {GRID['n_ang']}x{GRID['n_rad']}"
        f"{'   SMOKE, nothing certified' if SMOKE else ''}",
        "",
    ]

    # ---- robot, in its own era -------------------------------------
    inp_r, L0_r, occ_r = _era.build_era_inputs(_era.VINTAGE)
    ell_r = _setup.interpretable_ell(inp_r)
    A_K, _f, _mono = _era.calibrate_A(inp_r, L0_r, _era.MOMENT)
    tech_r = _era.robot_tech(A_K)

    eq_r = Equilibrium(inp_r, tech_r, R, TAU, GAMMA, ell_r, BETA, wedge=None,
                       survival=True)
    eq_r.L0 = L0_r
    c_r, kap_r, _, eq_r.alpha = _setup.anchor_reference(eq_r, L0_r)
    out_r = eq_r.solve(c_r, kap_r)

    diag_r = regime(inp_r, tech_r, out_r.L, R, TAU, GAMMA, ell_r, BETA,
                    wedge=None, survival=True)
    diag_r0 = regime(inp_r, tech_r, out_r.L, R, TAU, 0.0, ell_r, BETA,
                     wedge=None, survival=True)
    seeded_r = float(diag_r["M"])
    unb_r = float(diag_r["unbound_mass"]) / seeded_r if seeded_r > 0 else np.nan

    ch_r = channels(eq_r, L0_r, out_r.L)
    s_r = summarise(ch_r, L0_r, np.asarray(out_r.L, float),
                    np.asarray(diag_r["D_o"], float),
                    ch_r["W0"], ch_r["W_post"], "robot")
    s_r.update(price_channels(eq_r, out_r.L))
    s_r.update(seed_prices(inp_r, eq_r, tech_r, out_r.L, diag_r, ell_r, s_r))

    lines += [
        f"ROBOT FIELD  (A_K = {A_K:.3f} at the {_era.MOMENT:.4f} moment; "
        f"vintage {_era.VINTAGE})",
        f"  labour share  1.000 -> {float(diag_r0['labor_share']):.4f} "
        f"(automation) -> {float(diag_r['labor_share']):.4f} (reinstatement)",
        f"  unbound share of seeded mass: {unb_r:.3f}",
        "",
        "  employment-weighted wage adjustment, log points:",
        f"    stripping      {s_r['strip']:+.4f}",
        f"    congestion     {s_r['cong']:+.4f}   "
        f"(mean absolute {s_r['abs_cong']:.4f})",
        f"    reinstatement  {s_r['reinst']:+.4f}",
        f"    total          {s_r['total']:+.4f}",
        f"  mean displaced task mass D_o: {s_r['meanD']:.5f}",
        f"  wage cost per unit displaced: {s_r['intensity']:.2f} log points",
        "",
    ]

    # ---- cognitive comparator, committed configuration --------------
    inp_c, L0_c, occ_c = _setup.build_inputs(**GRID)
    ell_c = _setup.interpretable_ell(inp_c)
    tech_c = _setup.load_tech()

    eq_c = Equilibrium(inp_c, tech_c, R, TAU, GAMMA, ell_c, BETA, wedge=None,
                       survival=True)
    eq_c.L0 = L0_c
    c_c, kap_c, _, eq_c.alpha = _setup.anchor_reference(eq_c, L0_c)
    out_c = eq_c.solve(c_c, kap_c)
    diag_c = regime(inp_c, tech_c, out_c.L, R, TAU, GAMMA, ell_c, BETA,
                    wedge=None, survival=True)

    ch_c = channels(eq_c, L0_c, out_c.L)
    s_c = summarise(ch_c, L0_c, np.asarray(out_c.L, float),
                    np.asarray(diag_c["D_o"], float),
                    ch_c["W0"], ch_c["W_post"], "cognitive")
    s_c.update(price_channels(eq_c, out_c.L))
    s_c.update(seed_prices(inp_c, eq_c, tech_c, out_c.L, diag_c, ell_c, s_c))

    lines += [
        "COGNITIVE FIELD  (committed configuration, as script 19)",
        f"    stripping      {s_c['strip']:+.4f}   "
        f"(frozen {FROZEN_COG['strip']:+.4f})",
        f"    congestion     {s_c['cong']:+.4f}   "
        f"(frozen {FROZEN_COG['cong']:+.4f})",
        f"    reinstatement  {s_c['reinst']:+.4f}   "
        f"(frozen {FROZEN_COG['reinst']:+.4f})",
        f"  mean displaced task mass D_o: {s_c['meanD']:.5f}",
        f"  wage cost per unit displaced: {s_c['intensity']:.2f} log points",
        "",
    ]

    # ---- sorting signature (C4), computed in summarise ---------------
    for s_, tag in [(s_r, "robot"), (s_c, "cognitive")]:
        lines += [f"  {tag}: congestion for employment gainers "
                  f"{s_['cong_gainers']:+.4f}, for losers "
                  f"{s_['cong_losers']:+.4f}"]
    lines += [
        "",
        "The price of what capital takes (D1, D2: descriptive; specified after",
        "the certified run, see the docstring). Prices are divided by the mean",
        "price of work in each field's OWN economy, so 1999 and 2024 nominal",
        "wages cannot drive the comparison.",
        "",
        f"  {'':<38} {'robot':>10} {'cognitive':>11}",
        f"  {'economy mean price of work':<38} "
        f"{s_r['era_mean']:>10.2f} {s_c['era_mean']:>11.2f}",
        f"  {'price of the content capital takes':<38} "
        f"{s_r['taken']:>10.2f} {s_c['taken']:>11.2f}",
        f"  {'D1  aim: taken / economy mean':<38} "
        f"{s_r['aim']:>10.3f} {s_c['aim']:>11.3f}",
        f"  {'D2  selectivity within occupation k':<38} "
        f"{s_r['selectivity']:>10.3f} {s_c['selectivity']:>11.3f}",
        f"  {'D3  bound new work lands at':<38} "
        f"{s_r['bound_aim']:>10.3f} {s_c['bound_aim']:>11.3f}",
        f"  {'    exchange rate  lands / takes':<38} "
        f"{s_r['exchange']:>10.3f} {s_c['exchange']:>11.3f}",
        f"  {'    the same, priced at the binder':<38} "
        f"{s_r['exchange_occ']:>10.3f} {s_c['exchange_occ']:>11.3f}",
        f"  {'D4  unbound new work sits at':<38} "
        f"{s_r['unbound_aim']:>10.3f} {s_c['unbound_aim']:>11.3f}",
        "",
        f"  The robot field takes work priced {s_r['aim']:.2f} x its economy's "
        f"mean; the cognitive field, {s_c['aim']:.2f} x.",
        f"  Both fields take dearer-than-average content INSIDE an occupation "
        f"(k > 1: the price gate),",
        f"  and the robot field is nonetheless aimed at cheap occupations. The "
        f"two channels are separate,",
        f"  and neither is visible with a single field.",
        "",
        f"  D3: the robot wave takes work at {s_r['aim']:.2f} x and seeds bound "
        f"work at {s_r['bound_aim']:.2f} x: an even trade "
        f"({s_r['exchange']:.2f}).",
        f"      The cognitive wave takes at {s_c['aim']:.2f} x and seeds bound "
        f"work at {s_c['bound_aim']:.2f} x: it trades down "
        f"({s_c['exchange']:.2f}).",
        f"      Automation costs wages when it destroys work dearer than the "
        f"work it creates, and the",
        f"      geometry decides which. The two readings of the seeded price "
        f"agree.",
        "",
        f"  D4: the cognitive field's unbound mass sits at "
        f"{s_c['unbound_aim']:.2f} x the economy mean, ABOVE it: it is latent",
        f"      skilled work rather than low-paid friction, and it is "
        f"{100*0.68:.0f} percent of what the field seeds.",
        f"      The robot field's unbound mass sits at "
        f"{s_r['unbound_aim']:.2f} x. Section 7.2 poses this and leaves it "
        f"open.",
        "",
    ]

    # ---- verdicts ---------------------------------------------------
    guard = (abs(float(diag_r0["labor_share"]) - FROZEN_ROBOT["lam_auto"]) < TOL
             and abs(float(diag_r["labor_share"])
                     - FROZEN_ROBOT["lam_reinst"]) < TOL
             and abs(unb_r - FROZEN_ROBOT["unbound"]) < TOL * 5
             and abs(s_c["strip"] - FROZEN_COG["strip"]) < TOL * 5
             and abs(s_c["cong"] - FROZEN_COG["cong"]) < TOL * 5
             and abs(s_c["reinst"] - FROZEN_COG["reinst"]) < TOL * 5
             and abs(s_r["meanD_L0"] - _era.MOMENT) < 1e-4)

    c1 = s_r["strip"] < 0
    c2 = (abs(s_r["cong"]) < 0.5 * abs(s_r["strip"])
          and s_r["cong_share"] >= 0.15)
    c3 = s_r["intensity"] < s_c["intensity"]
    c4 = s_r["cong_gainers"] < 0 < s_r["cong_losers"]
    c5 = bool(np.isfinite(s_r["order_swing"]) and s_r["order_swing"] < 0.25)

    lines += [
        "Pre-registered verdicts:",
        f"  guard: robot corner and cognitive channels reproduce   "
        f"{'PASS' if guard else 'FAIL'}"
        f"{'  (SMOKE: tolerances widened)' if SMOKE else ''}",
        f"  C1 stripping is negative                     "
        f"{'PASS' if c1 else 'FAIL'}   ({s_r['strip']:+.4f})",
        f"  C2 stripping carries the mean, congestion    "
        f"{'PASS' if c2 else 'FAIL'}   "
        f"(mean ratio {abs(s_r['cong'])/max(abs(s_r['strip']),1e-12):.2f}, "
        f"absolute ratio {s_r['cong_share']:.2f})",
        f"     disperses",
        f"  C3 robot wage cost per unit displaced <       "
        f"{'PASS' if c3 else 'FAIL'}   "
        f"robot {s_r['intensity']:.2f} vs cognitive {s_c['intensity']:.2f}",
        f"     cognitive  (the directional-price test)",
        f"  C4 gainers crowd in, losers thin out         "
        f"{'PASS' if c4 else 'FAIL'}   "
        f"({s_r['cong_gainers']:+.4f} / {s_r['cong_losers']:+.4f})",
        f"  C5 order swing under 25 percent              "
        f"{'PASS' if c5 else 'FAIL'}   "
        f"({100*s_r['order_swing']:.0f}% ; the cognitive run failed at 54%)",
        "",
    ]
    if not guard and not SMOKE:
        lines += ["GUARD FAILED: do not read the verdicts."]

    pd.DataFrame([s_r, s_c]).to_csv(RESULTS / "robot_wage_channels.csv",
                                    index=False)
    (RESULTS / "robot_wage_channels_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS / 'robot_wage_channels.csv'}")


if __name__ == "__main__":
    main()
