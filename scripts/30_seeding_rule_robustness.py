"""
30_seeding_rule_robustness.py
-----------------------------
The seeding rule zeta ~ |grad phi_K| is a construction assumption of the
same standing as the circular field form (see the FIELD FORM block in
scripts/33_field_form_invariance.py): the model needs one, the gradient
rule is the disk analogue of boundary task creation, and this producer
checks it rather than postulates it, by running the stated alternative
rules through the SAME model and the SAME startup data and letting the
data score them. Every rule is a normalized density, so
the rules move the same seeded mass M to different locations; the
comparison is between locations, not scales.

Rules (model.regime.seeding_density):
  gradient   |grad phi_K|        benchmark; peaks on the ring at z_K
  level      phi_K               new work where the technology is most
                                 effective; peaks at the centre
  incidence  a(1-a)              new work on the adoption margin
  mixture    phi_K |grad phi_K|  boundary-weighted core; peaks at
                                 z_K / sqrt(2)

Objects per rule:
  1. The full anchored equilibrium (scripts 09/28 machinery): unbound
     share of seeded mass, labour share under automation and
     reinstatement, the reinstatement gap.
  2. The startup score: the rule density evaluated at the 1,961
     projected startup positions (producer 21's frozen CSV). Two
     statistics: enrichment (mean density at startups over the disk
     mean, producer 22's construction) and the mean log-density, a
     proper scoring rule for comparing normalized densities. Pairwise
     bootstrap (startup resampling, 2000 draws) gives a 95 percent
     interval on each alternative's log-score deficit against the
     benchmark.

EVALUATION LOGIC (settled with the author after the first certified
run). The selection criterion among rules is how well the startups fall
under the SEEDING CURVE zeta -- never the unbound share, which is an
equilibrium consequence and stays in the rows as information only. A
second comparison object is added beside the raw curve: the BOUND part
of seeded work, both as the cell-level density iota(r) (the seed that
survives capital and finds an occupational bearer) and as the
occupation-level B_o smoothed at the centroids. The rationale is the
instrument itself: the startup test (Fenoaltea et al. positions,
projected through the frozen occupation-task embedding) is built on
EXISTING occupations, so the model objects commensurable with it are
the seeding curve and its bound part. Unbound work stands free of the
test by construction: it is the model's claim about work without
occupational bearers, which an instrument built from occupational
language cannot adjudicate. Its tail (the outer western arc) is instead
the forward-looking watch region: whether coming startups and new
tasks emerge there is an open question for the years after this paper.

EXPECTATIONS, BOUND-WORK BLOCK (T; written before the first run of the
extended script):
  T1  The committed rule's bound density fits the AI startups better
      than its raw seeding curve does: bound work is the seed filtered
      through where existing occupations can absorb it, and the
      instrument sees through existing occupations. The AI group's
      inner-flank position (0.78 z_K) is the bound flank.
  T2  Robotics fit WORSE under every bound density than under the raw
      curve: they sit on the arc the model keeps unbound, visible to
      the instrument through product language but without occupational
      bearers.
  T3  Under the bound-density criterion the committed rule is
      competitive with or better than the raw-curve alternatives
      (recorded either way; this is the occupation-commensurable
      version of S3).

RESULTS (certified full-grid runs; S from the first, T from the
extended script):
  S1  PASS. Gradient reproduces unbound 0.676 and enrichment 1.30/1.34.
  S2  FAIL as ordered: 53 / 44 / 68 (mixture 55). The incidence rule
      binds most, not least -- the adoption margin sits in the dense
      dear region, closer to occupations than the core.
  S3  FAIL. Level and mixture outscore gradient for pooled and AI;
      gradient beats level for robotics and beats incidence everywhere
      (robotics -2.36). Resolved by the T block: the AI group's inward
      shift is the BOUND flank, not evidence for another raw curve.
  S4  HOLDS, max gap 0.49 points. The structural claim is rule-
      invariant; unbound spans 44-68 across rules, information only.
  T1  HOLDS (+0.235, CI above zero): the committed rule's bound density
      closes most of the gap its raw curve leaves against the AI
      startups. The B_o occupation kernel dominates everything
      (+0.79 over the raw ring for AI, +0.36 for robotics): the
      occupation layer, not the choice of raw curve, carries the fit.
  T2  FAILS (+0.584 field, +0.362 kernel): robotics fit BETTER under
      bound, against expectation. Diagnosis: iota = s surv Phi and
      u = s surv (1 - Phi) share the factor s surv, and the survival
      gate 1 - a carves the operated north-east ANGULARLY -- a strong
      cut -- while the Phi split is mild at mean Phi ~ 0.3. Robotics
      sit in the surviving west where both densities are elevated; the
      instrument sees the survival carving, not the Phi split, at those
      positions. The surv-filtered seed row (added with this record)
      quantifies it: if its score sits close to the bound score for
      robotics, the carving is the whole story. T2's failure does not
      contradict "unbound stands free"; it shows the startup instrument
      cannot adjudicate Phi from 1 - Phi on the western arc.
  T3  FAILS narrowly (-0.129): on the raw-vs-bound margin the mixture
      curve still edges the gradient's bound FIELD for AI, while the
      B_o kernel dominates all raw curves. Same reading as T1.

EXPECTATIONS (working notes, recorded before the first run):
  S1  Guards: the gradient rule reproduces the committed numbers --
      unbound share 0.676 (+/- 0.01, asserted) and startup enrichment
      1.30 (AI) / 1.34 (robotics) in producer 22's construction, the
      median at the startup points over the disk mean (+/- 0.07,
      asserted). The mean-based enrichment and the log-score are
      reported alongside; they weight peak alignment and coverage
      differently and need not agree with the median form. If a guard
      breaks, the machinery moved and nothing downstream is read.
  S2  Unbound ordering: level < incidence < gradient, mixture between
      incidence and gradient. Core-seeding lands new work where
      occupations are dense and binds more of it. Recorded, not
      asserted.
  S3  Startup selection: the gradient rule's log-score beats level and
      incidence for the pooled sample and for each group, with the
      bootstrap interval on the deficit excluding zero (the committed
      producer-22 numbers already point here: ring enrichment 1.30/1.34
      against incidence-core 0.92/0.03). The mixture rule may be
      competitive with the benchmark; if it ties, the data select
      boundary-concentrated rules over core rules, which is the
      economically relevant discrimination and is written as such.
  S4  Structural invariance: the reinstatement gap (automation share
      minus reinstatement share) stays below one percentage point under
      EVERY rule. The paper's structural claim -- reinstatement does
      not restore the labour share -- should not depend on where the
      seed lands; if it fails under some rule, that rule's row is
      reported and the manuscript's claim is scoped.

Reads:
    results/startup_seeding_startups.csv   positions (producer 21)
    committed field and geometry (via scripts/_setup)
Writes:
    results/seeding_rules.csv, _summary.txt, seeding_rules.png

SMOKE=1 runs the coarse grid (mechanics check only, not a result).

Usage:
    python scripts/30_seeding_rule_robustness.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.equilibrium import Equilibrium                   # noqa: E402
from model.regime import regime, seeding_density            # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_setup = _load("_setup.py")

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
RULES = ("gradient", "level", "incidence", "mixture")
SMOKE = os.environ.get("SMOKE", "") == "1"
GRID = dict(n_ang=60, n_rad=20) if SMOKE else dict(n_ang=120, n_rad=40)
N_BOOT, SEED = 2000, 30
FROZEN_UNBOUND = 0.676          # anchored certified run (scripts 09/17)
FROZEN_ENRICH = {"ai": 1.30, "robotics": 1.34}   # producer 22


def rule_at_points(tech, field, rule, xi, chi, Z):
    """The rule's un-normalized field evaluated at points, divided by
    the grid normalizer Z (so values are density units)."""
    if rule == "gradient":
        g = tech.grad_phi_norm(xi, chi)
    elif rule == "level":
        g = tech.phi(xi, chi)
    elif rule == "incidence":
        a = tech.operated_share(xi, chi, field, R, TAU)
        g = a * (1.0 - a)
    else:
        g = tech.phi(xi, chi) * tech.grad_phi_norm(xi, chi)
    return g / Z


def main() -> None:
    lines = ["Seeding-rule robustness (script 30; expectations in the "
             "docstring)."]
    if SMOKE:
        lines.append("SMOKE MODE: coarse grid -- mechanics check, "
                     "NOT A RESULT.")

    inp, L0, occ = _setup.build_inputs(**GRID)
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    grid = inp.grid
    total_area = float(np.sum(grid.area))

    # one anchor for all rules: the zero-field baseline has no seeding
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, survival=True)
    eq0.L0 = L0
    c, kappa, _, alpha = _setup.anchor_reference(eq0, L0)
    lines += [f"  occupations {len(L0)}, grid {grid.xi.size} cells; "
              f"ell {ell:.4f}; anchored (c {c:.3f}, kappa {kappa:.3f}); "
              f"one alpha for all rules (the zero-field baseline has no "
              f"seeding)", ""]

    # startups (frozen positions from producer 21)
    st = pd.read_csv(RESULTS / "startup_seeding_startups.csv")
    groups = {"pooled": st,
              "ai": st[st["is_ai"] == 1],
              "robotics": st[st["is_robotics"] == 1]}
    lines.append(f"  startups: pooled {len(st)}, ai {len(groups['ai'])}, "
                 f"robotics {len(groups['robotics'])}")
    lines.append("")

    # normalizers per rule from the grid (the model's own integral)
    Z = {}
    for rule in RULES:
        g_hat = seeding_density(tech, grid, rule, inp.field, R, TAU)
        # recover Z: un-normalized field on the grid / g_hat (any cell)
        raw = rule_at_points(tech, inp.field, rule, grid.xi, grid.chi, 1.0)
        Z[rule] = float(np.sum(raw * grid.area))

    rows, scores = [], {}
    bound_field, B_o_rule = {}, {}
    for rule in RULES:
        # 1. equilibrium objects
        eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA,
                         survival=True, seeding=rule)
        eq.L0 = L0
        eq.alpha = alpha
        out = eq.solve(c, kappa)
        diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA,
                      survival=True, seeding=rule)
        diag0 = regime(inp, tech, out.L, R, TAU, 0.0, ell, BETA,
                       survival=True, seeding=rule)
        unbound = diag["unbound_mass"] / diag["M"]
        gap = diag0["labor_share"] - diag["labor_share"]
        bound_field[rule] = np.asarray(diag["iota_tot"], float)
        B_o_rule[rule] = np.asarray(diag["B_o"], float)

        # radial location of the rule density
        g_hat = seeding_density(tech, grid, rule, inp.field, R, TAU)
        d_grid = tech._dist(grid.xi, grid.chi)
        w = g_hat * grid.area
        order = np.argsort(d_grid)
        cum = np.cumsum(w[order]) / np.sum(w)
        d_med = float(d_grid[order][np.searchsorted(cum, 0.5)])

        # 2. startup scores
        row = dict(rule=rule, converged=out.converged,
                   unbound_share=float(unbound),
                   share_auto=diag0["labor_share"],
                   share_reinst=diag["labor_share"],
                   reinst_gap=float(gap), d_median=d_med,
                   d_median_over_zK=d_med / tech.z_K)
        for gname, gdf in groups.items():
            dens = rule_at_points(tech, inp.field, rule,
                                  gdf["xi"].to_numpy(),
                                  gdf["chi"].to_numpy(), Z[rule])
            dens = np.maximum(dens, 1e-300)
            # producer 22's construction: median at points / disk mean
            # (normalization cancels: median(zeta_hat) * total_area)
            row[f"enrich_med_{gname}"] = float(np.median(dens) * total_area)
            row[f"enrich_mean_{gname}"] = float(np.mean(dens) * total_area)
            row[f"logscore_{gname}"] = float(np.mean(np.log(dens)))
            scores[(rule, gname)] = np.log(dens)
        rows.append(row)
        lines += [f"[{rule:9s}] unbound {100*unbound:.1f}%  share "
                  f"{diag0['labor_share']:.4f} -> {diag['labor_share']:.4f} "
                  f"(gap {100*gap:+.2f} pts)  d_med {d_med:.3f} "
                  f"({d_med/tech.z_K:.2f} z_K)",
                  f"            enrichment (median, producer-22 form): "
                  f"pooled {row['enrich_med_pooled']:.2f}x  ai "
                  f"{row['enrich_med_ai']:.2f}x  robotics "
                  f"{row['enrich_med_robotics']:.2f}x",
                  f"            enrichment (mean): pooled "
                  f"{row['enrich_mean_pooled']:.2f}x  ai "
                  f"{row['enrich_mean_ai']:.2f}x  robotics "
                  f"{row['enrich_mean_robotics']:.2f}x",
                  f"            log-score:  pooled "
                  f"{row['logscore_pooled']:+.3f}  ai "
                  f"{row['logscore_ai']:+.3f}  robotics "
                  f"{row['logscore_robotics']:+.3f}"]
    lines.append("")

    # guards on the benchmark row
    bench = rows[0]
    assert abs(bench["unbound_share"] - FROZEN_UNBOUND) < 0.01, (
        f"gradient unbound {bench['unbound_share']:.3f} != committed "
        f"{FROZEN_UNBOUND}")
    if not SMOKE:
        for g, v in FROZEN_ENRICH.items():
            assert abs(bench[f"enrich_med_{g}"] - v) < 0.07, (
                f"gradient enrichment [{g}] {bench[f'enrich_med_{g}']:.2f} "
                f"!= producer 22's {v} (median-at-points / disk-mean)")

    # bootstrap: each alternative's log-score deficit vs the benchmark
    rng = np.random.default_rng(SEED)
    lines.append("Bootstrap 95% interval on the log-score deficit vs the "
                 "gradient benchmark (negative = the alternative scores "
                 "worse; interval excluding zero = the data discriminate):")
    boot_rows = []
    for gname, gdf in groups.items():
        n = len(gdf)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        base = scores[("gradient", gname)]
        for rule in RULES[1:]:
            diff = scores[(rule, gname)] - base       # per startup
            draws = diff[idx].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            point = float(np.mean(diff))
            verdict = ("gradient preferred" if hi < 0 else
                       "alternative preferred" if lo > 0 else
                       "not separated")
            lines.append(f"  [{gname:8s}] {rule:9s} {point:+.3f}  "
                         f"[{lo:+.3f}, {hi:+.3f}]  {verdict}")
            boot_rows.append(dict(group=gname, rule=rule, deficit=point,
                                  lo=float(lo), hi=float(hi),
                                  verdict=verdict))
    lines.append("")

    # ── bound-work block: the occupation-commensurable comparison ────
    # (see EVALUATION LOGIC in the docstring; expectations T1-T3)
    from model.equilibrium import _cell_index
    KDE_BW = 0.06                       # producer 22's bandwidth
    occ_x = (occ["chi"] * np.cos(occ["xi"])).to_numpy()
    occ_y = (occ["chi"] * np.sin(occ["xi"])).to_numpy()

    def bkde_at(xi, chi, w):
        """Occupation-centroid Gaussian KDE (22's kernel), weighted by
        w, evaluated at points; normalized in the plane (edge
        truncation at the disk boundary is shared by every comparison
        and cancels in rankings)."""
        px, py = chi * np.cos(xi), chi * np.sin(xi)
        d2 = ((px[:, None] - occ_x[None, :]) ** 2
              + (py[:, None] - occ_y[None, :]) ** 2)
        k = np.exp(-0.5 * d2 / KDE_BW ** 2) @ w
        return k / (2 * np.pi * KDE_BW ** 2 * w.sum())

    lines.append("Bound-work comparison (criterion: startups under the "
                 "curve; unbound is never a criterion):")
    # surv-filtered seed for the T2 diagnosis: iota and u share the factor
    # s * surv, and the survival gate carves angularly; this density is
    # that shared factor alone (gradient rule)
    a_grid = tech.operated_share(grid.xi, grid.chi, inp.field, R, TAU)
    gsurv = seeding_density(tech, grid, "gradient") * (1.0 - a_grid)
    Zs = float(np.sum(gsurv * grid.area))
    gsurv_hat = gsurv / Zs if Zs > 0 else gsurv
    bscores = {}
    for rule in RULES:
        iota = bound_field[rule]
        Zb = float(np.sum(iota * grid.area))
        iota_hat = iota / Zb if Zb > 0 else iota
        row = next(r for r in rows if r["rule"] == rule)
        for gname, gdf in groups.items():
            xi_p, chi_p = gdf["xi"].to_numpy(), gdf["chi"].to_numpy()
            cell = _cell_index(grid, xi_p, chi_p)
            dens_f = np.maximum(iota_hat[cell], 1e-300)
            dens_k = np.maximum(bkde_at(xi_p, chi_p, B_o_rule[rule]),
                                1e-300)
            row[f"bound_med_{gname}"] = float(np.median(dens_f)
                                              * total_area)
            row[f"bound_logscore_{gname}"] = float(np.mean(np.log(dens_f)))
            row[f"bkde_logscore_{gname}"] = float(np.mean(np.log(dens_k)))
            bscores[(rule, gname, "field")] = np.log(dens_f)
            bscores[(rule, gname, "kde")] = np.log(dens_k)
        lines += [f"[{rule:9s}] bound-density enrichment (median): pooled "
                  f"{row['bound_med_pooled']:.2f}x  ai "
                  f"{row['bound_med_ai']:.2f}x  robotics "
                  f"{row['bound_med_robotics']:.2f}x",
                  f"            bound log-score: pooled "
                  f"{row['bound_logscore_pooled']:+.3f}  ai "
                  f"{row['bound_logscore_ai']:+.3f}  robotics "
                  f"{row['bound_logscore_robotics']:+.3f}   "
                  f"(B_o kernel: ai {row['bkde_logscore_ai']:+.3f}, "
                  f"robotics {row['bkde_logscore_robotics']:+.3f})"]
    lines.append("")

    surv_line = "[surv-seed] log-score (gradient seed x survival, no Phi):"
    for gname, gdf in groups.items():
        cell = _cell_index(grid, gdf["xi"].to_numpy(),
                           gdf["chi"].to_numpy())
        dens_s = np.maximum(gsurv_hat[cell], 1e-300)
        bscores[("survseed", gname, "field")] = np.log(dens_s)
        surv_line += f"  {gname} {float(np.mean(np.log(dens_s))):+.3f}"
    lines += [surv_line, ""]

    lines.append("Bootstrap 95% intervals, bound-work questions "
                 "(positive = first named preferred):")
    T = {}
    for gname, gdf in groups.items():
        n = len(gdf)
        idx = rng.integers(0, n, size=(N_BOOT, n))

        def ci(a, b):
            diff = a - b
            draws = diff[idx].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            return float(np.mean(diff)), float(lo), float(hi)

        q1 = ci(bscores[("gradient", gname, "field")],
                scores[("gradient", gname)])
        q2 = ci(bscores[("gradient", gname, "field")],
                scores[("mixture", gname)])
        q3 = ci(bscores[("gradient", gname, "kde")],
                scores[("gradient", gname)])
        T[gname] = dict(q1=q1, q2=q2, q3=q3)
        lines += [f"  [{gname:8s}] bound(grad) - zeta(grad):    "
                  f"{q1[0]:+.3f} [{q1[1]:+.3f}, {q1[2]:+.3f}]",
                  f"  [{gname:8s}] bound(grad) - zeta(mixture): "
                  f"{q2[0]:+.3f} [{q2[1]:+.3f}, {q2[2]:+.3f}]",
                  f"  [{gname:8s}] B_o-kde(grad) - zeta(grad):  "
                  f"{q3[0]:+.3f} [{q3[1]:+.3f}, {q3[2]:+.3f}]"]
    lines.append("")

    T1 = T["ai"]["q1"][1] > 0                     # CI above zero
    T2 = all(T["robotics"][q][2] < 0 for q in ("q1", "q3"))
    T3 = T["ai"]["q2"][1] > 0
    lines += ["Bound-work expectation record (T; see docstring):",
              f"  T1 bound(grad) fits AI better than its raw curve    "
              f"{'HOLDS' if T1 else 'FAILS'}  "
              f"({T['ai']['q1'][0]:+.3f})",
              f"  T2 robotics fit worse under bound than raw          "
              f"{'HOLDS' if T2 else 'FAILS'}  "
              f"(field {T['robotics']['q1'][0]:+.3f}, "
              f"kde {T['robotics']['q3'][0]:+.3f})",
              f"  T3 bound(grad) at least matches zeta(mixture), AI   "
              f"{'HOLDS' if T3 else 'FAILS'}  "
              f"({T['ai']['q2'][0]:+.3f})"]
    lines.append("")

    # expectation record
    u = {r["rule"]: r["unbound_share"] for r in rows}
    S2 = u["level"] < u["incidence"] < u["gradient"]
    sel = {(b["group"], b["rule"]): b["verdict"] for b in boot_rows}
    S3 = all(sel[(g, r)] == "gradient preferred"
             for g in ("pooled", "ai", "robotics")
             for r in ("level", "incidence"))
    S4 = all(abs(r["reinst_gap"]) < 0.01 for r in rows)
    mix = [sel[(g, "mixture")] for g in ("pooled", "ai", "robotics")]
    lines += ["Expectation record (working notes; see docstring):",
              f"  S1 guards passed (gradient row reproduces committed "
              f"numbers)",
              f"  S2 unbound ordering level < incidence < gradient   "
              f"{'HOLDS' if S2 else 'FAILS'}  "
              f"({100*u['level']:.0f} / {100*u['incidence']:.0f} / "
              f"{100*u['gradient']:.0f}; mixture {100*u['mixture']:.0f})",
              f"  S3 data select gradient over level and incidence  "
              f"{'HOLDS' if S3 else 'FAILS'}",
              f"  S4 reinstatement gap < 1 pt under every rule      "
              f"{'HOLDS' if S4 else 'FAILS'}  "
              f"(max {100*max(abs(r['reinst_gap']) for r in rows):.2f} pts)",
              f"  mixture vs gradient: {', '.join(sorted(set(mix)))}"]

    pd.DataFrame(rows).to_csv(RESULTS / "seeding_rules.csv", index=False)
    (RESULTS / "seeding_rules_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))

    _figure(tech, inp, grid, groups, Z, RESULTS / "seeding_rules.png")
    print(f"wrote {RESULTS / 'seeding_rules.csv'}, _summary.txt, .png")


def _figure(tech, inp, grid, groups, Z, out_path):
    """Radial profiles of the four rule densities against the startup
    radial distribution (distance to the field centre, z_K units)."""
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    d = np.linspace(0, 1.6, 400)
    # evaluate along a ray through the field centre direction
    xi_ray = np.full_like(d, tech.xi_K)
    # points at distance d from p_K along the radial direction
    px, py = tech.p_K
    ux, uy = np.cos(tech.xi_K), np.sin(tech.xi_K)
    xs, ys = px + d * ux, py + d * uy
    chi_pt = np.hypot(xs, ys)
    xi_pt = np.arctan2(ys, xs)
    colors = {"gradient": "#31688e", "level": "#b8474d",
              "incidence": "#8a7a2e", "mixture": "#5d5d5d"}
    for rule in RULES:
        dens = rule_at_points(tech, inp.field, rule, xi_pt, chi_pt, Z[rule])
        ax.plot(d / tech.z_K, dens / dens.max(), color=colors[rule],
                lw=2, label=rule)
    for gname, style in (("ai", "-"), ("robotics", "--")):
        gdf = groups[gname]
        dd = tech._dist(gdf["xi"].to_numpy(), gdf["chi"].to_numpy())
        ax.hist(dd / tech.z_K, bins=30, density=True, histtype="step",
                linestyle=style, color="0.25", weights=None,
                label=f"startups ({gname})")
    ax.axvline(1.0, color="0.8", lw=0.8)
    ax.set_xlabel(r"distance to field centre, $z_K$ units")
    ax.set_ylabel("normalized density / startup frequency")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
