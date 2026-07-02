"""
14_sensitivity.py
-----------------
Sensitivity of the headline quantities to the economy parameters
(paper: parameter table in the calibration section, sweep report in the
cognitive case).

Baseline: R 18, tau 0.08, beta 0.5, gamma 0.5, ell from the SD rule
(scripts/_setup.interpretable_ell), attachment shape rho 0.5, lam_over 1,
eta 1, survival gate on, no wedge. Mobility (c, kappa) from the reference
rule per point (kappa = SD of baseline value, median move costs one kappa),
matching scripts/09.

Sweeps, one block per free parameter group:
  A  attachment scale and seeding: ell x {1/4, 1/2, 1, 2, 4}, gamma
     {0.25, 0.5, 0.75}. Targets the unbound share and the reinstatement
     gap in the labour share.
  B  adoption gate: R {12, 18, 24} x tau {0.04, 0.08, 0.16}. Targets the
     operated mass and the automation drop in the labour share.
  C  attachment shape: rho {0.25, 1.0}, lam_over {0} around the baseline.

Reported per point: labour share after automation (gamma = 0 at the
converged L) and after reinstatement, displaced mass, seeded-mass fates
(captured / bound / unbound, shares of seeded mass), re-sorted mass,
share of occupations with negative bundle wage change.

The baseline row must reproduce scripts/09 (0.6301 / 0.6259, unbound 68%
of seeded, re-sorting 58%); asserted below.

Outputs:
    results/sensitivity_sweep.csv
    results/sensitivity_sweep_summary.txt

Usage:
    python scripts/14_sensitivity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.equilibrium import Equilibrium
from model.regime import regime

_spec = importlib.util.spec_from_file_location("_setup",
                                               Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

RESULTS = REPO_ROOT / "results"
R0, TAU0, BETA, GAMMA0 = 18.0, 0.08, 0.5, 0.5
RHO0, LAMOVER0 = 0.5, 1.0


def run_point(inp, tech, L0, *, R, tau, gamma, ell, rho, lam_over) -> dict:
    """Solve the sorting fixed point and evaluate the diagnostics at it."""
    eq = Equilibrium(inp, tech, R, tau, gamma, ell, BETA, wedge=None,
                     survival=True, rho=rho, lam_over=lam_over)
    eq.L0 = L0
    _, _, W0 = eq.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0, eq.d)
    out = eq.solve(c, kappa)

    diag = regime(inp, tech, out.L, R, tau, gamma, ell, BETA,
                  survival=True, rho=rho, lam_over=lam_over)
    diag0 = regime(inp, tech, out.L, R, tau, 0.0, ell, BETA,
                   survival=True, rho=rho, lam_over=lam_over)

    M = float(diag["M"])
    unbound = float(diag["unbound_mass"])
    bound = float(diag["bound_mass"])
    captured = M - bound - unbound
    dL = out.L - L0
    return {
        "R": R, "tau": tau, "gamma": gamma, "ell": ell,
        "rho": rho, "lam_over": lam_over,
        "converged": out.converged,
        "share_auto": diag0["labor_share"],
        "share_reinst": diag["labor_share"],
        "seeded": M,
        "captured_of_seeded": captured / M if M > 0 else np.nan,
        "bound_of_seeded": bound / M if M > 0 else np.nan,
        "unbound_of_seeded": unbound / M if M > 0 else np.nan,
        "resort_share": 0.5 * float(np.abs(dL).sum()),
        "dW_neg_share": float(np.mean(diag["dW_bundle"] < 0)),
    }


def main() -> None:
    inp, L0, _ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell0 = _setup.interpretable_ell(inp)

    points, tags = [], []

    def add(tag, **kw):
        base = dict(R=R0, tau=TAU0, gamma=GAMMA0, ell=ell0,
                    rho=RHO0, lam_over=LAMOVER0)
        base.update(kw)
        if base in points:
            return
        points.append(base)
        tags.append(tag)

    add("baseline")
    for m in (0.25, 0.5, 2.0, 4.0):                       # A: ell
        add("A:ell", ell=ell0 * m)
    for g in (0.25, 0.75):                                # A: gamma
        add("A:gamma", gamma=g)
    for m in (0.25, 0.5, 2.0, 4.0):                       # A: ell x gamma corners
        add("A:ell_g25", ell=ell0 * m, gamma=0.25)
        add("A:ell_g75", ell=ell0 * m, gamma=0.75)
    for R in (12.0, 24.0):                                # B: R
        add("B:R", R=R)
    for tau in (0.04, 0.16):                              # B: tau
        add("B:tau", tau=tau)
    for R in (12.0, 24.0):                                # B: corners
        for tau in (0.04, 0.16):
            add("B:Rtau", R=R, tau=tau)
    add("C:rho", rho=0.25)                                # C: shape
    add("C:rho", rho=1.0)
    add("C:lam_over", lam_over=0.0)

    rows = []
    for tag, kw in zip(tags, points):
        r = run_point(inp, tech, L0, **kw)
        r["block"] = tag
        rows.append(r)
        print(f"{tag:10s} R={kw['R']:4.0f} tau={kw['tau']:.2f} "
              f"g={kw['gamma']:.2f} ell={kw['ell']:.3f} rho={kw['rho']:.2f} "
              f"lo={kw['lam_over']:.0f} | auto {r['share_auto']:.3f} "
              f"reinst {r['share_reinst']:.3f} "
              f"unbound {100*r['unbound_of_seeded']:.0f}% "
              f"resort {100*r['resort_share']:.0f}%")

    df = pd.DataFrame(rows)
    base = df[df["block"] == "baseline"].iloc[0]
    assert abs(base["share_auto"] - 0.6301) < 5e-4, base["share_auto"]
    assert abs(base["share_reinst"] - 0.6259) < 5e-4, base["share_reinst"]
    assert abs(base["unbound_of_seeded"] - 0.68) < 0.01
    assert abs(base["resort_share"] - 0.58) < 0.01

    df.to_csv(RESULTS / "sensitivity_sweep.csv", index=False)

    lines = ["Sensitivity sweep (script 14). Baseline reproduces script 09.",
             f"  ell* = {ell0:.4f} (SD rule); all points converged: "
             f"{bool(df['converged'].all())}", ""]

    def block(name, sel):
        g = df[sel]
        lines.append(f"{name}  (n={len(g)})")
        for col, lab in [("share_auto", "labour share, automation"),
                         ("share_reinst", "labour share, reinstatement"),
                         ("unbound_of_seeded", "unbound / seeded"),
                         ("captured_of_seeded", "captured / seeded"),
                         ("resort_share", "re-sorted mass")]:
            lines.append(f"  {lab:28s} {g[col].min():.3f} .. {g[col].max():.3f}")
        gap = g["share_auto"] - g["share_reinst"]
        lines.append(f"  {'reinstatement gap (auto-reinst)':28s} "
                     f"{gap.min():+.4f} .. {gap.max():+.4f}")
        lines.append("")

    block("A: attachment ell x gamma", df["block"].str.startswith(("A", "base")))
    block("B: adoption gate R x tau", df["block"].str.startswith(("B", "base")))
    block("C: attachment shape", df["block"].str.startswith(("C", "base")))

    # the two claims the sweep is for
    reinst_never_restores = bool((df["share_reinst"] <= df["share_auto"] + 1e-9).all())
    lines.append(f"Reinstatement raises the labour share at no point in the sweep: "
                 f"{reinst_never_restores}")
    lines.append(f"Unbound share of seeded mass across all points: "
                 f"{100*df['unbound_of_seeded'].min():.0f}% .. "
                 f"{100*df['unbound_of_seeded'].max():.0f}%")

    (RESULTS / "sensitivity_sweep_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS/'sensitivity_sweep.csv'} and _summary.txt")


if __name__ == "__main__":
    main()
