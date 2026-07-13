"""
33_field_form_invariance.py
---------------------------
FIELD FORM IS A PRIMITIVE, NOT AN EMPIRICAL CLAIM. The model represents a
technology as an ISOTROPIC Gaussian field (eq. phi-K): three interpretable
parameters, centre, reach, amplitude. This is a theoretical simplification of
the same kind as the one-dimensional task line in Acemoglu-Restrepo: nobody
claims real technologies are circular, and calibration therefore asks for the
best CIRCULAR field the exposure surface supports, not the best field. Richer
shapes (anisotropic covariances, other decay laws) fit better by construction,
because they nest the circle; refitting one and observing a higher R2 is
expected and is not a finding. Surfaces without an interior maximum (ramps,
such as Webb's robot exposure) pin the fitted centre of a circular field to
the edge of the support; that too is a property of approximating a ramp with
a circle, not a defect. Do not report imperfect fit, boundary-pinned centres,
or better-fitting alternatives as discoveries; the model's claims do not rest
on the shape. What the claims DO rest on is certified here.

THE CERTIFIED QUESTION. Refit the cognitive technology on the same task-level
exposure surface with a strictly richer shape, a full 2x2 covariance
(anisotropic Gaussian: centre, two axes, orientation, amplitude), and run the
SAME anchored economy under both fields: same alpha anchoring rule, same R,
tau, gamma, beta, ell. If the manuscript's numbers depend on the circular
simplification, they move here. The exploratory run behind this producer found
they do not: the seeding density itself changes substantially (grid Spearman
about 0.65 between the two fields), while every reported aggregate moves by
less than a point. This producer freezes that as the certified record.

EXPECTATIONS (informed by the exploratory run; the certified run is the
record, and adverse outcomes are reported):
  I1  labour share (automation and reinstatement) moves < 1.0 pt
  I2  the reinstatement gap moves < 0.5 pt
  I3  the unbound share of seeded mass moves < 2.0 pt
  I4  re-sorted employment mass moves < 1.0 pt
  I5  the AI startup ring enrichment (producer 22's median form) moves < 0.05
  G   guard: the isotropic run reproduces the committed numbers
      (labour share 0.6488 / 0.6439, unbound 0.676, resort 0.1373)

The anisotropic refit itself is reported for the record (centre, axes, R2
against the isotropic R2) but carries no hypothesis: that it fits better is
arithmetic, not evidence.

Outputs:
    results/field_form_invariance.csv
    results/field_form_invariance_summary.txt

Usage:
    python scripts/33_field_form_invariance.py
    SMOKE=1 python scripts/33_field_form_invariance.py   (coarse grid; the
        guard tolerances widen and nothing is certified)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.equilibrium import Equilibrium               # noqa: E402
from model.regime import regime                         # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_setup = _load("_setup.py")
_calib = _load("08_calibrate_technology.py")

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5

SMOKE = os.environ.get("SMOKE", "") == "1"
GRID = dict(n_ang=60, n_rad=20) if SMOKE else dict(n_ang=120, n_rad=40)

# Certified isotropic corner (scripts 09/17); the guard.
FROZEN = dict(lam_auto=0.6488, lam_reinst=0.6439, unbound=0.676, resort=0.1373)
GUARD_TOL = 0.003

THRESH = dict(share=0.010, gap=0.005, unbound=0.020, resort=0.010,
              enrich=0.05)


# ─────────────────────────────────────────────────────────────────────
# The strictly richer field: full 2x2 covariance, duck-typing Technology
# ─────────────────────────────────────────────────────────────────────

class AnisoTechnology:
    """phi(r) = A exp[-1/2 (u^2/sx^2 + v^2/sy^2)], with (u, v) the coordinates
    of r - p_K rotated by theta. Exposes the interface the economy consumes:
    phi, grad_phi_norm, operated_share, s_K. The rotation is orthogonal, so
    |grad phi| = phi * hypot(u/sx^2, v/sy^2)."""

    def __init__(self, chi_K, xi_K, sx, sy, theta, A_K, s_K=1.0):
        self.chi_K, self.xi_K = float(chi_K), float(xi_K)
        self.sx, self.sy, self.theta = float(sx), float(sy), float(theta)
        self.A_K, self.s_K = float(A_K), float(s_K)

    @property
    def p_K(self):
        return (self.chi_K * np.cos(self.xi_K),
                self.chi_K * np.sin(self.xi_K))

    def _uv(self, xi, chi):
        x = np.asarray(chi, float) * np.cos(np.asarray(xi, float))
        y = np.asarray(chi, float) * np.sin(np.asarray(xi, float))
        px, py = self.p_K
        dx, dy = x - px, y - py
        ct, st = np.cos(self.theta), np.sin(self.theta)
        return ct * dx + st * dy, -st * dx + ct * dy

    def phi(self, xi, chi):
        u, v = self._uv(xi, chi)
        return self.A_K * np.exp(-0.5 * ((u / self.sx) ** 2
                                         + (v / self.sy) ** 2))

    def grad_phi_norm(self, xi, chi):
        u, v = self._uv(xi, chi)
        return self.phi(xi, chi) * np.hypot(u / self.sx ** 2,
                                            v / self.sy ** 2)

    def operated_share(self, xi, chi, field, R, tau, log_wedge=0.0):
        pi_eff = np.exp(log_wedge) * field.pi(xi, chi)
        margin = self.s_K * self.phi(xi, chi) - R / pi_eff
        return 1.0 / (1.0 + np.exp(-margin / tau))


def fit_aniso(x, y, b):
    """Fit the anisotropic field to the same task-level exposure surface
    script 08 calibrates on, seeded from the certified isotropic optimum."""
    iso = _setup.load_tech()

    def resid(par):
        c, a, sx, sy, th, A = par
        px, py = c * np.cos(a), c * np.sin(a)
        dx, dy = x - px, y - py
        ct, st = np.cos(th), np.sin(th)
        u, v = ct * dx + st * dy, -st * dx + ct * dy
        return A * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)) - b

    x0 = [iso.chi_K, iso.xi_K, iso.z_K, iso.z_K, 0.0, iso.A_K]
    res = least_squares(resid, x0, method="trf",
                        bounds=([0, -2 * np.pi, 1e-3, 1e-3, -np.pi, 0],
                                [2.0, 2 * np.pi, 4.0, 4.0, np.pi, 5.0]))
    sst = float(((b - b.mean()) ** 2).sum())
    r2 = 1.0 - 2.0 * res.cost / sst
    return res.x, r2


# ─────────────────────────────────────────────────────────────────────

def run_economy(inp, L0, tech):
    """Anchor and solve exactly as script 09 does (wedge-free reading)."""
    eq0 = Equilibrium(inp, tech, R, TAU, GAMMA, ell_g, BETA, wedge=None,
                      survival=True)
    eq0.L0 = L0
    c, kappa, dmed, alpha = _setup.anchor_reference(eq0, L0)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell_g, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    eq.alpha = alpha
    out = eq.solve(c, kappa)

    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell_g, BETA, wedge=None,
                  survival=True)
    diag0 = regime(inp, tech, out.L, R, TAU, 0.0, ell_g, BETA, wedge=None,
                   survival=True)
    seeded = float(diag["M"])
    return dict(
        c=c, kappa=kappa, converged=bool(out.converged),
        lam_auto=float(diag0["labor_share"]),
        lam_reinst=float(diag["labor_share"]),
        unbound=float(diag["unbound_mass"]) / seeded if seeded > 0 else np.nan,
        resort=float(np.abs(out.L - L0).sum() / 2.0),
    )


def enrichment(tech, group="ai"):
    """Producer 22's median form: seeding density at the frozen startup
    positions over the disk mean."""
    f = RESULTS / "startup_seeding_startups.csv"
    if not f.exists():
        return np.nan
    s = pd.read_csv(f)
    s = s[s["is_ai"] == 1] if group == "ai" else s[s["is_robotics"] == 1]
    n_ang, n_rad = 180, 90
    ang = (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang
    rad = (np.arange(n_rad) + 0.5) / n_rad
    XI, CHI = np.meshgrid(ang, rad, indexing="ij")
    XI, CHI = XI.ravel(), CHI.ravel()
    area = CHI * (1.0 / n_rad) * (2 * np.pi / n_ang)
    g = tech.grad_phi_norm(XI, CHI)
    disk_mean = float((g * area).sum() / area.sum())
    v = tech.grad_phi_norm(s["xi"].to_numpy(), s["chi"].to_numpy())
    return float(np.median(v)) / disk_mean


def main() -> None:
    global ell_g
    RESULTS.mkdir(exist_ok=True)

    inp, L0, occ = _setup.build_inputs(**GRID)
    ell_g = _setup.interpretable_ell(inp)
    iso = _setup.load_tech()

    df = _calib.load_exposure_surface()
    par, r2_an = fit_aniso(df["x"].to_numpy(), df["y"].to_numpy(),
                           df["phi"].to_numpy(float))
    c_, a_, sx, sy, th, A = par
    ani = AnisoTechnology(c_, a_, sx, sy, th, A, s_K=1.0)

    d_iso = np.hypot(df["x"] - iso.p_K[0], df["y"] - iso.p_K[1])
    r2_iso = 1.0 - float(((iso.A_K * np.exp(-0.5 * (d_iso / iso.z_K) ** 2)
                           - df["phi"]) ** 2).sum()) \
        / float(((df["phi"] - df["phi"].mean()) ** 2).sum())

    lines = [
        "Field-form invariance (script 33; framing and expectations in the "
        "docstring).",
        f"  grid {GRID['n_ang']}x{GRID['n_rad']}"
        f"{'  SMOKE (not a certified run)' if SMOKE else ''}; "
        f"R {R}, tau {TAU}, gamma {GAMMA}, beta {BETA}, ell {ell_g:.4f}",
        "",
        "The circular primitive and the strictly richer refit, same "
        "task-level exposure surface:",
        f"  isotropic  : chi_K {iso.chi_K:.3f}  xi_K "
        f"{np.degrees(iso.xi_K) % 360:5.1f} deg  z_K {iso.z_K:.3f}  "
        f"A {iso.A_K:.3f}   R2 {r2_iso:.4f}",
        f"  anisotropic: chi_K {c_:.3f}  xi_K {np.degrees(a_) % 360:5.1f} deg"
        f"  axes ({sx:.3f}, {sy:.3f})  A {A:.3f}   R2 {r2_an:.4f}",
        "  (the richer form fitting better is arithmetic, not a finding; "
        "see the docstring)",
        "",
    ]

    econ = {}
    for tag, tech in [("isotropic", iso), ("anisotropic", ani)]:
        econ[tag] = run_economy(inp, L0, tech)
        e = econ[tag]
        lines += [f"  {tag:<11} anchored: kappa {e['kappa']:.3f}, "
                  f"c {e['c']:.3f}, converged {e['converged']}"]
    lines += [""]

    g = inp.grid
    za = iso.grad_phi_norm(g.xi, g.chi)
    zb = ani.grad_phi_norm(g.xi, g.chi)
    rho = float(spearmanr(za, zb).statistic)

    i, n = econ["isotropic"], econ["anisotropic"]
    gap_i = i["lam_auto"] - i["lam_reinst"]
    gap_n = n["lam_auto"] - n["lam_reinst"]
    enr_i = enrichment(iso, "ai")
    enr_n = enrichment(ani, "ai")

    rows = []

    def rep(name, vi, vn, thr, key):
        move = vn - vi
        ok = abs(move) < thr
        rows.append(dict(quantity=name, isotropic=vi, anisotropic=vn,
                         move=move, threshold=thr, holds=ok))
        lines.append(f"  {name:<32} {vi:9.4f} {vn:12.4f} {move:+9.4f}   "
                     f"{'HOLDS' if ok else 'MOVES'} (thr {thr})")
        return ok

    lines += [f"  {'quantity':<32} {'isotropic':>9} {'anisotropic':>12} "
              f"{'move':>9}", "  " + "-" * 78]
    i1a = rep("labour share, automation", i["lam_auto"], n["lam_auto"],
              THRESH["share"], "lam_auto")
    i1b = rep("labour share, reinstatement", i["lam_reinst"],
              n["lam_reinst"], THRESH["share"], "lam_reinst")
    i2 = rep("reinstatement gap", gap_i, gap_n, THRESH["gap"], "gap")
    i3 = rep("unbound share of seeded mass", i["unbound"], n["unbound"],
             THRESH["unbound"], "unbound")
    i4 = rep("re-sorted employment mass", i["resort"], n["resort"],
             THRESH["resort"], "resort")
    i5 = rep("AI startup ring enrichment", enr_i, enr_n, THRESH["enrich"],
             "enrich")

    lines += [
        "",
        f"  seeding density between the two fields, model grid: "
        f"Spearman {rho:+.3f}",
        "  (the density itself is NOT invariant; the reported quantities "
        "are. That is the point:",
        "   the manuscript's numbers are integrals the field form does not "
        "reach.)",
        "",
    ]

    guard_ok = (abs(i["lam_auto"] - FROZEN["lam_auto"]) < GUARD_TOL
                and abs(i["lam_reinst"] - FROZEN["lam_reinst"]) < GUARD_TOL
                and abs(i["unbound"] - FROZEN["unbound"]) < GUARD_TOL * 10
                and abs(i["resort"] - FROZEN["resort"]) < GUARD_TOL * 10)
    guard_line = ("SKIPPED (coarse grid; the frozen corner belongs to the "
                  "certified grid)" if SMOKE
                  else ("PASS" if guard_ok else "FAIL"))
    lines += [
        "Verdicts:",
        f"  G  isotropic run reproduces the committed corner   {guard_line}",
        f"  I1 labour share moves < {100*THRESH['share']:.1f} pt            "
        f"       {'PASS' if (i1a and i1b) else 'FAIL'}",
        f"  I2 reinstatement gap moves < {100*THRESH['gap']:.1f} pt         "
        f"       {'PASS' if i2 else 'FAIL'}",
        f"  I3 unbound share moves < {100*THRESH['unbound']:.1f} pt         "
        f"       {'PASS' if i3 else 'FAIL'}",
        f"  I4 re-sorting moves < {100*THRESH['resort']:.1f} pt             "
        f"       {'PASS' if i4 else 'FAIL'}",
        f"  I5 AI enrichment moves < {THRESH['enrich']:.2f}                 "
        f"      {'PASS' if i5 else 'FAIL'}",
    ]
    if SMOKE:
        lines += ["", "*** SMOKE RUN: coarse grid, nothing certified. ***"]
    if not SMOKE and not guard_ok:
        lines += ["", "GUARD FAILED: the isotropic corner does not reproduce "
                  "the committed numbers; do not read the invariance rows."]

    pd.DataFrame(rows).to_csv(RESULTS / "field_form_invariance.csv",
                              index=False)
    (RESULTS / "field_form_invariance_summary.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS / 'field_form_invariance.csv'}")


if __name__ == "__main__":
    main()
