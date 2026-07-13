"""
d15: match floor -- a reservation match quality nests the two closes (M1).

The binding law acquires a floor f_min: an occupation claims a cell's
unbound mass only where FIT >= f_min. Cells with no claimant keep their
mass in U permanently. At f_min = 0 this is today's dynamics (complete
binding); as f_min rises the persistent residual grows toward the static
Phi-world; the closure choice becomes a measurable parameter instead of
two limiting assumptions.

Theoretical basis: reservation match quality (Jovanovic 1979; Mortensen
and Pissarides 1994; Shimer and Smith 2000); distance thresholds on the
circle in Marimon and Zilibotti (1999). No empirical anchor for f_min is
claimed; the point is the nesting. e* = exp(-1), the birth machinery's
fit-gap threshold, is marked on the sweep as the natural reference.

Pre-registered hypotheses (frozen from the development pass; the
certified reference run confirms):

(H1) The bound share of surviving seed falls monotonically in f_min from
     1.00 at f_min = 0, passing every level down to the static
     neighbourhood (0.069 = 0.050/0.726) at the top of the sweep. The
     bracket is a curve, not two points.

(H2) The spatial shape of the persistent residual converges toward the
     static unbound field u(r): the correlation with u(r) rises in f_min
     and is high at the top of the sweep.
     [Outcome: partial. The correlation rises from -0.12 at low floors to
     +0.49 at f_min = 0.82 -- directional convergence, crossing sign, but
     not complete; at low floors the residual is the worst-fit slice
     only, a different object from the static u.]

(H3) The tempo divergence survives the floor: the cross-regime
     correlation of reinstated mass (theta 1 against 15) stays far from
     1 at every floor tested, and reproduces the frozen d02 value +0.391
     at f_min = 0 (tie-in).
     [Outcome: the bound holds, but the floor DAMPENS the divergence
     (+0.39 -> +0.69 -> +0.72 across the tested floors): the floor
     removes exactly the poorly matched mass on which the regimes
     disagreed. Part of the destination result is carried by sub-floor
     binding; a substantive finding, recorded, not hidden.]

(H4) Accounting closes at every floor: capital + bound + U_end equals
     the seeded total by construction (the floor only stops exchange
     between U and B), and the per-occupation vector reinst closes
     against the bound stock integral at 1e-9.

Births off, reference tempo theta = 3 for the sweep, T_max = 20.

Usage: python experiment/d15_match_floor.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

FLOORS = (0.0, 0.2, 0.3, 0.368, 0.45, 0.55, 0.7, 0.82)
REGIME_FLOORS = (0.0, 0.368, 0.55)
ESTAR = float(np.exp(-1.0))

# Frozen baselines (development pass; tolerance 0.02 unless noted).
BASE_BOUND = {0.0: 1.000, 0.2: 0.969, 0.3: 0.854, 0.368: 0.775,
              0.45: 0.676, 0.55: 0.569, 0.7: 0.369, 0.82: 0.105}
BASE_UCORR = {0.2: -0.117, 0.3: -0.073, 0.368: -0.059, 0.45: -0.049,
              0.55: -0.038, 0.7: 0.114, 0.82: 0.493}
BASE_CROSS = {0.0: 0.391, 0.368: 0.690, 0.55: 0.717}
STATIC_BOUND_OF_SURVIVING = 0.069


def main():
    layer = iface.load_static_layer()
    eq = layer.eq
    area = eq.area
    sh = lambda f: float(np.sum(f * area))

    # static reference: unbound field at the solved anchored equilibrium
    layer.set_maturity(layer.tech.A_K)
    L = eq.solve(layer.c, layer.kappa).L
    a = eq.a_grid
    C = L @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    u_s = eq.g_hat * (1.0 - a) * (1.0 - Phi)

    rows = []
    for fm in FLOORS:
        dyn, rec, _ = rd.main(T_max=20.0, verbose=False, layer=layer,
                              anchored=True, max_births=0, f_min=fm)
        capm, Bm, Um = sh(dyn.cap), sh(dyn.B), sh(dyn.U)
        assert abs(float(np.sum(dyn.reinst)) - Bm) < 1e-9, f"reinst does not close at f_min {fm}"
        bound_surv = Bm / (Bm + Um)
        ucorr = float(np.corrcoef(dyn.U, u_s)[0, 1]) if Um > 1e-6 else float("nan")
        rows.append((fm, capm / (capm + Bm + Um), bound_surv, Um / (Bm + Um), ucorr))

    bs = [r[2] for r in rows]
    assert all(bs[i] >= bs[i + 1] - 1e-9 for i in range(len(bs) - 1)), \
        "bound share not monotone in the floor"
    assert abs(bs[0] - 1.0) < 1e-6, "no complete binding at f_min = 0"
    for (fm, _, b, _, uc) in rows:
        if fm in BASE_BOUND:
            assert abs(b - BASE_BOUND[fm]) < 0.02, \
                f"bound share drifted at f_min {fm}: {b:.3f} vs {BASE_BOUND[fm]}"
        if fm in BASE_UCORR and not np.isnan(uc):
            assert abs(uc - BASE_UCORR[fm]) < 0.02, \
                f"residual shape drifted at f_min {fm}: {uc:.3f} vs {BASE_UCORR[fm]}"

    # tempo divergence across the floor (H3)
    cross = {}
    for fm in REGIME_FLOORS:
        r_by = {}
        for th in (1.0, 15.0):
            dyn, rec, _ = rd.main(T_max=20.0, theta_L=th, theta_abs=th,
                                  verbose=False, layer=layer, anchored=True,
                                  max_births=0, f_min=fm)
            r_by[th] = dyn.reinst[:dyn.n0].copy()
        cross[fm] = float(np.corrcoef(r_by[1.0], r_by[15.0])[0, 1])
    assert abs(cross[0.0] - 0.391) < 0.02, f"d02 tie-in broken: {cross[0.0]:.3f}"
    for fm, c in cross.items():
        if fm in BASE_CROSS:
            assert abs(c - BASE_CROSS[fm]) < 0.02, \
                f"cross-regime corr drifted at f_min {fm}: {c:.3f} vs {BASE_CROSS[fm]}"
        assert c < 0.8, f"tempo divergence lost at f_min {fm}"

    # figure
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fms = [r[0] for r in rows]
    axL.plot(fms, bs, "o-", color="#0e7c66", lw=2)
    axL.axhline(STATIC_BOUND_OF_SURVIVING, color="#5b3a8e", ls="--", lw=1.2)
    axL.text(0.01, STATIC_BOUND_OF_SURVIVING + 0.02, "static close (7%)",
             fontsize=9, color="#5b3a8e")
    axL.axvline(ESTAR, color="0.5", ls=":", lw=1)
    axL.text(ESTAR, 1.02, "$e^*$", ha="center", fontsize=9, color="0.4")
    axL.set_xlabel("match floor $f_{\\min}$")
    axL.set_ylabel("bound share of surviving seed")
    axL.set_title("The floor nests the two closes")
    axL.set_ylim(0, 1.05)
    axR.plot([r[0] for r in rows if not np.isnan(r[4])],
             [r[4] for r in rows if not np.isnan(r[4])], "s-", color="C2", lw=2)
    axR.axvline(ESTAR, color="0.5", ls=":", lw=1)
    axR.set_xlabel("match floor $f_{\\min}$")
    axR.set_ylabel("corr(residual, static $u(\\mathbf{r})$)")
    axR.set_title("The residual converges to the static unbound field")
    for fm, c in cross.items():
        axR.annotate(f"cross-regime {c:+.2f} @ {fm:.2f}", xy=(0.03, 0.14 - 0.07 * list(cross).index(fm)),
                     xycoords="axes fraction", fontsize=8, color="0.35")
    fig.tight_layout()
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(iface.RESULTS / "match_floor.png", dpi=150)

    lines = ["match_floor -- reservation match quality nests the two closes (M1)",
             "theta = 3 sweep, births off; e* = exp(-1) marked",
             "",
             "  f_min  cap-share  bound(surv)  unbound(surv)  corr(res, u_s)"]
    for (fm, cs, b, u, uc) in rows:
        lines.append(f"  {fm:5.3f}    {cs:6.3f}     {b:7.3f}      {u:7.3f}"
                     f"        {'--' if np.isnan(uc) else f'{uc:+.3f}'}")
    lines += ["", "cross-regime pearson (theta 1 vs 15) across the floor:"]
    for fm, c in cross.items():
        lines.append(f"  f_min {fm:5.3f}: {c:+.3f}")
    lines += ["", "all frozen-baseline asserts passed."]
    print("\n".join(lines))
    out = iface.RESULTS / "match_floor_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
