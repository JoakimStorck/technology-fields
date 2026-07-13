"""
d17: the birth margin, activated -- new occupations claim the residual (M3).

No new mechanism. The engine's existing birth machinery (fit-gap trigger
at e* = exp(-1), carrying capacity carry_thresh, staffing viability)
becomes meaningful once the match floor (M1) leaves a persistent residual
for births to claim; without the floor incumbents absorb everything first
and the machinery is idle. The floor is set at f_min = e*, so the same
threshold that blocks incumbent binding defines the vacant niche -- the
floor and the fit gap are one object.

Theoretical basis: incumbents search locally in capability space (Nelson
and Winter 1982), so the distant residual stays vacant; new organisational
forms are founded in unexploited niches under carrying-capacity conditions
(Hannan and Freeman 1977, 1989) -- the engine's birth rule is ecological
niche theory; recombination of unclaimed elements (Weitzman 1998); entry
where incumbent capabilities do not reach (Klepper 1996); where new work
appears (Autor, Chin, Salomons and Seegmiller 2024; Lin 2011).

Pre-registered hypotheses (frozen from the development pass; the
certified reference run confirms):

(H1) With the floor at e* and births on, births occur (the machinery is
     no longer idle) and they land on the residual: the mean static
     unbound density u(r) at birth locations exceeds the seeded-mass-
     weighted mean of u by a factor of at least two.
     [Outcome: FALSIFIED in the density form. Births fire (10) and land
     in the gap region outside z_K, but on its priced northern rim
     (mean u at births 0.8x the weighted mean, not >2x): the birth rule
     maximises Pi-weighted potential, so entrants chase value, not
     mass. Recorded; the density claim is replaced by the frozen
     ratio.]

(H2) Births resolve residual: the end unbound stock with births on is
     strictly below the births-off control at the same floor, and the
     newborn occupations hold a strictly positive share of bound mass.

(H3) The race: with task decay on (delta = 0.2), decay and births split
     the control residual; the split is frozen. Faster decay would starve
     the birth channel -- the tempo is the referee (forward look; only
     one decay point is certified here).

(H4) Geography: the median birth distance from the technology centre
     lies outside z_K, on the robotics side of the AI/robotics divide of
     the startup validation (AI median 0.447, robotics 0.635, z_K 0.583).
     [Outcome: the distance holds (median 0.779 > z_K) but the direction
     does not -- births sit on the analytical north rim, not the
     technical-physical west arc where the robotics firms sit. The
     model's entrants weight price; the empirical entrants may weight
     mass or feasibility. An informative mismatch, for the memo's
     follow-up, not hidden.]

Reference tempo theta = 3, T_max = 20, f_min = e*; engine defaults for
birth_every, carry_thresh, max_births. The carrying-capacity rule
predates the anchoring repair (development memo, Section 4); this
producer certifies its behaviour under the anchored kernel rather than
re-deriving it.

Usage: python experiment/d17_birth_margin.py
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

ESTAR = float(np.exp(-1.0))

# Frozen baselines (development pass).
BASE = {"n_births": 10, "med_dist": 0.779, "u_ratio": 0.82,
        "resolved": 0.155, "newborn_bound": 0.0032, "emp_new": 0.00018,
        "race_resolved": 0.971, "race_lost": 0.977, "U_control": 0.0206}


def main():
    layer = iface.load_static_layer()
    eq, g = layer.eq, layer.inp.grid
    area = eq.area
    px, py = layer.tech.p_K
    zK = layer.tech.z_K
    sh = lambda f: float(np.sum(f * area))

    layer.set_maturity(layer.tech.A_K)
    L = eq.solve(layer.c, layer.kappa).L
    a = eq.a_grid
    C = L @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    u_s = eq.g_hat * (1.0 - a) * (1.0 - Phi)

    runs = {}
    for key, kw in (("control", dict(max_births=0)),
                    ("births", dict()),
                    ("race", dict(task_decay=0.2))):
        dyn, rec, _ = rd.main(T_max=20.0, verbose=False, layer=layer,
                              anchored=True, f_min=ESTAR, **kw)
        newborn_bound = sum(sh(gf) for o, gf in dyn.grain.items())
        runs[key] = {"U_end": sh(dyn.U), "B": sh(dyn.B), "lost": sh(dyn.lost),
                     "births": list(dyn.birth_log), "newborn_bound": newborn_bound,
                     "emp_new": float(rec["emp_newborn"][-1]),
                     "U_path": np.asarray(rec["U_tot"])}

    b = runs["births"]
    n_b = len(b["births"])
    assert n_b > 0, "birth machinery idle despite the floor"

    # H1: fodslarna sitter pa residualen
    bx = np.array([x for (_, x, _) in b["births"]])
    by = np.array([y for (_, _, y) in b["births"]])
    cell = np.array([int(np.argmin((g.x - x) ** 2 + (g.y - y) ** 2))
                     for x, y in zip(bx, by)])
    u_at_births = float(np.mean(u_s[cell]))
    u_mean = float(np.sum(u_s * eq.g_hat * area) / np.sum(eq.g_hat * area))
    u_ratio = u_at_births / u_mean

    # H2: fodslar loser residual
    assert b["U_end"] < runs["control"]["U_end"] - 1e-6, "births do not resolve residual"
    assert b["newborn_bound"] > 1e-6, "newborns hold no bound mass"
    resolved = 1.0 - b["U_end"] / runs["control"]["U_end"]

    # H3: kapplopningen
    r = runs["race"]
    race_resolved = 1.0 - r["U_end"] / runs["control"]["U_end"]
    lost_share_of_control = r["lost"] / runs["control"]["U_end"]

    # H4: geografi
    dists = np.hypot(bx - px, by - py)
    med_d = float(np.median(dists))
    assert med_d > zK, f"births inside z_K: median {med_d:.3f}"

    stats = {"n_births": n_b, "med_dist": med_d, "u_ratio": u_ratio,
             "resolved": resolved,
             "newborn_bound": b["newborn_bound"], "emp_new": b["emp_new"],
             "race_resolved": race_resolved,
             "race_lost": lost_share_of_control,
             "U_control": runs["control"]["U_end"]}
    for k, v in BASE.items():
        tol = 0.5 if k == "n_births" else max(0.002, 0.05 * abs(v))
        assert abs(stats[k] - v) <= tol, f"{k} drifted: {stats[k]:.4f} vs {v}"

    # figur
    fig = plt.figure(figsize=(12.6, 5.0))
    axL = fig.add_subplot(1, 2, 1)
    axL.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1))
    axL.tricontourf(g.x, g.y, u_s, levels=12, cmap="Oranges", zorder=1)
    axL.add_patch(plt.Circle((px, py), zK, fill=False, color="0.25", ls="--",
                             lw=1.2, zorder=3))
    axL.scatter([px], [py], marker="x", s=70, color="0.15", zorder=4)
    try:
        import pandas as pd
        su = pd.read_csv(REPO / "results" / "startup_seeding_startups.csv")
        ro = su[su["is_robotics"] == 1]
        axL.scatter(ro["chi"] * np.cos(ro["xi"]), ro["chi"] * np.sin(ro["xi"]),
                    s=8, color="#d62728", alpha=0.6, zorder=5,
                    label=f"robotics firms (n={len(ro)})")
    except Exception:
        pass
    axL.scatter(bx, by, marker="*", s=140, color="#0e7c66", edgecolor="k",
                lw=0.5, zorder=6, label=f"births (n={n_b})")
    axL.legend(loc="lower left", frameon=False, fontsize=8)
    axL.set_xlim(-1.15, 1.15); axL.set_ylim(-1.15, 1.15)
    axL.set_aspect("equal"); axL.axis("off")
    axL.set_title("Births land on the static unbound field")
    axR = fig.add_subplot(1, 2, 2)
    ts = np.arange(len(runs["control"]["U_path"])) * 0.2
    axR.plot(ts, runs["control"]["U_path"], color="0.5", lw=2,
             label="floor, births off (control)")
    axR.plot(ts, b["U_path"], color="#0e7c66", lw=2, label="floor, births on")
    axR.plot(ts, runs["race"]["U_path"], color="#c8452a", lw=2,
             label="floor, births + decay 0.2")
    axR.set_xlabel("year"); axR.set_ylabel("unbound stock $U_{tot}$")
    axR.set_title("The residual: kept, claimed, or lost")
    axR.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(iface.RESULTS / "birth_margin.png", dpi=150)

    lines = ["birth_margin -- new occupations claim the residual (M3)",
             f"floor f_min = e* = {ESTAR:.3f}, theta = 3, engine birth defaults",
             "",
             f"births: {n_b}; median distance from p_K {med_d:.3f}"
             f" (z_K {zK:.3f}); mean u at births {u_at_births:.3f}"
             f" against g-weighted mean {u_mean:.3f} (x{u_at_births/u_mean:.1f})",
             f"residual: control U_end {runs['control']['U_end']:.4f};"
             f" births resolve {resolved:.1%}; newborn bound mass"
             f" {b['newborn_bound']:.4f}; newborn employment {b['emp_new']:.5f}",
             f"race (decay 0.2): resolved {race_resolved:.1%} of control residual,"
             f" of which lost-to-decay {lost_share_of_control:.1%}",
             ""]
    for (tb, x, y) in b["births"]:
        lines.append(f"  birth t={tb:5.1f} at ({x:+.2f},{y:+.2f}),"
                     f" d(p_K)={np.hypot(x-px, y-py):.2f}")
    lines += ["", "all frozen-baseline asserts passed."]
    print("\n".join(lines))
    out = iface.RESULTS / "birth_margin_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
