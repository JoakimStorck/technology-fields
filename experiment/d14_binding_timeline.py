"""
d14: binding timeline -- the binding year by year, and who does it.

Eight deterministic runs with growing T_max snapshot the bound stock
B(r, t) at integer years; a final tracked run (track_binding=True)
attributes bound mass to occupations cell by cell through the engine's
occupation-by-cell accumulator. Reference tempo theta = 3, births off,
fixed price. Three views own the figure: the disks year by year (with
the unbound front U as contours where it is alive), the top-10 binder
occupations as named trajectories, and the end-state territory map of
the dominant job family per cell.

The temporal front and the spatial invariance are two sides of one
fact: the ring fills in proportion everywhere while the calendar moves,
so the front runs through time, not through the disk (d13's H4 is the
same statement at the endpoints).

Pre-registered hypotheses (frozen from the development pass on the
anchored pipeline; the certified reference run confirms them):

(H1) The yearly bound totals reproduce d09's fixed-price path at the
     shared years: 0.0016 (1), 0.0103 (2), 0.0452 (3), 0.0821 (4),
     0.0912 (5), 0.0914 (8). Binding saturates by year 5 at 99.8
     percent of the final stock.

(H2) Accounting closes. The per-occupation vector reinst sums to the
     bound stock integral at every snapshot (1e-9), and the tracked
     occupation-by-cell mass sums to the same total.

(H3) The reference tempo blends the regime archetypes of d02. Top
     three binders: Preventive Medicine Physicians (1.67 percent of
     reinstated mass), Chief Executives, Natural Sciences Managers;
     the top ten mix front specialists (Nanosystems Engineers, Fuel
     Cell Engineers) with large absorbers (Dishwashers, Dining Room
     Attendants).

(H4) The binding is broad but geographically sorted by family.
     Architecture and Engineering leads with 0.110 of bound mass,
     Production 0.098, Educational Instruction and Library 0.091; no
     family exceeds 0.15.

(H5) The unbound front peaks at year 3: U_tot(3) = 0.00267, the d01
     theta-3 peak (5 percent relative), and is drained at every year
     from 5 on.

Usage: python experiment/d14_binding_timeline.py
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
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

# Frozen baselines (development pass, anchored pipeline).
BASE_BTOT = {1: 0.0016, 2: 0.0103, 3: 0.0452, 4: 0.0821, 5: 0.0912, 8: 0.0914}
BASE_SATURATION = 0.998        # B_tot(5)/B_tot(8); tolerance 0.005
BASE_TOP3 = ("Preventive Medicine Physicians", "Chief Executives",
             "Natural Sciences Managers")
BASE_TOP1_SHARE = 0.0167       # of reinstated mass; tolerance 0.002
BASE_FAM3 = ("Architecture and Engineering", "Production",
             "Educational Instruction and Library")
BASE_FAM_SHARES = (0.110, 0.098, 0.091)   # tolerance 0.02
BASE_U3 = 0.00267              # d01 theta-3 peak; 5 percent relative
YEARS = list(range(1, 9))
TOPN = 10
TOPF = 7


def _disk(ax):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=0.8))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    layer = iface.load_static_layer()
    eq, g, occ = layer.eq, layer.inp.grid, layer.inp.occ
    area = eq.area
    px, py = layer.tech.p_K
    zK = layer.tech.z_K

    # -- yearly snapshots -------------------------------------------------
    B_t, U_t, R_t = [], [], []
    for k in YEARS:
        dyn, rec, _ = rd.main(T_max=float(k), verbose=False, layer=layer,
                              anchored=True, max_births=0)
        B_t.append(dyn.B.copy())
        U_t.append(dyn.U.copy())
        R_t.append(dyn.reinst[:dyn.n0].copy())
        Btot = float(np.sum(dyn.B * area))
        assert abs(Btot - float(np.sum(dyn.reinst))) < 1e-9, \
            f"reinst does not close against B at year {k}"
        if k in BASE_BTOT:
            b = BASE_BTOT[k]
            assert abs(Btot - b) < max(0.05 * b, 2e-4), \
                f"B_tot drifted at year {k}: {Btot:.4f} vs frozen {b}"
    B_t, U_t, R_t = np.stack(B_t), np.stack(U_t), np.stack(R_t)
    Btot_t = (B_t * area).sum(axis=1)
    Utot_t = (U_t * area).sum(axis=1)
    sat = Btot_t[4] / Btot_t[7]
    assert abs(sat - BASE_SATURATION) < 0.005, f"saturation drifted: {sat:.4f}"
    assert abs(Utot_t[2] - BASE_U3) <= 0.05 * BASE_U3, \
        f"U front at year 3 drifted: {Utot_t[2]:.5f}"
    assert all(Utot_t[k] < 1e-6 for k in (4, 5, 6, 7)), "U not drained after year 5"

    # -- final tracked run: who binds, and where --------------------------
    dyn, rec, _ = rd.main(T_max=20.0, verbose=False, layer=layer,
                          anchored=True, max_births=0, track_binding=True)
    Rfin = dyn.reinst[:dyn.n0].copy()
    bind_oc = dyn.bind_oc
    assert abs(float(bind_oc.sum()) - float(Rfin.sum())) < 1e-9, \
        "tracked occupation-by-cell mass does not close against reinst"

    titles = occ["Title"].to_numpy()
    top = np.argsort(Rfin)[::-1][:TOPN]
    top_titles = tuple(titles[o] for o in top[:3])
    assert top_titles == BASE_TOP3, f"top-3 binders changed: {top_titles}"
    top1 = float(Rfin[top[0]] / Rfin.sum())
    assert abs(top1 - BASE_TOP1_SHARE) < 0.002, f"top share drifted: {top1:.4f}"

    fams = occ["Job Family"].fillna("Other").to_numpy()
    ufam = sorted(set(fams))
    F = np.zeros((len(ufam), area.size))
    for i, fam in enumerate(ufam):
        F[i] = bind_oc[fams == fam].sum(axis=0)
    fam_mass = F.sum(axis=1)
    fam_share = fam_mass / fam_mass.sum()
    order = np.argsort(fam_mass)[::-1]
    fam3 = tuple(ufam[i] for i in order[:3])
    assert fam3 == BASE_FAM3, f"top families changed: {fam3}"
    for i, b in zip(order[:3], BASE_FAM_SHARES):
        assert abs(fam_share[i] - b) < 0.02, \
            f"family share drifted: {ufam[i]} {fam_share[i]:.3f} vs {b}"
    assert fam_share.max() < 0.15, "a single family dominates the binding"

    # -- figure ------------------------------------------------------------
    frac = Btot_t / Btot_t[-1]
    vmax = B_t[-1].max()
    Umax = U_t.max()
    cmap10 = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(15.6, 12.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.35],
                          hspace=0.18, wspace=0.05)
    for i in range(8):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        _disk(ax)
        ax.tricontourf(g.x, g.y, B_t[i], levels=np.linspace(0, vmax, 13),
                       cmap="Purples", zorder=1)
        if U_t[i].max() > 0.02 * Umax:
            ax.tricontour(g.x, g.y, U_t[i], levels=[0.3 * Umax, 0.6 * Umax],
                          colors="#e07020", linewidths=0.9, zorder=2)
        ax.add_patch(plt.Circle((px, py), zK, fill=False, color="0.3",
                                ls="--", lw=0.9, zorder=3))
        ax.plot([px], [py], "x", color="0.15", ms=6, zorder=4)
        ax.set_title(f"year {i + 1}  ·  {frac[i]:.0%} of final bound",
                     fontsize=10)

    axT = fig.add_subplot(gs[2, :2])
    yrs = np.arange(0, 9)
    for j, o in enumerate(top):
        traj = np.concatenate([[0.0], R_t[:, o]])
        axT.plot(yrs, 100 * traj / Rfin.sum(), color=cmap10(j % 10),
                 lw=1.8, label=titles[o][:44])
    axT.set_xlabel("year")
    axT.set_ylabel("share of final reinstated mass (%)")
    axT.set_title("Who binds the work: top-10 occupations, $\\theta = 3$",
                  fontsize=11)
    axT.legend(frameon=False, fontsize=7.2, loc="upper left")
    axT.set_xlim(0, 8)

    axF = fig.add_subplot(gs[2, 2:])
    _disk(axF)
    dens = F.sum(axis=0) / area
    dom = np.argmax(F, axis=0)
    alpha = np.clip((dens / dens.max()) ** 0.55, 0, 1)
    alpha[dens < 0.01 * dens.max()] = 0.0
    topf = list(order[:TOPF])
    colmap = {fi: cmap10(k) for k, fi in enumerate(topf)}
    dxi = 2 * np.pi / len(np.unique(g.xi))
    chi_u = np.unique(g.chi)
    dchi = float(np.diff(chi_u).mean())
    Q = np.empty((g.xi.size, 4, 2))
    for k, (dx, dc) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        a_, c_ = g.xi + dx * dxi / 2, np.clip(g.chi + dc * dchi / 2, 0, 1)
        Q[:, k, 0], Q[:, k, 1] = c_ * np.cos(a_), c_ * np.sin(a_)
    rgba = np.zeros((g.xi.size, 4))
    for c in range(g.xi.size):
        col = colmap.get(dom[c], (0.72, 0.72, 0.72, 1.0))
        rgba[c, :3] = col[:3]
        rgba[c, 3] = alpha[c]
    axF.add_collection(PolyCollection(Q, facecolors=rgba, edgecolors="none",
                                      zorder=1))
    axF.add_patch(plt.Circle((px, py), zK, fill=False, color="0.25",
                             ls="--", lw=1.1, zorder=3))
    axF.plot([px], [py], "x", color="0.1", ms=7, zorder=4)
    handles = [Patch(facecolor=cmap10(k),
                     label=f"{ufam[fi]}  ({fam_share[fi]:.0%})")
               for k, fi in enumerate(topf)]
    handles.append(Patch(facecolor="0.72", label="other families"))
    axF.legend(handles=handles, frameon=False, fontsize=7.2,
               loc="center left", bbox_to_anchor=(0.94, 0.5))
    axF.set_title("Whose territory: dominant job family of the binder, "
                  "end state", fontsize=11)

    fig.suptitle("The binding year by year, and who does it  "
                 "($\\theta = 3$, births off)", fontsize=13, y=0.985)
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(iface.RESULTS / "binding_timeline.png", dpi=140,
                bbox_inches="tight")

    # -- summary -----------------------------------------------------------
    lines = [
        "binding_timeline -- the binding year by year, and who does it",
        "reference tempo theta = 3, births off, fixed price;",
        "occupation-by-cell attribution from the tracked engine run",
        "",
        "(H1) bound stock per year (share of final in parens):",
    ] + [
        f"     year {k}: B_tot {Btot_t[k - 1]:.4f}  ({frac[k - 1]:.1%})"
        f"   U_tot {Utot_t[k - 1]:.5f}"
        for k in YEARS
    ] + [
        f"     saturation B(5)/B(8) = {sat:.4f}",
        "",
        f"(H3) top-{TOPN} binders (share of reinstated mass):",
    ] + [
        f"     {Rfin[o] / Rfin.sum():.4%}  {titles[o]}"
        for o in top
    ] + [
        "",
        f"(H4) top-{TOPF} job families of the binder:",
    ] + [
        f"     {fam_share[i]:.3f}  {ufam[i]}"
        for i in order[:TOPF]
    ] + [
        "",
        f"(H5) U front: peak year 3 at {Utot_t[2]:.5f}"
        f" (d01 theta-3 peak {BASE_U3}), drained from year 5",
        "",
        "all frozen-baseline asserts passed.",
    ]
    print("\n".join(lines))
    out = iface.RESULTS / "binding_timeline_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
