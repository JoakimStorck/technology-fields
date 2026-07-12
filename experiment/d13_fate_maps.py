"""
d13: fate maps -- where seeded mass binds, static equilibrium against the
dynamics, on the polar disk and as radial mass profiles.

The static equilibrium splits seeded mass into three fates at the solved
anchored allocation: capital captures zeta a, occupations bind
zeta (1-a) Phi(C), and the rest, zeta (1-a) (1-Phi), stays unbound with
no bearer. The dynamics split the same seeding into two: capital
captures at seeding time, gamma dGamma g_hat a(t) accumulated along the
maturation path (the engine's cap accumulator), and everything that
survives the gate is eventually bound by labour, since the residue
cascades to the next-best match in the same cell and U drains to zero.

The figure owns two comparisons. On the disk: the static bound field
iota(r) against the dynamic bound stock B(r), on a common colour scale,
with the startup positions of the static validation overlaid. Radially:
stacked mass profiles per model in a common unit, the share of seeded
mass per radial bin, so the areas under the bands equal the fate shares
and the magnitudes are honest (5 percent bound against 87).

Pre-registered hypotheses (frozen from the development pass on the
anchored pipeline; the certified reference run confirms them):

(H1) Static decomposition. The three fate fields sum to the seeding
     density cell by cell (identity, machine precision), and integrate
     to capital 0.274, bound 0.050, unbound 0.676 of seeded mass,
     matching the static paper's frozen shares.

(H2) Dynamic decomposition at the reference tempo theta = 3. Capital
     captures 0.133 and labour binds 0.867 of the run's seeded mass;
     U drains below 1e-8. The capital share is roughly half the static
     0.274 because seeding runs while the adoption gate is still
     opening, so early seed passes a gate the mature economy would
     close.

(H3) The capital share is monotone in the tempo: 0.123 at theta = 1,
     0.133 at theta = 3, 0.145 at theta = 15. Slower redistribution
     delays the displacement flow into a more open gate.

(H4) Tempo invariance of place. corr(B at theta 1, B at theta 15)
     exceeds 0.99 (measured 0.997) and the half-L1 distance between the
     normalised mass fields stays below 0.02 (measured 0.017). The tempo
     moves which occupations hold the work (d02); it does not move
     where the work binds.

(H5) Firm positions, inherited from the static validation (producer 21
     cache). The AI median distance from p_K is 0.447, inside
     z_K = 0.583; the robotics median is 0.635, outside. AI firms sit
     where both models bind; robotics firms sit where the static
     equilibrium leaves work unbound and the dynamics bind it.

Births are off (max_births = 0) and the occupation set is fixed, as in
d01-d11.

Usage: python experiment/d13_fate_maps.py
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
import pandas as pd


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

# Frozen baselines (development pass, anchored pipeline). Shares are
# fractions of seeded mass; tolerance 0.02 absolute unless noted.
BASE_STATIC = {"capital": 0.274, "bound": 0.050, "unbound": 0.676}
BASE_DYN = {1.0: 0.123, 3.0: 0.133, 15.0: 0.145}   # capital share per tempo
BASE_B_CORR = 0.9968      # corr(B theta 1, B theta 15); bound 0.99
BASE_B_HALFL1 = 0.0170    # half-L1 of the normalised mass fields; bound 0.02
BASE_MED = {"ai": 0.447, "robotics": 0.635}        # tolerance 0.005
TOL = 0.02
THETAS = (1.0, 3.0, 15.0)
THETA_REF = 3.0

STARTUPS = REPO / "results" / "startup_seeding_startups.csv"

POLES = [(0, "E"), (90, "N"), (180, "W"), (270, "S")]


def _disk(ax):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1))
    for ang in range(0, 360, 30):
        t = np.radians(ang)
        ax.plot([0, np.cos(t)], [0, np.sin(t)], color="0.9", lw=0.5, zorder=0)
    for deg, lab in POLES:
        t = np.radians(deg)
        ax.text(1.14 * np.cos(t), 1.14 * np.sin(t), lab, ha="center",
                va="center", fontsize=8, color="0.35")
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.26, 1.26)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    layer = iface.load_static_layer()
    eq, g = layer.eq, layer.inp.grid
    area = eq.area
    px, py = layer.tech.p_K
    zK = layer.tech.z_K
    d = np.hypot(g.x - px, g.y - py)
    sh = lambda f: float(np.sum(f * area))

    # -- static fates at the solved anchored equilibrium, mature field --
    layer.set_maturity(layer.tech.A_K)
    L = eq.solve(layer.c, layer.kappa).L
    a = eq.a_grid
    C = L @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    zeta = eq.g_hat
    cap_s = zeta * a
    iota_s = zeta * (1.0 - a) * Phi
    u_s = zeta * (1.0 - a) * (1.0 - Phi)

    gap = float(np.max(np.abs(cap_s + iota_s + u_s - zeta)))
    assert gap < 1e-12, f"static fates do not sum to the seeding density: {gap:.2e}"
    shares_s = {"capital": sh(cap_s), "bound": sh(iota_s), "unbound": sh(u_s)}
    for k, v in BASE_STATIC.items():
        assert abs(shares_s[k] - v) < TOL, \
            f"static {k} share drifted: {shares_s[k]:.3f} vs frozen {v}"

    # -- dynamic fates across tempos ------------------------------------
    runs = {}
    for th in THETAS:
        dyn, rec, _ = rd.main(T_max=20.0, theta_L=th, theta_abs=th,
                              verbose=False, layer=layer, anchored=True,
                              max_births=0)
        assert rec["U_tot"][-1] < 1e-8, f"U not drained at theta {th}"
        tot = sh(dyn.cap) + sh(dyn.B)
        runs[th] = {"cap": dyn.cap / tot, "B": dyn.B / tot,
                    "cap_share": sh(dyn.cap) / tot}
    for th, v in BASE_DYN.items():
        assert abs(runs[th]["cap_share"] - v) < TOL, \
            f"dynamic capital share drifted at theta {th}: " \
            f"{runs[th]['cap_share']:.3f} vs frozen {v}"
    assert runs[1.0]["cap_share"] < runs[3.0]["cap_share"] < runs[15.0]["cap_share"], \
        "capital share not monotone in tempo"

    b1 = runs[1.0]["B"]
    b15 = runs[15.0]["B"]
    corr = float(np.corrcoef(b1, b15)[0, 1])
    n1, n15 = b1 * area, b15 * area
    half_l1 = float(np.abs(n1 / n1.sum() - n15 / n15.sum()).sum() / 2)
    assert corr > 0.99 and abs(corr - BASE_B_CORR) < 0.005, \
        f"B field not tempo-invariant: corr {corr:.4f}"
    assert half_l1 < 0.02 and abs(half_l1 - BASE_B_HALFL1) < 0.005, \
        f"B field half-L1 drifted: {half_l1:.4f}"

    cap_d, B_d = runs[THETA_REF]["cap"], runs[THETA_REF]["B"]

    # -- firm positions from the static validation cache ----------------
    assert STARTUPS.exists(), \
        f"missing {STARTUPS}; run the static producer 21 first"
    su = pd.read_csv(STARTUPS)
    ai = su[su["is_ai"] == 1]
    ro = su[su["is_robotics"] == 1]
    _dist = lambda xi, chi: np.hypot(chi * np.cos(xi) - px, chi * np.sin(xi) - py)
    med = {"ai": float(np.median(_dist(ai["xi"], ai["chi"]))),
           "robotics": float(np.median(_dist(ro["xi"], ro["chi"])))}
    for k, v in BASE_MED.items():
        assert abs(med[k] - v) < 0.005, f"{k} median drifted: {med[k]:.3f}"
    assert med["ai"] < zK < med["robotics"], "firm medians do not straddle z_K"

    # -- radial mass profiles, share of seeded mass per bin -------------
    edges = np.linspace(0, d.max(), 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bi = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    mass = lambda f: np.bincount(bi, weights=f * area, minlength=len(edges) - 1)

    # -- figure ----------------------------------------------------------
    vmax = max(iota_s.max(), B_d.max())
    levels = np.linspace(0, vmax, 13)
    fig = plt.figure(figsize=(15.6, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.32)
    axL = fig.add_subplot(gs[:, 0])
    axM = fig.add_subplot(gs[:, 1])
    axS = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 2])

    for ax, F, tt in (
        (axL, iota_s, f"Static: bound field $\\iota(\\mathbf{{r}})$ "
                      f"— {shares_s['bound']:.0%} of seeded"),
        (axM, B_d, f"Dynamic: bound stock $B(\\mathbf{{r}})$ "
                   f"— {1 - runs[THETA_REF]['cap_share']:.0%} of seeded"),
    ):
        _disk(ax)
        ax.tricontourf(g.x, g.y, F, levels=levels, cmap="Purples", zorder=1)
        ax.add_patch(plt.Circle((px, py), zK, fill=False, color="0.25",
                                ls="--", lw=1.2, zorder=3))
        ax.scatter([px], [py], marker="x", s=80, color="0.15", zorder=4)
        ax.scatter(ai["chi"] * np.cos(ai["xi"]), ai["chi"] * np.sin(ai["xi"]),
                   s=4, color="#1f77b4", alpha=0.35, zorder=5)
        ax.scatter(ro["chi"] * np.cos(ro["xi"]), ro["chi"] * np.sin(ro["xi"]),
                   s=9, color="#d62728", alpha=0.75, zorder=6)
        ax.set_title(tt, fontsize=11)
    axM.plot([], [], color="#1f77b4", lw=1.5, label=f"AI (n={len(ai)})")
    axM.plot([], [], color="#d62728", lw=1.5, label=f"robotics (n={len(ro)})")
    axM.legend(loc="lower left", frameon=False, fontsize=8)

    mS = [mass(cap_s), mass(iota_s), mass(u_s)]
    mD = [mass(cap_d), mass(B_d)]
    ymax = 1.06 * max(sum(mS).max(), sum(mD).max())

    def stacked(ax, parts, colors, labels):
        base = np.zeros_like(centers)
        for p_, c_, l_ in zip(parts, colors, labels):
            ax.fill_between(centers, base, base + p_, color=c_, alpha=0.75,
                            lw=0, label=l_)
            base = base + p_
        ax.plot(centers, base, color="0.3", lw=0.8)
        ax.axvline(zK, color="0.4", lw=1, ls="--")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("share of seeded mass / bin", fontsize=9)
        for m_, c_ in ((med["ai"], "#1f77b4"), (med["robotics"], "#d62728")):
            ax.plot([m_], [0], marker="^", ms=8, color=c_, clip_on=False,
                    zorder=6)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax_ in (axS, axD):
        ax_.set_xlim(0, edges[-1])
    stacked(axS, mS, ["0.55", "#5b3a8e", "C2"],
            [f"capital {shares_s['capital']:.0%}",
             f"bound by labour {shares_s['bound']:.0%}",
             f"unbound {shares_s['unbound']:.0%}"])
    axS.set_title("Static: fate of seeded mass, per radius", fontsize=11)
    axS.text(zK, ymax * 1.01, "$z_K$", ha="center", fontsize=9, color="0.3")
    plt.setp(axS.get_xticklabels(), visible=False)

    stacked(axD, mD, ["0.55", "#0e7c66"],
            [f"capital {runs[THETA_REF]['cap_share']:.0%}",
             f"bound by labour {1 - runs[THETA_REF]['cap_share']:.0%}"])
    axD.set_title("Dynamic: fate of seeded mass, per radius", fontsize=11)
    axD.set_xlabel("distance from technology centre $p_K$ (disk units)")
    axD.plot([], [], marker="^", ls="none", color="#1f77b4",
             label="AI median position")
    axD.plot([], [], marker="^", ls="none", color="#d62728",
             label="robotics median")
    axD.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Where seeded work binds, and to what: static equilibrium "
                 "against the dynamics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(iface.RESULTS / "binding_fates_map.png", dpi=150)

    # -- summary ---------------------------------------------------------
    lines = [
        "fate_maps -- where seeded mass binds, static vs dynamic",
        "static fates at the solved anchored equilibrium, mature field;",
        "dynamic fates from the engine's capital accumulator, births off",
        "",
        f"(H1) static shares of seeded mass: capital {shares_s['capital']:.3f}"
        f"   bound {shares_s['bound']:.3f}   unbound {shares_s['unbound']:.3f}",
        f"     fate identity sup gap: {gap:.2e}",
        "",
        "(H2/H3) dynamic capital share per tempo (rest bound by labour):",
    ] + [
        f"     theta {th:>4}: capital {runs[th]['cap_share']:.3f}"
        f"   bound {1 - runs[th]['cap_share']:.3f}"
        for th in THETAS
    ] + [
        "",
        f"(H4) B field tempo invariance: corr(theta 1, theta 15) = {corr:.4f}"
        f"   half-L1 = {half_l1:.4f}",
        "     the tempo moves which occupations hold the work, not where it binds",
        "",
        f"(H5) firm median distance from p_K: AI {med['ai']:.3f}"
        f"   robotics {med['robotics']:.3f}   (z_K = {zK:.3f} between them)",
        "",
        "all frozen-baseline asserts passed.",
    ]
    print("\n".join(lines))
    out = iface.RESULTS / "binding_fates_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
