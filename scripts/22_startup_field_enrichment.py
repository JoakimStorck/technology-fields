"""
22_startup_field_enrichment.py
------------------------------
Where does the AI startup ecosystem sit relative to the model's three fields?
Producer 21 embeds the startups and writes their disk positions to
results/startup_seeding_startups.csv; this producer reads those positions and
measures, for AI and robotics separately:

  d_epi   distance to the technology centre p_K;
  d_ring  distance to the seeding ring, |d_epi - z_K| (the radial gap to the
          circle of radius z_K where |grad phi_K| and seeding peak);
  field enrichment at the startup locations vs a uniform-disk null --
    zeta  seeding density |grad phi_K|            (on the ring)
    u     unbound density                          (seeded and unbindable)
    a     operated share                           (incidence / displacement)
  reported as median(field at points) / disk-mean, so > 1 = over-represented.

FINDINGS (1,885 AI, 153 robotics, 77 carrying both tags as in Fenoaltea's
RSE; committed anchored equilibrium R=18, tau=0.08, gamma=0.5, beta=0.5):
  AI startups sit on the seeding ring's inner flank (zeta ~1.30x at the
  disk null; d_epi median 0.447 against z_K 0.583) and at the incidence
  core's edge (a ~0.92x); their unbound enrichment (u ~1.31x) equals
  their ring enrichment (u/zeta ~1.01), i.e. they lie on the ring, not
  specifically on its unbound arc. Robotics startups are out of the core
  entirely (a ~0.03x) and on the unbound arc (u/zeta ~1.26), in the
  technical-physical west; n is small. The startups sit on the gradient
  field's rim, away from the technology centre. Producer 30 adds that
  the bound-work densities (iota, B_o) fit both groups better than any
  raw seeding curve: the occupation layer carries the fit.
  Field-form note: the fields scored here are the model's circular
  primitives; see the FIELD FORM block in scripts/33 before reading any
  fit-quality statement in this file as a defect.
  Measurement note: document embeddings shrink norms toward the corpus
  mean (max startup chi 0.58 against the occupational support 0.75), so
  radial positions are conservative; angular placement, which carries
  the group separation, is unaffected by the shrinkage.

EMPIRICAL NULL (control corpus; expectations written before its first
run). fetch_startups --control-n samples YC companies carrying NEITHER
tag set; producer 21 embeds and projects them identically; this
producer reports them beside AI and robotics with bootstrap intervals
and the median-zeta contrast. Read with the shrinkage note in mind: any
document corpus shrinks toward the origin, and the origin sits at
0.79 z_K from p_K where zeta is ~0.95 of peak, so a centre blob scores
mechanically elevated radial and field statistics. The discriminating
comparison is ANGULAR.
  C1  The control sits centre-shrunken with a diffuse or east-south
      compass (general product language), against AI's ring-arc spread
      and robotics' western concentration; its zeta enrichment may be
      high for the mechanical reason above and is not read as ring
      placement.
  C2  The robotics-control separation is sharp: the western compass
      concentration and the a-enrichment gap have no counterpart in a
      centre blob.
  C3  Recorded either way: if the control reproduces the full pattern
      including the western arc, position is a corpus artifact and the
      validation section's claims are scoped down accordingly.

Figures:
  results/startup_field_enrichment_map.png     disk: occupations, startups,
      p_K, the z_K ring, and the a-core / u-periphery radii
  results/startup_field_enrichment_radial.png  radial position vs the a/ring/u
      fields, and the field-enrichment bars with u/zeta
Summary: results/startup_field_enrichment_summary.txt

Reads results/startup_seeding_startups.csv (from producer 21) and the frozen
inputs in data/. No API, no network.

Usage:
    python scripts/22_startup_field_enrichment.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model.equilibrium import Equilibrium
from model.regime import regime, _cell_index

_spec = importlib.util.spec_from_file_location(
    "_setup", Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

RESULTS = REPO_ROOT / "results"
STARTUPS = RESULTS / "startup_seeding_startups.csv"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
POLES = [(0, "human-centred\n(east)"), (90, "analytical (north)"),
         (180, "technical-physical\n(west)"), (270, "service (south)")]

# ── figure tuning (vary these; used by the AI/robotics density panels) ──────
FILL_ALPHA = 0.15     # fill transparency for the AI and robotics densities
FILL_GAMMA = 0.50     # density**gamma before contouring; lower lifts the sparse
                      # tails and the smaller robotics field into visibility
FILL_LEVELS = 12      # number of filled contour levels
CMAP_TRUNC = 0.30     # drop this fraction of the near-white colormap low end
AI_CMAP, RO_CMAP = "Blues", "Reds"          # AI vs robotics fill colormaps
KDE_BW = 0.06         # density kernel bandwidth on the disk
AI_PT = dict(color="#08519c", s=5, alpha=0.30)     # AI point cloud
RO_PT = dict(color="#a50f15", s=11, alpha=0.85)    # robotics point cloud
# ────────────────────────────────────────────────────────────────────────────


def mw_median_dist(w, d, area):
    w = np.maximum(np.asarray(w, float), 0.0) * area
    o = np.argsort(d)
    cw = np.cumsum(w[o]) / max(w.sum(), 1e-30)
    return float(np.interp(0.5, cw, d[o]))


def density_on_grid(xi, chi, grid, bw=0.06, weights=None):
    px, py = chi * np.cos(xi), chi * np.sin(xi)
    d2 = (grid.x[:, None] - px[None, :]) ** 2 + (grid.y[:, None] - py[None, :]) ** 2
    k = np.exp(-0.5 * d2 / bw ** 2)
    if weights is not None:
        k = k * np.asarray(weights, float)[None, :]
    kde = k.sum(axis=1)
    Z = (kde * grid.area).sum()
    return kde / Z if Z > 0 else kde


def _sector(deg):
    return np.array(["E", "N", "W", "S"])[((deg + 45) % 360 // 90).astype(int)]


def _disk(ax):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1))
    for a in range(0, 360, 30):
        t = np.radians(a)
        ax.plot([0, np.cos(t)], [0, np.sin(t)], color="0.9", lw=0.5, zorder=0)
    for deg, lab in POLES:
        t = np.radians(deg)
        ax.text(1.13 * np.cos(t), 1.13 * np.sin(t), lab, ha="center",
                va="center", fontsize=8, color="0.35")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")


def build_model():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)
    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    c, kappa, _, eq.alpha = _setup.anchor_reference(eq, L0)
    out = eq.solve(c, kappa)
    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA, wedge=None,
                  survival=True)
    g = inp.grid
    px, py = tech.p_K
    d = np.hypot(g.x - px, g.y - py)
    fields = {"a": tech.operated_share(g.xi, g.chi, inp.field, R, TAU),
              "zeta": tech.grad_phi_norm(g.xi, g.chi), "u": diag["u"]}
    return dict(inp=inp, tech=tech, occ=occ, g=g, d=d, px=px, py=py,
                z_K=tech.z_K, fields=fields, B_o=diag["B_o"],
                unbound_share=float(diag["unbound_mass"] / diag["M"]),
                med_a=mw_median_dist(fields["a"], d, g.area),
                med_u=mw_median_dist(diag["u"], d, g.area))


def enrichment_block(su, M, ctrl=None):
    g, d, px, py, z_K = M["g"], M["d"], M["px"], M["py"], M["z_K"]
    tech, inp = M["tech"], M["inp"]
    A = g.area.sum()
    null = {kk: float((vv * g.area).sum() / A) for kk, vv in M["fields"].items()}
    dual = int(((su.get("is_ai", 0) == 1)
                & (su.get("is_robotics", 0) == 1)).sum())
    lines = [
        "Startup field-enrichment (producer 22; positions from 21).",
        f"  p_K: chi_K {tech.chi_K:.3f}, xi_K {np.degrees(tech.xi_K):.1f} deg;  "
        f"z_K {z_K:.3f};  a-core {M['med_a']:.3f};  u-periphery {M['med_u']:.3f}",
        f"  disk-mean nulls:  zeta {null['zeta']:.4f}  u {null['u']:.4f}  "
        f"a {null['a']:.3f}",
        f"  groups overlap: {dual} firms carry both tags (as in Fenoaltea's "
        f"RSE); each group is reported inclusively",
        "",
    ]
    enr = {}
    rng = np.random.default_rng(22)
    N_BOOT = 2000

    def _stats(dd):
        xi, chi = dd["xi"].to_numpy(), dd["chi"].to_numpy()
        cx, cy = chi * np.cos(xi), chi * np.sin(xi)
        d_epi = np.hypot(cx - px, cy - py)
        ze = tech.grad_phi_norm(xi, chi)
        ap = tech.operated_share(xi, chi, inp.field, R, TAU)
        # u exists only on the grid (nearest cell); zeta and a are analytic.
        # At 4,800 cells the mixed evaluation is a negligible error source.
        up = M["fields"]["u"][_cell_index(g, xi, chi)]
        return d_epi, ze, ap, up

    def _ci(vals, stat):
        n = len(vals[0])
        idx = rng.integers(0, n, size=(N_BOOT, n))
        draws = stat(*[v[idx] for v in vals])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(lo), float(hi)

    ctrl_stats = None
    groups = [("AI startups", su[su.get("is_ai", 1) == 1]),
              ("robotics", su[su.get("is_robotics", 0) == 1])]
    if ctrl is not None and len(ctrl):
        groups.append(("control (non-AI/robotics)", ctrl))
    for tag, dd in groups:
        if not len(dd):
            continue
        d_epi, ze, ap, up = _stats(dd)
        e = {"zeta": float(np.median(ze) / null["zeta"]),
             "u": float(np.median(up) / null["u"]),
             "a": float(np.median(ap) / null["a"])}
        enr[tag] = e
        z_lo, z_hi = _ci([ze], lambda z: np.median(z, axis=1) / null["zeta"])
        d_lo, d_hi = _ci([d_epi], lambda d_: np.median(d_, axis=1))
        if tag.startswith("control"):
            ctrl_stats = dict(ze=ze, d_epi=d_epi)
        xi = dd["xi"].to_numpy()
        d_ring = np.abs(d_epi - z_K)
        sec = pd.Series(_sector(np.degrees(xi) % 360)).value_counts(normalize=True)
        secs = "  ".join(f"{s} {sec.get(s, 0.0):.0%}" for s in "ENWS")
        lines += [
            f"  {tag} (n={len(dd)}):",
            f"    d_epi  median {np.median(d_epi):.3f} "
            f"[{d_lo:.3f}, {d_hi:.3f}]   d_ring  median "
            f"{np.median(d_ring):.3f}   beyond z_K/2, i.e. d_ring < d_epi: "
            f"{float((d_ring < d_epi).mean()):.0%}",
            f"    enrichment  zeta {e['zeta']:.2f}x "
            f"[{z_lo:.2f}, {z_hi:.2f}]  u {e['u']:.2f}x  "
            f"a {e['a']:.2f}x   (u/zeta {e['u'] / e['zeta']:.2f})",
            f"    compass  {secs}",
            "",
        ]

    if ctrl_stats is not None:
        # the empirical-null contrast. Radial and field statistics are weak
        # against a centre-shrunken control (the origin sits at 0.79 z_K
        # from p_K, where zeta is ~0.95 of peak); the angular structure --
        # the compass rows above -- carries the discrimination.
        for tag, dd in groups[:2]:
            d_epi, ze, ap, up = _stats(dd)
            ratio = float(np.median(ze) / np.median(ctrl_stats["ze"]))
            n1, n2 = len(ze), len(ctrl_stats["ze"])
            idx1 = rng.integers(0, n1, size=(N_BOOT, n1))
            idx2 = rng.integers(0, n2, size=(N_BOOT, n2))
            draws = (np.median(ze[idx1], axis=1)
                     / np.median(ctrl_stats["ze"][idx2], axis=1))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            lines += [f"  {tag} vs control: median-zeta ratio "
                      f"{ratio:.2f} [{lo:.2f}, {hi:.2f}]"]
        lines.append("")
    return lines, enr


def _raw_kde(df, g, bw=0.06):
    """Un-normalised KDE (sum of kernels), so AI and robotics are on a shared
    absolute scale: robotics reads fainter because it has fewer firms."""
    xi = df["xi"].to_numpy(); chi = df["chi"].to_numpy()
    px, py = chi * np.cos(xi), chi * np.sin(xi)
    d2 = (g.x[:, None] - px[None, :]) ** 2 + (g.y[:, None] - py[None, :]) ** 2
    return np.exp(-0.5 * d2 / bw ** 2).sum(axis=1)


def _trunc_cmap(name, lo=CMAP_TRUNC, hi=1.0, n=256):
    """Colormap with the near-white low end dropped, so even the faintest
    filled band reads as a visible colour on a white disk."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_t", base(np.linspace(lo, hi, n)))


_CMAP_AI = _trunc_cmap(AI_CMAP)
_CMAP_RO = _trunc_cmap(RO_CMAP)


def _contours_and_points(ax, ai, ro, g):
    """AI (blue) and robotics (red) as FILLED contours on a shared scale, with
    the point clouds on top. The shared density is raised to FILL_GAMMA (<1) so
    the peak is compressed and the sparse tails become visible; the same
    transform is applied to both fields, so a blue and a red band still mean the
    same underlying density. All look parameters are the FILL_/CMAP_/PT config
    at the top of the file."""
    at = _raw_kde(ai, g, bw=KDE_BW) ** FILL_GAMMA
    levels = np.linspace(at.max() * 0.06, at.max(), FILL_LEVELS)
    ax.tricontourf(g.x, g.y, at, levels=levels, cmap=_CMAP_AI, alpha=FILL_ALPHA,
                   zorder=2, extend="max")
    if len(ro):
        ax.tricontourf(g.x, g.y, _raw_kde(ro, g, bw=KDE_BW) ** FILL_GAMMA,
                       levels=levels, cmap=_CMAP_RO, alpha=FILL_ALPHA,
                       zorder=3, extend="max")
    ax.scatter(ai["chi"] * np.cos(ai["xi"]), ai["chi"] * np.sin(ai["xi"]),
               edgecolors="none", zorder=4, **AI_PT)
    if len(ro):
        ax.scatter(ro["chi"] * np.cos(ro["xi"]), ro["chi"] * np.sin(ro["xi"]),
                   edgecolors="none", zorder=5, **RO_PT)


def figure_map(su, M):
    g, px, py = M["g"], M["px"], M["py"]
    occ = M["occ"]
    ai = su[su.get("is_ai", 1) == 1]
    ro = su[su.get("is_robotics", 0) == 1]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 6.6))
    _disk(axA)
    axA.scatter(occ["chi"] * np.cos(occ["xi"]), occ["chi"] * np.sin(occ["xi"]),
                s=6, color="0.8", zorder=1, label="occupations")
    axA.scatter(ai["chi"] * np.cos(ai["xi"]), ai["chi"] * np.sin(ai["xi"]),
                s=10, color="#1f77b4", alpha=0.55, edgecolors="none", zorder=2,
                label=f"AI startups (n={len(ai)})")
    if len(ro):
        axA.scatter(ro["chi"] * np.cos(ro["xi"]), ro["chi"] * np.sin(ro["xi"]),
                    s=12, color="#d62728", alpha=0.6, edgecolors="none", zorder=3,
                    label=f"robotics (n={len(ro)})")
    axA.scatter([px], [py], marker="x", s=90, color="k", zorder=5)
    axA.add_patch(plt.Circle((px, py), M["z_K"], fill=False, color="k",
                             ls="--", lw=1.1, zorder=4))
    axA.text(px, py + 0.06, "$p_K$", ha="center", fontsize=9, zorder=6)
    axA.set_title("Where the AI startups land")
    axA.legend(loc="lower left", frameon=False, fontsize=8)

    _disk(axB)
    _contours_and_points(axB, ai, ro, g)
    for rad, col, lab in [(M["med_a"], "0.25", "a(r) core"),
                          (M["med_u"], "#2ca02c", "u(r) periphery")]:
        axB.add_patch(plt.Circle((px, py), rad, fill=False, color=col,
                                 ls="--", lw=1.4, zorder=6))
        axB.text(px, py + rad + 0.02, lab, ha="center", color=col, fontsize=8,
                 zorder=7)
    axB.scatter([px], [py], marker="x", s=90, color="0.15", zorder=8)
    axB.set_title("AI (blue) and robotics (red) firms vs core and ring")
    fig.suptitle("AI startups in the occupational task geometry", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS / "startup_field_enrichment_map.png", dpi=150)
    plt.close(fig)


def figure_radial(su, M, enr):
    g, d, px, py = M["g"], M["d"], M["px"], M["py"]
    ai = su[su.get("is_ai", 1) == 1]
    ro = su[su.get("is_robotics", 0) == 1]
    edges = np.linspace(0, d.max(), 21)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _prof(w):
        bi = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
        num = np.bincount(bi, weights=w * g.area, minlength=len(edges) - 1)
        den = np.bincount(bi, weights=g.area, minlength=len(edges) - 1)
        p = np.where(den > 0, num / den, 0.0)
        return p / p.max() if p.max() > 0 else p

    def _hist(dd):
        h, _ = np.histogram(dd, bins=edges)
        return h / h.max() if h.max() > 0 else h

    def _dist(df):
        return np.hypot(df["chi"] * np.cos(df["xi"]) - px,
                        df["chi"] * np.sin(df["xi"]) - py)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    for w, ls, lab in [(M["fields"]["a"], "-", "a(r) incidence"),
                       (M["fields"]["zeta"], ":", "seeding ring $|\\nabla\\phi_K|$"),
                       (M["fields"]["u"], "-", "u(r) unbound")]:
        axA.plot(centers, _prof(w), ls, lw=2, label=lab)
    axA.axvline(M["z_K"], color="0.4", lw=1, ls="--")
    axA.text(M["z_K"], 1.03, "$z_K$", ha="center", fontsize=9, color="0.3")
    wb = edges[1] - edges[0]
    axA.bar(centers, _hist(_dist(ai)), width=wb * 0.9, alpha=0.25,
            color="#1f77b4", label=f"AI startups (n={len(ai)})")
    if len(ro):
        axA.bar(centers, _hist(_dist(ro)), width=wb * 0.5, alpha=0.55,
                color="#d62728", label=f"robotics (n={len(ro)})")
    axA.set_xlabel("distance from technology centre $p_K$ (disk units)")
    axA.set_ylabel("peak-normalised density")
    axA.set_title("Radial position vs the model fields")
    axA.legend(frameon=False, fontsize=8)

    labs = ["$\\zeta$\nseeding\nring", "$u$\nunbound", "$a$\nincidence"]
    xp = np.arange(3)
    ea = enr.get("AI startups")
    er = enr.get("robotics")
    if ea:
        axB.bar(xp - 0.19, [ea["zeta"], ea["u"], ea["a"]], width=0.38,
                color="#1f77b4", label="AI")
    if er:
        axB.bar(xp + 0.19, [er["zeta"], er["u"], er["a"]], width=0.38,
                color="#d62728", label="robotics")
    axB.axhline(1.0, color="0.4", ls="--", lw=1)
    axB.text(2.42, 1.03, "disk null", fontsize=8, color="0.4", ha="right")
    axB.set_xticks(xp); axB.set_xticklabels(labs, fontsize=8)
    axB.set_ylabel("enrichment vs disk null (ratio)")
    uz_a = ea["u"] / ea["zeta"] if ea and ea["zeta"] else float("nan")
    uz_r = er["u"] / er["zeta"] if er and er["zeta"] else float("nan")
    axB.set_title(f"Field enrichment at startup locations\n"
                  f"u/$\\zeta$: AI {uz_a:.2f}   robotics {uz_r:.2f}", fontsize=10)
    axB.legend(frameon=False, fontsize=8)
    fig.suptitle("Startups on the seeding ring, out of the incidence core",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS / "startup_field_enrichment_radial.png", dpi=150)
    plt.close(fig)


def figure_comparison(su, M, ctrl=None):
    """Model bound reinstatement (B_o over occupations) beside the empirical
    startup positions, with unbound shown as a field and a share, never as
    points (unbound work has no occupations to place)."""
    g, px, py = M["g"], M["px"], M["py"]
    occ = M["occ"]
    B_o = np.asarray(M["B_o"], float)
    occ_xi = occ["xi"].to_numpy(); occ_chi = occ["chi"].to_numpy()
    ai = su[su.get("is_ai", 1) == 1]
    ro = su[su.get("is_robotics", 0) == 1]

    def _dist(xi, chi):
        return np.hypot(chi * np.cos(xi) - px, chi * np.sin(xi) - py)

    fig = plt.figure(figsize=(15.6, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    axL, axM, axR = (fig.add_subplot(gs[i]) for i in range(3))

    # Left: model bound reinstatement B_o over occupations
    _disk(axL)
    densB = density_on_grid(occ_xi, occ_chi, g, weights=B_o)
    axL.tricontourf(g.x, g.y, densB, levels=12, cmap="Purples", zorder=1)
    axL.add_patch(plt.Circle((px, py), M["z_K"], fill=False, color="0.25",
                             ls="--", lw=1.2, zorder=3))
    axL.scatter([px], [py], marker="x", s=80, color="0.15", zorder=4)
    axL.set_title("Model: bound reinstatement $B_o$ over occupations")

    # Middle: empirical AI (blue) and robotics (red) contour lines + points
    _disk(axM)
    if ctrl is not None and len(ctrl):
        axM.scatter(ctrl["chi"] * np.cos(ctrl["xi"]),
                    ctrl["chi"] * np.sin(ctrl["xi"]),
                    s=4, color="0.6", alpha=0.25, zorder=1)
        axM.plot([], [], color="0.6", lw=1.5,
                 label=f"control (n={len(ctrl)})")
    _contours_and_points(axM, ai, ro, g)
    axM.add_patch(plt.Circle((px, py), M["z_K"], fill=False, color="0.25",
                             ls="--", lw=1.2, zorder=6))
    axM.scatter([px], [py], marker="x", s=80, color="0.15", zorder=7)
    axM.plot([], [], color="#1f77b4", lw=1.5, label=f"AI (n={len(ai)})")
    if len(ro):
        axM.plot([], [], color="#d62728", lw=1.5,
                 label=f"robotics (n={len(ro)})")
    axM.legend(loc="lower left", frameon=False, fontsize=8)
    axM.set_title("Empirical: AI and robotics firms")

    # Right: radial profiles, model vs empirical, against the ring and z_K
    edges = np.linspace(0, M["d"].max(), 21)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _fieldprof(w):
        bi = np.clip(np.digitize(M["d"], edges) - 1, 0, len(edges) - 2)
        num = np.bincount(bi, weights=w * g.area, minlength=len(edges) - 1)
        den = np.bincount(bi, weights=g.area, minlength=len(edges) - 1)
        p = np.where(den > 0, num / den, 0.0)
        return p / p.max() if p.max() > 0 else p

    def _whist(dvals, w=None):
        h, _ = np.histogram(dvals, bins=edges, weights=w)
        return h / h.max() if h.max() > 0 else h

    axR.plot(centers, _fieldprof(M["fields"]["zeta"]), ":", lw=2, color="C1",
             label="seeding ring $|\\nabla\\phi_K|$")
    axR.fill_between(centers, _fieldprof(M["fields"]["u"]), color="C2",
                     alpha=0.12, zorder=0)
    axR.plot(centers, _fieldprof(M["fields"]["u"]), "-", lw=1, color="C2",
             alpha=0.6, label="u(r) unbound (field)")
    axR.plot(centers, _whist(_dist(occ_xi, occ_chi), B_o), "-", lw=2.4,
             color="#5b3a8e", label="model bound reinstatement $B_o$")
    axR.bar(centers, _whist(_dist(ai["xi"].to_numpy(), ai["chi"].to_numpy())),
            width=(edges[1] - edges[0]) * 0.9, alpha=0.30, color="#1f77b4",
            label=f"AI startups (n={len(ai)})")
    if len(ro):
        axR.bar(centers, _whist(_dist(ro["xi"].to_numpy(), ro["chi"].to_numpy())),
                width=(edges[1] - edges[0]) * 0.5, alpha=0.55, color="#d62728",
                label=f"robotics (n={len(ro)})")
    axR.axvline(M["z_K"], color="0.4", lw=1, ls="--")
    axR.text(M["z_K"], 1.03, "$z_K$", ha="center", fontsize=9, color="0.3")
    axR.text(0.97, 0.55, f"unbound share\n{M['unbound_share']:.0%} of seeded",
             transform=axR.transAxes, ha="right", va="top", fontsize=8,
             color="C2")
    axR.set_xlabel("distance from technology centre $p_K$ (disk units)")
    axR.set_ylabel("peak-normalised density")
    axR.set_title("Radial position: model vs empirical")
    axR.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle("Model prediction beside the empirical startup positions",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "startup_field_enrichment_comparison.png", dpi=150)
    plt.close(fig)


def figure_points(su, M):
    """Standalone point cloud: AI and robotics firms as scatter, to see the
    actual spread and sparsity that the density fills smooth over."""
    g, px, py = M["g"], M["px"], M["py"]
    occ = M["occ"]
    ai = su[su.get("is_ai", 1) == 1]
    ro = su[su.get("is_robotics", 0) == 1]
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    _disk(ax)
    ax.scatter(occ["chi"] * np.cos(occ["xi"]), occ["chi"] * np.sin(occ["xi"]),
               s=6, color="0.82", zorder=1, label="occupations")
    ax.scatter(ai["chi"] * np.cos(ai["xi"]), ai["chi"] * np.sin(ai["xi"]),
               s=8, color="#1f77b4", alpha=0.45, edgecolors="none", zorder=2,
               label=f"AI startups (n={len(ai)})")
    if len(ro):
        ax.scatter(ro["chi"] * np.cos(ro["xi"]), ro["chi"] * np.sin(ro["xi"]),
                   s=14, color="#d62728", alpha=0.75, edgecolors="none",
                   zorder=3, label=f"robotics (n={len(ro)})")
    ax.add_patch(plt.Circle((px, py), M["z_K"], fill=False, color="k",
                            ls="--", lw=1.1, zorder=4))
    ax.scatter([px], [py], marker="x", s=90, color="k", zorder=5)
    ax.text(px, py + 0.06, "$p_K$", ha="center", fontsize=9, zorder=6)
    ax.set_title("AI and robotics firms in the task geometry")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "startup_field_enrichment_points.png", dpi=150)
    plt.close(fig)


def main():
    if not STARTUPS.exists():
        sys.exit(f"missing {STARTUPS}. Run producer 21 first to embed and place "
                 "the startups.")
    su = pd.read_csv(STARTUPS)
    if "xi" not in su.columns or "chi" not in su.columns:
        sys.exit("startup file has no xi/chi columns; re-run producer 21.")
    M = build_model()
    ctrl_file = RESULTS / "startup_control_positions.csv"
    ctrl = pd.read_csv(ctrl_file) if ctrl_file.exists() else None
    lines, enr = enrichment_block(su, M, ctrl)
    figure_map(su, M)
    figure_radial(su, M, enr)
    figure_comparison(su, M, ctrl)
    figure_points(su, M)
    txt = "\n".join(lines) + "\n"
    (RESULTS / "startup_field_enrichment_summary.txt").write_text(txt)
    print(txt)
    for fn in ("map", "radial", "comparison", "points"):
        print(f"wrote {RESULTS / ('startup_field_enrichment_' + fn + '.png')}")


if __name__ == "__main__":
    main()
