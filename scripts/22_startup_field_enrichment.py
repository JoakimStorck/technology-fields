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

FINDINGS (809 AI, 37 robotics; committed equilibrium R=18, tau=0.08, gamma=0.5,
beta=0.5):
  AI startups sit on the seeding ring (zeta ~1.30x) and out of the incidence
  core (a ~0.92x, at the disk null); their unbound enrichment (u ~1.28x) equals
  their ring enrichment (u/zeta ~0.99), i.e. they lie on the ring, not
  specifically on its unbound arc. Robotics startups are out of the core
  entirely (a ~0.04x) and on the unbound arc (u/zeta ~1.23), in the
  technical-physical west; n is small. The startups sit on the gradient field's
  rim, away from the technology centre.

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


def mw_median_dist(w, d, area):
    w = np.maximum(np.asarray(w, float), 0.0) * area
    o = np.argsort(d)
    cw = np.cumsum(w[o]) / max(w.sum(), 1e-30)
    return float(np.interp(0.5, cw, d[o]))


def density_on_grid(xi, chi, grid, bw=0.06):
    px, py = chi * np.cos(xi), chi * np.sin(xi)
    d2 = (grid.x[:, None] - px[None, :]) ** 2 + (grid.y[:, None] - py[None, :]) ** 2
    kde = np.exp(-0.5 * d2 / bw ** 2).sum(axis=1)
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
    _, _, W0 = eq.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0, eq.d)
    out = eq.solve(c, kappa)
    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA, wedge=None,
                  survival=True)
    g = inp.grid
    px, py = tech.p_K
    d = np.hypot(g.x - px, g.y - py)
    fields = {"a": tech.operated_share(g.xi, g.chi, inp.field, R, TAU),
              "zeta": tech.grad_phi_norm(g.xi, g.chi), "u": diag["u"]}
    return dict(inp=inp, tech=tech, occ=occ, g=g, d=d, px=px, py=py,
                z_K=tech.z_K, fields=fields,
                med_a=mw_median_dist(fields["a"], d, g.area),
                med_u=mw_median_dist(diag["u"], d, g.area))


def enrichment_block(su, M):
    g, d, px, py, z_K = M["g"], M["d"], M["px"], M["py"], M["z_K"]
    tech, inp = M["tech"], M["inp"]
    A = g.area.sum()
    null = {kk: float((vv * g.area).sum() / A) for kk, vv in M["fields"].items()}
    lines = [
        "Startup field-enrichment (producer 22; positions from 21).",
        f"  p_K: chi_K {tech.chi_K:.3f}, xi_K {np.degrees(tech.xi_K):.1f} deg;  "
        f"z_K {z_K:.3f};  a-core {M['med_a']:.3f};  u-periphery {M['med_u']:.3f}",
        f"  disk-mean nulls:  zeta {null['zeta']:.4f}  u {null['u']:.4f}  "
        f"a {null['a']:.3f}",
        "",
    ]
    enr = {}
    for tag, mask in [("AI startups", su.get("is_ai", 1) == 1),
                      ("robotics", su.get("is_robotics", 0) == 1)]:
        dd = su[mask]
        if not len(dd):
            continue
        xi, chi = dd["xi"].to_numpy(), dd["chi"].to_numpy()
        cx, cy = chi * np.cos(xi), chi * np.sin(xi)
        d_epi = np.hypot(cx - px, cy - py)
        d_ring = np.abs(d_epi - z_K)
        ze = tech.grad_phi_norm(xi, chi)
        ap = tech.operated_share(xi, chi, inp.field, R, TAU)
        up = M["fields"]["u"][_cell_index(g, xi, chi)]
        e = {"zeta": float(np.median(ze) / null["zeta"]),
             "u": float(np.median(up) / null["u"]),
             "a": float(np.median(ap) / null["a"])}
        enr[tag] = e
        sec = pd.Series(_sector(np.degrees(xi) % 360)).value_counts(normalize=True)
        secs = "  ".join(f"{s} {sec.get(s, 0.0):.0%}" for s in "ENWS")
        lines += [
            f"  {tag} (n={len(dd)}):",
            f"    d_epi  median {np.median(d_epi):.3f}   d_ring  median "
            f"{np.median(d_ring):.3f}   closer to ring: "
            f"{float((d_ring < d_epi).mean()):.0%}",
            f"    enrichment  zeta {e['zeta']:.2f}x  u {e['u']:.2f}x  "
            f"a {e['a']:.2f}x   (u/zeta {e['u'] / e['zeta']:.2f})",
            f"    compass  {secs}",
            "",
        ]
    return lines, enr


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
    dens = density_on_grid(ai["xi"].to_numpy(), ai["chi"].to_numpy(), g)
    axB.tricontourf(g.x, g.y, dens, levels=12, cmap="viridis", zorder=1)
    for rad, col, lab in [(M["med_a"], "white", "a(r) core"),
                          (M["med_u"], "red", "u(r) periphery")]:
        axB.add_patch(plt.Circle((px, py), rad, fill=False, color=col,
                                 ls="--", lw=1.4, zorder=3))
        axB.text(px, py + rad + 0.02, lab, ha="center", color=col, fontsize=8,
                 zorder=6)
    axB.scatter([px], [py], marker="x", s=90, color="white", zorder=5)
    axB.set_title("Startup density vs the model's core (a) and ring (u)")
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


def main():
    if not STARTUPS.exists():
        sys.exit(f"missing {STARTUPS}. Run producer 21 first to embed and place "
                 "the startups.")
    su = pd.read_csv(STARTUPS)
    if "xi" not in su.columns or "chi" not in su.columns:
        sys.exit("startup file has no xi/chi columns; re-run producer 21.")
    M = build_model()
    lines, enr = enrichment_block(su, M)
    figure_map(su, M)
    figure_radial(su, M, enr)
    txt = "\n".join(lines) + "\n"
    (RESULTS / "startup_field_enrichment_summary.txt").write_text(txt)
    print(txt)
    print(f"wrote {RESULTS / 'startup_field_enrichment_map.png'}")
    print(f"wrote {RESULTS / 'startup_field_enrichment_radial.png'}")


if __name__ == "__main__":
    main()
