"""
24_bundle_examples.py
---------------------
The companion panel to the compass map of scripts/15_geometry_map.py.
Where 15 shows every occupational centroid on the disk, this panel opens
a chosen few of its labelled examples and shows how an occupation is
*built* from tasks: the individual task locations, the bundle centroid,
and the task radius that measures how far the bundle spreads around it.

Which occupations appear is set by the SELECTED list below: it holds the
nine examples of the geometry map, all but the first five commented out.
Uncomment a line to add an occupation, comment one to drop it. The order
is the plotting order; colours and markers are assigned by position, the
first being the deep blue the geometry map uses to highlight examples.

The FIRST occupation in the list also carries the coordinate marks, as in
the polar-geometry figure of the paper: the radial depth chi (a dashed
line from the origin to the centroid), the angle xi (a small arc from the
east axis to that line), and the task radius r_o.

The drawing is factored into build_panel(numbered=False) so that the
supplementary producer (scripts/25_bundle_examples_numbered.py) can reuse
it to draw the identical figure with a fine number beside each task.

Objects drawn, in the notation of the geometry section:
  * task locations r = (xi, chi), plotted on the same scaled disk as 15;
  * the centroid mu_o = int r b_o(r) dr, the RT-weighted mean task
    position (the dot of Figure 15 for these occupations);
  * the task radius r_o, the RT-weighted root-mean-square distance of the
    bundle from its centroid,
        r_o = sqrt( int || r - mu_o ||^2 b_o(r) dr ),
    the bundle's radius of gyration on the disk.

The r -> chi scaling is uniform, so the RT-weighted mean of the scaled
task positions coincides exactly with the published centroid of
occupation_embeddings_polar_scaled.csv (asserted below); the panel's
centroids therefore land on the same points as Figure 15.

Output:
    results/bundle_examples.png
    results/bundle_examples_summary.txt

Usage:
    python scripts/24_bundle_examples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

# --- what to plot -------------------------------------------------------
# The nine examples of the geometry map (scripts/15_geometry_map.py), in
# plotting order. Comment a line out to drop that occupation; the first
# uncommented occupation carries the chi, xi and r_o marks. Each entry is
#   (onet_code, published xi in deg, (label dx, dy, ha)).
SELECTED = [
    ("15-1252.00",  75, ( 18, -12, "left")),    # Software Developers
    ("23-1011.00",   5, ( 16,   6, "left")),    # Lawyers
    ("51-4041.00", 164, (-16,   8, "right")),   # Machinists
    ("35-9021.00", 239, (-12, -14, "right")),   # Dishwashers
    ("35-3031.00", 272, (  0, -16, "left")),    # Waiters and Waitresses
    # ("17-2041.00", 109, (-16,  16, "right")),   # Chemical Engineers
    # ("29-1222.00",  44, ( 14,  10, "left")),    # Physicians, Pathologists
    # ("47-2031.00", 196, (-16,  -8, "right")),   # Carpenters
    # ("25-1042.00", 340, ( 14, -10, "left")),    # Biological Science Teachers
]

# Colour and marker by position; the first is the geometry-map highlight blue.
PALETTE = ["#1f3d7a", "#c1440e", "#2c7a3f", "#7a3fa0", "#b58a00",
           "#0f6d78", "#a01f4f", "#555555", "#8a5a2b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

POLES = [(90, "analytical"), (0, "human-centered"),
         (270, "service"), (180, "technical-physical")]

ARC_R = 0.11        # radius of the xi angle arc (about a third of the former 0.32)


def band(deg: float) -> str:
    """Compass direction of a task angle, matching the paper's poles."""
    d = deg % 360
    if 45 <= d < 135:
        return "analytical"
    if 135 <= d < 225:
        return "technical-physical"
    if 225 <= d < 315:
        return "service"
    return "human-centered"


def bundle(tasks: pd.DataFrame, occ: pd.Series) -> dict:
    """Task cloud, centroid and task radius for one occupation, on the
    scaled disk (x, y) = (chi cos xi, chi sin xi). Tasks are returned as a
    frame ordered by angle and numbered from one, for the labelled variant."""
    tt = tasks.copy()
    tt["deg"] = np.degrees(tt["xi"]) % 360
    tt = tt.sort_values("deg").reset_index(drop=True)
    tt["x"] = tt["chi"] * np.cos(tt["xi"])
    tt["y"] = tt["chi"] * np.sin(tt["xi"])
    tt["num"] = np.arange(1, len(tt) + 1)
    tt["dir"] = tt["deg"].map(band)
    w = tt["rt"].to_numpy()
    cx = occ["chi"] * np.cos(occ["xi"])         # published centroid on the disk
    cy = occ["chi"] * np.sin(occ["xi"])
    # the RT-weighted mean of the scaled tasks must reproduce that centroid
    mx = np.average(tt["x"], weights=w)
    my = np.average(tt["y"], weights=w)
    assert abs(mx - cx) < 1e-6 and abs(my - cy) < 1e-6, (occ["onet_code"],
                                                         mx, my, cx, cy)
    dist2 = (tt["x"] - cx) ** 2 + (tt["y"] - cy) ** 2
    r_o = float(np.sqrt(np.average(dist2, weights=w)))   # radius of gyration
    return {"tasks": tt, "cx": float(cx), "cy": float(cy), "r_o": r_o,
            "deg": float(np.degrees(occ["xi"]) % 360), "chi": float(occ["chi"]),
            "n": len(tt), "rt_sum": float(w.sum()),
            "title": occ["Title"], "onet_code": occ["onet_code"]}


def mark_coordinates(ax, b: dict, colour: str) -> None:
    """Draw the chi line, the xi arc and the r_o tick on one bundle."""
    # radial coordinate chi: dashed line from the origin to the centroid
    ax.plot([0, b["cx"]], [0, b["cy"]], color="0.30", lw=1.0,
            ls=(0, (4, 3)), zorder=5)
    ax.annotate(r"$\chi$", (0.5 * b["cx"], 0.5 * b["cy"]),
                textcoords="offset points", xytext=(-12, 3), fontsize=12,
                color="0.30", ha="right", va="center", zorder=7)
    # angular coordinate xi: small arc from the east axis to the centroid ray
    ax.add_patch(Arc((0, 0), 2 * ARC_R, 2 * ARC_R, angle=0.0,
                     theta1=0.0, theta2=b["deg"], color="0.30", lw=1.0,
                     zorder=5))
    mid = np.radians(b["deg"] / 2)
    ax.annotate(r"$\xi$", ((ARC_R + 0.055) * np.cos(mid),
                           (ARC_R + 0.055) * np.sin(mid)),
                fontsize=12, color="0.30", ha="center", va="center", zorder=7)
    # task radius r_o: a tick along the radius circle
    th = np.radians(45)
    ex, ey = b["cx"] + b["r_o"] * np.cos(th), b["cy"] + b["r_o"] * np.sin(th)
    ax.plot([b["cx"], ex], [b["cy"], ey], color=colour, lw=1.1, zorder=5)
    ax.annotate(r"$r_o$", (b["cx"] + 0.55 * b["r_o"] * np.cos(th),
                           b["cy"] + 0.55 * b["r_o"] * np.sin(th)),
                textcoords="offset points", xytext=(6, -8), fontsize=10,
                color=colour, zorder=7)


def number_tasks(ax, b: dict, colour: str) -> None:
    """Place a fine number beside each task point, fanned outward from the
    centroid to reduce overlap, with a light halo for legibility."""
    for _, r in b["tasks"].iterrows():
        vx, vy = r["x"] - b["cx"], r["y"] - b["cy"]
        nrm = np.hypot(vx, vy)
        ox, oy = (6 * vx / nrm, 6 * vy / nrm) if nrm > 1e-6 else (4, 3)
        ax.annotate(str(int(r["num"])), (r["x"], r["y"]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=6, color=colour, ha="center", va="center",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.04", fc="white",
                              ec="none", alpha=0.65))


def build_panel(numbered: bool = False):
    """Draw the bundle panel and return (fig, bundles). Does not save.
    With numbered=True a fine number is placed beside each task, keyed to
    the supplementary table of scripts/25_bundle_examples_numbered.py."""
    tasks = pd.read_csv(DATA / "task_embeddings_polar_scaled.csv")
    occs = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv")

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.75", lw=0.9))
    for ang in range(0, 360, 45):
        a = np.radians(ang)
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="0.9", lw=0.6, zorder=0)

    bundles = []
    for i, (code, pub_deg, (dx, dy, ha)) in enumerate(SELECTED):
        colour = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        occ_m = occs[occs["onet_code"] == code]
        task_m = tasks[tasks["onet_code"] == code]
        assert len(occ_m) == 1, (code, len(occ_m))
        assert not task_m.empty, code
        b = bundle(task_m, occ_m.iloc[0])
        b["colour"] = colour
        assert abs(b["deg"] - pub_deg) < 1.0, (code, b["deg"], pub_deg)

        ax.scatter(b["tasks"]["x"], b["tasks"]["y"], s=30, marker=marker,
                   facecolor=colour, edgecolors="white", linewidths=0.3,
                   alpha=0.70, zorder=3)
        ax.add_patch(plt.Circle((b["cx"], b["cy"]), b["r_o"], fill=False,
                                 color=colour, lw=1.5, ls=(0, (5, 3)),
                                 alpha=0.95, zorder=4))
        ax.scatter(b["cx"], b["cy"], s=185, marker="*", facecolor=colour,
                   edgecolors="black", linewidths=0.9, zorder=6)
        ax.annotate(b["title"].split(",")[0], (b["cx"], b["cy"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=10,
                    ha=ha, color=colour, zorder=7)

        if i == 0:
            mark_coordinates(ax, b, colour)
        if numbered:
            number_tasks(ax, b, colour)

        bundles.append(b)

    for ang, lab in POLES:
        a = np.radians(ang)
        ax.text(1.13 * np.cos(a), 1.13 * np.sin(a), lab, fontsize=11,
                ha="center", va="center", style="italic", color="0.25")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=6, label="task"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="0.45",
               markeredgecolor="black", markersize=13, label="centroid $\\mu_o$"),
        Line2D([0], [0], color="0.45", lw=1.5, ls=(0, (5, 3)), label="$r_o$"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False,
              handletextpad=0.6, borderpad=0.2, labelspacing=0.5)

    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    return fig, bundles


def main() -> None:
    fig, bundles = build_panel(numbered=False)
    fig.savefig(RESULTS / "bundle_examples.png", dpi=170)
    plt.close(fig)

    with open(RESULTS / "bundle_examples_summary.txt", "w") as fh:
        fh.write("Bundle examples for Figure 2 (companion to geometry_map).\n")
        fh.write("Centroid mu_o and task radius r_o (RT-weighted RMS distance\n")
        fh.write("from the centroid, on the scaled disk).\n\n")
        for b in bundles:
            fh.write(f"{b['title']}\n")
            fh.write(f"  centroid  xi={b['deg']:.1f} deg  chi={b['chi']:.4f}"
                     f"  (x,y)=({b['cx']:.4f},{b['cy']:.4f})\n")
            fh.write(f"  task radius r_o = {b['r_o']:.4f}\n")
            fh.write(f"  tasks n = {b['n']}   RT sum = {b['rt_sum']:.2f}\n\n")
            print(f"  {b['title'][:34]:34s} deg={b['deg']:5.1f} chi={b['chi']:.3f} "
                  f"r_o={b['r_o']:.3f} n={b['n']:2d} RT={b['rt_sum']:.1f}")

    print(f"wrote {RESULTS/'bundle_examples.png'}")
    print(f"wrote {RESULTS/'bundle_examples_summary.txt'}")


if __name__ == "__main__":
    main()
