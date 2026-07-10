"""
15_geometry_map.py
------------------
The compass map for the geometry section: all occupational centroids on
the polar disk, the four directional poles of storck2026geometry
labelled at the rim (analytical north, human-centered east, service
south, technical-physical west), and the example occupations of that
paper highlighted. Coordinates come from the frozen geometry, so the
examples must match the published values (asserted below).

Output:
    results/geometry_map.png

Usage:
    python scripts/15_geometry_map.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

# One dial scales every text element of this figure.
FONT_SCALE = 1.5
FS_LABEL = 9 * FONT_SCALE          # occupation labels
FS_POLE = 11 * FONT_SCALE          # directional pole labels

# Marker sizes and the background task cloud.
S_TASK = 2                         # background task points
S_OTHER = 16                       # non-example occupation centroids
S_EXAMPLE = 140                    # example occupations (stars, cf. script 24)
TASK_COLOR = "0.80"                # background task cloud
OTHER_COLOR = "steelblue"               # non-example occupation centroids
EXAMPLE_COLOR = "#c1440e"          # example marker fill
EXAMPLE_EDGE = "black"             # example marker outline

EXAMPLES = {                      # title match -> (published xi deg, label dx, dy)
    "Lawyers": (5, 9, 9),
    #"Physicians, Pathologists": (44, 10, 8),
    "Software Developers": (75, 4, 12),
    #"Chemical Engineers": (109, -120, 9),
    "Machinists": (164, -6, 9),
    #"Carpenters": (196, -66, -18),
    "Dishwashers": (239, -90, -18),
    "Waiters and Waitresses": (272, 4, -18),
    #"Biological Science Teachers": (340, -42, -18),
}
POLES = [(90, "analytical"), (0, "human-centered"),
         (270, "service"), (180, "technical-physical")]


def main() -> None:
    d = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv")
    d["deg"] = np.degrees(d["xi"]) % 360
    d["x"] = d["chi"] * np.cos(d["xi"])
    d["y"] = d["chi"] * np.sin(d["xi"])

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.75", lw=0.9))
    for ang in range(0, 360, 45):
        a = np.radians(ang)
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="0.9", lw=0.6, zorder=0)

    # all tasks as a light background cloud
    tasks = pd.read_csv(DATA / "task_embeddings_polar_scaled.csv")
    tx = tasks["chi"] * np.cos(tasks["xi"])
    ty = tasks["chi"] * np.sin(tasks["xi"])
    ax.scatter(tx, ty, s=S_TASK, c=TASK_COLOR, alpha=0.35,
               edgecolors="none", zorder=1)

    # locate the example occupations so the grey layer can exclude them
    examples = {}
    for key, (pub_deg, dx, dy) in EXAMPLES.items():
        m = d[d["Title"].str.contains(key.split(",")[0], case=False,
                                      na=False)]
        m = m[(m["deg"] - pub_deg).abs() < 6]
        if m.empty:
            print(f"  [skip] no match for {key}")
            continue
        r = m.iloc[0]
        assert abs(r["deg"] - pub_deg) < 6, (key, r["deg"])
        examples[r.name] = (r, dx, dy)

    # every other occupation as a grey centroid
    others = d.drop(index=list(examples.keys()))
    ax.scatter(others["x"], others["y"], s=S_OTHER, c=OTHER_COLOR, alpha=0.6,
               edgecolors="none", zorder=2)

    # the example occupations as stars, in line with script 24
    for r, dx, dy in examples.values():
        ax.scatter(r["x"], r["y"], s=S_EXAMPLE, marker="*",
                   facecolor=EXAMPLE_COLOR, edgecolors=EXAMPLE_EDGE,
                   linewidths=0.9, zorder=4)
        short = r["Title"].replace("Physicians, ", "").replace(
            " and Waitresses", "").replace(", Postsecondary", "")
        ax.annotate(short, (r["x"], r["y"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=FS_LABEL, zorder=5)
        print(f"  {r['Title'][:44]:44s} xi={r['deg']:5.1f} chi={r['chi']:.2f}")

    for ang, lab in POLES:
        a = np.radians(ang)
        ax.text(1.13 * np.cos(a), 1.13 * np.sin(a), lab, fontsize=FS_POLE,
                ha="center", va="center", style="italic", color="0.25")

    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "geometry_map.png", dpi=170)
    plt.close(fig)
    print(f"wrote {RESULTS/'geometry_map.png'}")


if __name__ == "__main__":
    main()
