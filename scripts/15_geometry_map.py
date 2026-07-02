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

EXAMPLES = {                      # title match -> (published xi deg, label dx, dy)
    "Lawyers": (5, 14, 4),
    "Physicians, Pathologists": (44, 10, 8),
    "Software Developers": (75, 4, 12),
    "Chemical Engineers": (109, -6, 12),
    "Machinists": (164, -16, 6),
    "Carpenters": (196, -16, -4),
    "Dishwashers": (239, -10, -12),
    "Waiters and Waitresses": (272, 2, -14),
    "Biological Science Teachers": (340, 12, -8),
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

    ax.scatter(d["x"], d["y"], s=9, c="0.62", alpha=0.55,
               edgecolors="none", zorder=2)

    for key, (pub_deg, dx, dy) in EXAMPLES.items():
        m = d[d["Title"].str.contains(key.split(",")[0], case=False,
                                      na=False)]
        m = m[(m["deg"] - pub_deg).abs() < 6]
        if m.empty:
            print(f"  [skip] no match for {key}")
            continue
        r = m.iloc[0]
        assert abs(r["deg"] - pub_deg) < 6, (key, r["deg"])
        ax.scatter(r["x"], r["y"], s=42, c="#1f3d7a", zorder=4)
        short = r["Title"].replace("Physicians, ", "").replace(
            " and Waitresses", "").replace(", Postsecondary", "")
        ax.annotate(short, (r["x"], r["y"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, zorder=5)
        print(f"  {r['Title'][:44]:44s} xi={r['deg']:5.1f} chi={r['chi']:.2f}")

    for ang, lab in POLES:
        a = np.radians(ang)
        ax.text(1.13 * np.cos(a), 1.13 * np.sin(a), lab, fontsize=11,
                ha="center", va="center", style="italic", color="0.25")

    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "geometry_map.png", dpi=170)
    plt.close(fig)
    print(f"wrote {RESULTS/'geometry_map.png'}")


if __name__ == "__main__":
    main()
