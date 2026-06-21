"""
11_centroid_shift_test.py
-------------------------
STUB -- empirical consistency test for the displacement channel (paper
section Empirical Analysis). NOT YET RUNNABLE: it needs OEWS wage levels for
2019 AND 2024 frozen into data/ (the repo currently carries 2023 only). The
method below is the agreed design; fill in the data join and uncomment.

Design (explicitly non-causal -- a directional consistency check):
  - take the equilibrium centroid shift Delta mu_o per occupation from the
    re-sorting in scripts/09_equilibrium_regime.py (how each occupation's
    bundle centroid moves once displacement + reinstatement rewrite the bundle);
  - project Delta mu_o onto the price gradient grad Pi at the occupation's
    location: the displacement channel predicts occupations whose bundle
    slides DOWN-gradient (toward cheaper work) should see LOWER wages;
  - compare the projection to observed OEWS log-wage changes 2019 -> 2024 on
    the same ~741 occupations: correlation and sign agreement.

Optional second leg: a cross-sectional Bessen check -- correlate observed
sectoral employment change under mechanisation with an estimated demand
elasticity, as external plausibility for the eta channel.

Outputs (when implemented):
    results/centroid_shift_test.csv
    results/centroid_shift_test.png
    results/centroid_shift_test_summary.txt
"""
from __future__ import annotations

import sys


def main() -> None:
    sys.exit(
        "11_centroid_shift_test.py is a stub: freeze OEWS 2019 and 2024 into "
        "data/ (with provenance in data/MANIFEST.json), then implement the "
        "centroid-shift projection described in the module docstring."
    )


if __name__ == "__main__":
    main()
