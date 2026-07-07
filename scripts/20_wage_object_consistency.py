"""
20_wage_object_consistency.py
-----------------------------
Consistency check across the paper's three occupation-level wage objects, and
against observed OEWS wage growth (the Section-8 data). Written and pre-
registered BEFORE the first run.

Motivation. Section 8 (Figure 12) confronts one model wage object -- the
predicted directional pressure proj_o = Delta mu_o . grad ln Pi, a first-order,
stripping-only, centroid-shift measure -- with observed 2019-2025 OEWS wage
growth. The paper carries two richer wage objects the figure does not use:
  bundle : Delta w_o = int Pi [-b a + iota/L], the exact bundle re-pricing at
           the frozen field (stripping + reinstatement, no congestion);
  value  : d ln W_o, the change in the occupation value workers sort on and are
           paid, decomposed exactly in script 19 into stripping + congestion +
           reinstatement.
The three form a chain of increasing richness (proj -> bundle -> value). This
script asks whether they are mutually consistent -- do they rank occupations the
same way -- and, the informative part, whether ADDING THE CONGESTION CHANNEL
(value vs bundle) moves the occupation ordering toward or away from the data.
Congestion is a distributional channel; if it re-ranks occupations against the
data, the stripping-only measures were closer to truth and congestion is an
internal redistribution; if it re-ranks with the data, congestion is part of the
observable wage story.

All three are projected to the same 725-occupation OEWS-matched sample used by
Figure 12, via scripts/11 (proj, oews) and the exact decomposition of scripts/19
(bundle, value components), so this cannot drift from the committed operator.

PRE-REGISTERED HYPOTHESES (before first run):
  K1  proj and bundle agree strongly: Spearman(proj, bundle) >= +0.8. (Both are
      stripping-driven; they should rank occupations alike.)
  K2  value is still positively aligned with the stripping measures but looser:
      +0.3 <= Spearman(value, bundle) <= +0.9. (Congestion re-ranks, but does
      not reverse, the stripping order.)
  K3  Sign consistency with data: all three carry the SAME SIGN of raw Spearman
      with observed OEWS wage growth as Figure 12's proj (positive, ~+0.34), so
      no object contradicts the paper's empirical direction.
  K4  Congestion's data effect is small in the raw cross-section: |Spearman(
      value, dlnw) - Spearman(bundle, dlnw)| < 0.10. (Consistent with Section 8:
      the 2019-2025 cross-section is confounded and cannot resolve channels; the
      congestion channel should not suddenly track the data, which would be
      suspicious given the pandemic confound.)
Adverse outcomes are reported. If K1 fails the objects are internally
inconsistent and the decomposition or the projection is suspect. If K3 fails a
model wage object contradicts the paper's own empirical direction and must be
reconciled before either result is reported.

Outputs:
    results/wage_object_consistency.csv
    results/wage_object_consistency_summary.txt

Usage:
    python scripts/20_wage_object_consistency.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.equilibrium import Equilibrium

def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).parent / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_setup = _load("_setup.py")
cst = _load("11_centroid_shift_test.py")

RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
NMIN = 1e-9
ORIGIN, ENDPOINT = 2019, 2025


def w_value(eq, content_wD, gcarrier, nb1):
    strip = np.bincount(eq.row_of, weights=content_wD * nb1[eq.cell_of],
                        minlength=eq.n_occ)
    reinst = eq.e @ (gcarrier * nb1)
    return BETA * (strip + reinst)


def value_components(eq, L0, Lstar):
    """Exact strip/cong/reinst adjustments and total d ln W_o (from script 19)."""
    n0 = (np.bincount(eq.cell_of, weights=L0[eq.row_of] * eq.b_w,
                      minlength=eq.area.size) / eq.area)
    M = eq.gamma * float(np.sum(Lstar * eq.D_o))
    s = M * eq.g_hat
    surv = 1.0 - eq.a_grid
    C = Lstar @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    npost = n0_star = (np.bincount(eq.cell_of, weights=Lstar[eq.row_of] * eq.b_w,
                                   minlength=eq.area.size) / eq.area) + s * surv * Phi
    with np.errstate(divide="ignore", invalid="ignore"):
        gcar = np.where(C > 0, s * surv * Phi / C, 0.0) * eq.D_grid * eq.pi_cell * eq.area
    nb1 = lambda n: np.maximum(n, NMIN) ** (BETA - 1.0)
    content_pre = eq.b_w * eq.pi_task
    content_strip = eq.strip_wD
    W0 = w_value(eq, content_pre, np.zeros(eq.area.size), nb1(n0))
    Ws = w_value(eq, content_strip, np.zeros(eq.area.size), nb1(n0))
    Wc = w_value(eq, content_strip, np.zeros(eq.area.size), nb1(npost))
    Wp = w_value(eq, content_strip, gcar, nb1(npost))
    dln = lambda a, b: np.where((a > 0) & (b > 0), np.log(a) - np.log(b), 0.0)
    return dln(Ws, W0), dln(Wc, Ws), dln(Wp, Wc), dln(Wp, W0)


def main() -> None:
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    _, _, W0chk = eq.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0chk, eq.d)
    out = eq.solve(c, kappa)

    codes = eq.codes
    # proj_o and bundle dW from script 11's centroid machinery
    mu_pre, mu_post, D_o, B_o = cst.post_centroids(inp, tech, L0, ell)
    dmu = mu_post - mu_pre
    xi_o, chi_o = occ["xi"].to_numpy(), occ["chi"].to_numpy()
    gr, ga = inp.field.grad_log_pi(xi_o, chi_o)
    gx = gr * np.cos(xi_o) - ga * np.sin(xi_o)
    gy = gr * np.sin(xi_o) + ga * np.cos(xi_o)
    proj = dmu[:, 0] * gx + dmu[:, 1] * gy

    # bundle dW_o at the frozen field (stripping + reinstatement, no congestion)
    from model.regime import regime
    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA, wedge=None,
                  survival=True)
    dW_bundle = diag["dW_bundle"]

    # value components (exact) at the solved equilibrium
    strip_adj, cong_adj, reinst_adj, dlnW = value_components(eq, L0, out.L)

    df = pd.DataFrame({
        "OCC_CODE": occ["OCC_CODE"].to_numpy(), "proj": proj, "bundle": dW_bundle,
        "value": dlnW, "strip": strip_adj, "cong": cong_adj, "reinst": reinst_adj,
    })
    # attach observed OEWS wage growth on the matched sample (Figure 12 sample)
    w0 = df["OCC_CODE"].map(cst.oews_median(ORIGIN))
    w1 = df["OCC_CODE"].map(cst.oews_median(ENDPOINT))
    df["dlnw"] = np.log(w1) - np.log(w0)
    m = df.dropna(subset=["dlnw"]).copy()

    def sp(a, b):
        return spearmanr(m[a], m[b])[0]

    # internal agreement
    s_proj_bundle = sp("proj", "bundle")
    s_value_bundle = sp("value", "bundle")
    s_value_proj = sp("value", "proj")
    # vs data
    d_proj = sp("proj", "dlnw")
    d_bundle = sp("bundle", "dlnw")
    d_value = sp("value", "dlnw")
    cong_data_shift = d_value - d_bundle

    K1 = s_proj_bundle >= 0.8
    K2 = 0.3 <= s_value_bundle <= 0.9
    K3 = (np.sign(d_proj) == np.sign(d_bundle) == np.sign(d_value)) and d_proj > 0
    K4 = abs(cong_data_shift) < 0.10

    lines = [
        "Wage-object consistency check (pre-registered; see docstring).",
        f"  N matched to OEWS {ORIGIN}-{ENDPOINT}: {len(m)} occupations "
        f"(Figure 12 sample)",
        "",
        "Internal agreement (Spearman):",
        f"  proj  vs bundle = {s_proj_bundle:+.3f}   (both stripping-driven)",
        f"  value vs bundle = {s_value_bundle:+.3f}   (adds congestion)",
        f"  value vs proj   = {s_value_proj:+.3f}",
        "",
        "Against observed OEWS wage growth (raw Spearman):",
        f"  proj   vs dlnw = {d_proj:+.3f}   (Figure 12's object; ~+0.34 committed)",
        f"  bundle vs dlnw = {d_bundle:+.3f}",
        f"  value  vs dlnw = {d_value:+.3f}",
        f"  congestion's data shift (value - bundle) = {cong_data_shift:+.3f}",
        "",
        "Pre-registered hypothesis verdicts:",
        f"  K1 (proj~bundle >= +0.8)                   {'PASS' if K1 else 'FAIL'}"
        f"  ({s_proj_bundle:+.3f})",
        f"  K2 (value~bundle in [+0.3,+0.9])           {'PASS' if K2 else 'FAIL'}"
        f"  ({s_value_bundle:+.3f})",
        f"  K3 (all same positive sign vs data)        {'PASS' if K3 else 'FAIL'}"
        f"  (proj {d_proj:+.2f}, bundle {d_bundle:+.2f}, value {d_value:+.2f})",
        f"  K4 (congestion barely moves data corr)     {'PASS' if K4 else 'FAIL'}"
        f"  (shift {cong_data_shift:+.3f})",
        "",
        "Reading: the three wage objects form a chain proj -> bundle -> value. "
        "proj and bundle are stripping-driven and should agree (K1); value adds "
        "the congestion channel and re-ranks occupations (K2) without reversing "
        "the stripping order or contradicting the paper's empirical direction "
        "(K3). Whether congestion moves the ordering toward the data (K4) speaks "
        "to Section 8: in this confounded window it should not, and a large shift "
        "would be suspect rather than confirmatory.",
    ]

    m.to_csv(RESULTS / "wage_object_consistency.csv", index=False)
    (RESULTS / "wage_object_consistency_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS/'wage_object_consistency.csv'} and _summary.txt")


if __name__ == "__main__":
    main()
