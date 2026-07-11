"""
d00_zero_field_anchor.py
------------------------
The anchoring guard: the dynamic mirror of the static anchoring requirement.
The static manuscript sets the occupation constants alpha_o so that observed
employment L0 is the fixed point of the zero-technology sorting map; without
them "a solved equilibrium mixes the technology's effect with a baseline
relocation that has nothing to do with the technology." The dynamic layer
consumes the same anchored reference (kappa, c, alpha) through
_interface.load_static_layer, and this script asserts that the guarantee
actually holds where the dynamics run: in the kernel, and in the engine.

Three checks, one per layer of the claim:

  (K) Kernel identity. One step of the anchored zero-field sorting map from
      L0 returns L0 to numerical precision: Equilibrium.resort evaluated at
      the zero-field value W^0 (the anchoring reference itself) satisfies
      |L0 P - L0|_1 below 1e-9. This is Sinkhorn's balancing condition read
      back, and it certifies that the interface carries the same alpha the
      static pipeline certifies.

  (G) Engine guard. A zero-shock run of the full dynamic engine
      (run_dynamic.main with A_scale = 0, T_max = 20 y, reference tempos,
      anchored kernel) leaves the population at the baseline up to the gate
      tail: |L(20y) - L0|_1 below 1e-2. The residual is not zero, and its
      source is stated: the anchor is exact at a = 0, while the engine's
      adoption sigmoid keeps a(r) > 0 at A_K = 0 (the tail
      a = 1/(1 + exp((R/Pi)/tau)), largest where work is dear), so the
      engine's zero-technology value differs from the anchoring reference by
      the stripped tail. The run also asserts what zero technology must
      imply structurally: no seeding (U stays identically zero), no births,
      and population conserved.

  (C) Contrast. The same zero-shock run under the unanchored pre-revision
      kernel (anchored = False) relocates a large share of employment. The
      contrast documents, as a number in the summary, what the anchoring
      removes from every shocked run.

Pre-registered hypotheses (written before the certified run; smoke values
from the development machine in parentheses):
  (H1) kernel one-step gap below 1e-9 (smoke: 3.2e-14).
  (H2) anchored zero-shock drift below 1e-2 in L1 over 20 years
       (smoke: 6.3e-3, i.e. 0.31 percent of mass relocated).
  (H3) the unanchored drift exceeds the anchored drift by more than a
       factor of 50 (smoke: 1.16 against 6.3e-3, a factor of ~186).

The bounds are guard semantics, not frozen point values: the certified run
should freeze the measured drift as an exact baseline with the usual
tolerance, alongside these bounds.

Usage: python experiment/d00_zero_field_anchor.py   (about 2 minutes)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

KERNEL_TOL = 1e-9      # (H1)
DRIFT_TOL = 1e-2       # (H2), L1 over the 20-year zero-shock window
CONTRAST_MIN = 50.0    # (H3), unanchored / anchored drift ratio

# Frozen point baselines from the certified recalibration pass (anchored
# kernel with the values() density floor), 5 percent relative tolerance.
# The bounds above are the guard semantics; these freeze the realisation.
DRIFT_A_BASE = 0.00488
DRIFT_U_BASE = 1.1649


def main():
    lines = []

    def emit(s=""):
        lines.append(s)

    layer = iface.load_static_layer()
    eq, L0 = layer.eq, layer.L0

    emit("d00: zero-field anchoring guard (the dynamic mirror of the static "
         "anchoring requirement)")
    emit(f"anchored mobility reference kappa {layer.kappa:.3f}, c {layer.c:.3f}; "
         f"alpha in [{layer.alpha.min():+.2f}, {layer.alpha.max():+.2f}]")
    emit("")

    # ---- (K) kernel identity: one step of the anchored zero-field map ----
    Wz = eq.zero_field_value(L0)
    T1 = eq.resort(Wz, layer.c, layer.kappa)
    gap = float(np.abs(T1 - L0).sum())
    emit(f"(K) kernel one-step gap |L0 P - L0|_1 = {gap:.3e}  (tol {KERNEL_TOL:g})")
    assert gap < KERNEL_TOL, f"anchoring broken in the kernel: gap {gap:.3e}"

    # ---- (G) engine guard: zero shock, anchored ----
    dyn, rec, _ = rd.main(A_scale=0.0, T_max=20.0, verbose=False, layer=layer,
                          anchored=True)
    drift_a = float(np.abs(dyn.L - L0).sum())
    emit(f"(G) anchored zero-shock run, 20 y at reference tempos:")
    emit(f"      drift |L - L0|_1 = {drift_a:.5f}  "
         f"({100 * drift_a / 2:.3f} percent of mass relocated; tol {DRIFT_TOL:g})")
    emit(f"      residual source: the adoption sigmoid keeps a > 0 at A_K = 0, "
         f"while the anchor is exact at a = 0")
    assert drift_a < DRIFT_TOL, f"anchored engine drifts: {drift_a:.4f}"
    assert abs(drift_a - DRIFT_A_BASE) <= 0.05 * DRIFT_A_BASE, \
        f"anchored drift baseline drifted: {drift_a:.5f} vs frozen {DRIFT_A_BASE}"
    assert float(np.max(rec["U_tot"])) == 0.0, "seeding at zero technology"
    assert dyn.n_occ == eq.n_occ, "birth at zero technology"
    assert abs(rec["Lsum"][0] - rec["Lsum"][-1]) < 1e-9, "population not conserved"

    # ---- (C) contrast: the unanchored pre-revision kernel ----
    dyn0, rec0, _ = rd.main(A_scale=0.0, T_max=20.0, verbose=False, layer=layer,
                            anchored=False)
    drift_u = float(np.abs(dyn0.L - L0).sum())
    ratio = drift_u / max(drift_a, 1e-12)
    emit(f"(C) unanchored contrast (pre-revision kernel), same run:")
    emit(f"      drift |L - L0|_1 = {drift_u:.4f}  "
         f"({100 * drift_u / 2:.1f} percent of mass relocated)")
    emit(f"      unanchored / anchored drift ratio = {ratio:.0f}  "
         f"(min {CONTRAST_MIN:g})")
    emit("      this baseline relocation, unrelated to any technology, is what")
    emit("      the anchoring removes from every shocked run.")
    assert ratio > CONTRAST_MIN, f"contrast ratio too small: {ratio:.1f}"
    assert abs(drift_u - DRIFT_U_BASE) <= 0.05 * DRIFT_U_BASE, \
        f"unanchored drift baseline drifted: {drift_u:.4f} vs frozen {DRIFT_U_BASE}"
    # The unanchored drift even manufactures seeding at zero technology:
    # the population drifts toward occupations with a larger gate-tail
    # displacement D_o, GammaD rises, and the positive part of its rate
    # seeds. Reported, not asserted away -- it is one more thing the
    # anchoring removes.
    u_spur = float(np.max(rec0["U_tot"]))
    if u_spur > 0.0:
        emit(f"      spurious zero-technology seeding under the unanchored "
             f"kernel: peak U_tot = {u_spur:.2e}")

    emit("")
    emit("all anchoring guards hold: L0 is a rest point of the zero-field")
    emit("sorting map in the kernel and, up to the gate tail, in the engine.")
    iface.write_summary("zero_field_anchor", lines)


if __name__ == "__main__":
    main()
