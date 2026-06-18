"""
model.capability_field
----------------------
The capability requirement fields q_k(r), eq. (capability-field) of the
paper: one field per capability cluster k in {S1, S2, A1, A2}, of the
same functional family as the price field,

    q_k(xi, chi) = a_{k,0} + a_{k,1} cos xi + a_{k,2} sin xi
                   + chi (a_{k,3} + a_{k,4} cos xi + a_{k,5} sin xi)

Coefficients are estimated once on the occupation cross-section
(scripts/06_capability_fields.py) from the frozen cluster intensities
and treated as fixed structure, like the price field. The same script
measures the cluster weights v_k (the cluster wage returns of the
Paper 1 mediation regression, replicated on the frozen inputs); both
are loaded from results/capability_field_coefficients.csv.

Two specs are exported by scripts/06: Q_plane (theory, the exact
plane form, a1 = a2 = a3 = 0) and Q1_field (measured, all six). The
deficit gate of the paper, eq. (deficit), reads

    delta_o(r) = sum_{k priced} v_k max(q_k(r) - q_{o,k}, 0),

summed over the priced clusters {S1, S2}.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COEF_FILE = REPO_ROOT / "results" / "capability_field_coefficients.csv"

CLUSTERS = ["S1", "S2", "A1", "A2"]
# The deficit gate is defined over the PRICED clusters: unpriced
# capabilities lie in the kernel of the price functional and cannot
# gate, and the sign restriction v_k >= 0 excludes noise estimates
# (decision 2026-06-12; see scripts/06 and Sec. 3 of the paper).
PRICED = ("S1", "S2")
PARAMS = ["a0", "a1", "a2", "a3", "a4", "a5"]


@dataclass(frozen=True)
class CapabilityField:
    """Container for the four q_k fields and the cluster weights v_k."""

    alpha: dict        # cluster -> np.ndarray of 6 coefficients
    v: dict            # cluster -> wage-return weight v_k

    # ── construction ──────────────────────────────────────────────

    @classmethod
    def from_results(cls, path: Path = COEF_FILE,
                     spec: str = "Q_plane",
                     v_spec: str = "V_mediation") -> "CapabilityField":
        df = pd.read_csv(path)
        alpha: dict = {}
        for k in CLUSTERS:
            s = (df.loc[(df["spec"] == spec) & (df["cluster"] == k)]
                 .set_index("param")["coef"])
            missing = [p for p in PARAMS if p not in s.index]
            if missing:
                raise ValueError(f"coefficients {missing} not found for "
                                 f"cluster {k} (spec '{spec}') in {path}")
            alpha[k] = np.array([float(s[p]) for p in PARAMS])
        vrows = df.loc[(df["spec"] == v_spec)
                       & (df["param"] == "v_k")].set_index("cluster")["coef"]
        missing = [k for k in CLUSTERS if k not in vrows.index]
        if missing:
            raise ValueError(f"v_k weights {missing} not found "
                             f"(spec '{v_spec}') in {path}")
        v = {k: float(vrows[k]) for k in CLUSTERS}
        return cls(alpha=alpha, v=v)

    # ── evaluation ────────────────────────────────────────────────

    def q(self, cluster: str, xi, chi):
        """Requirement level q_k(xi, chi) for one cluster."""
        a = self.alpha[cluster]
        xi = np.asarray(xi, dtype=float)
        chi = np.asarray(chi, dtype=float)
        return (a[0] + a[1] * np.cos(xi) + a[2] * np.sin(xi)
                + chi * (a[3] + a[4] * np.cos(xi) + a[5] * np.sin(xi)))

    def q_all(self, xi, chi) -> dict:
        """All four requirement levels at (xi, chi)."""
        return {k: self.q(k, xi, chi) for k in CLUSTERS}

    # ── the deficit gate ──────────────────────────────────────────

    @property
    def v_gate(self) -> dict:
        """Gate weights: the priced clusters only (theory variant)."""
        return {k: self.v[k] for k in PRICED}

    def deficit(self, xi, chi, q_o: dict, priced_only: bool = True):
        """delta_o(r) = sum_k v_k max(q_k(r) - q_{o,k}, 0), summed over
        the priced clusters (theory). `priced_only=False` is the
        all-cluster sensitivity variant. `q_o` maps cluster -> the
        occupation's own level q_{o,k} (scalar). Broadcasts over
        xi/chi arrays."""
        xi = np.asarray(xi, dtype=float)
        chi = np.asarray(chi, dtype=float)
        keys = PRICED if priced_only else tuple(CLUSTERS)
        d = np.zeros(np.broadcast(xi, chi).shape, dtype=float)
        for k in keys:
            d += self.v[k] * np.maximum(self.q(k, xi, chi) - q_o[k], 0.0)
        return d
