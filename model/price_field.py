"""
model.price_field
-----------------
The price of skill Pi(r), eq. (1) of the paper:

    ln Pi(xi, chi) = m0 + m1 cos xi + m2 sin xi
                     + chi (m3 + m4 cos xi + m5 sin xi)

Coefficients are estimated once on the Paper 1 occupation cross-section
(scripts/01_wage_field.py, spec S1_field) and treated as fixed structure.
This module loads them from results/wage_field_coefficients.csv and
provides evaluation, the gradient, and bundle pricing:

    w_o = sum_t b_{o,t} Pi(r_t)        (eq. wage-bundle, empirical form)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COEF_FILE = REPO_ROOT / "results" / "wage_field_coefficients.csv"

PARAMS = ["m0", "m1", "m2", "m3", "m4", "m5"]


@dataclass(frozen=True)
class PriceField:
    m0: float
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float

    # ── construction ──────────────────────────────────────────────

    @classmethod
    def from_results(cls, path: Path = COEF_FILE,
                     spec: str = "S1_field") -> "PriceField":
        df = pd.read_csv(path)
        s = df.loc[df["spec"] == spec].set_index("param")["coef"]
        missing = [p for p in PARAMS if p not in s.index]
        if missing:
            raise ValueError(f"coefficients {missing} not found in {path} "
                             f"(spec '{spec}')")
        return cls(*(float(s[p]) for p in PARAMS))

    # ── evaluation ────────────────────────────────────────────────

    def log_pi(self, xi, chi):
        xi = np.asarray(xi, dtype=float)
        chi = np.asarray(chi, dtype=float)
        return (self.m0
                + self.m1 * np.cos(xi) + self.m2 * np.sin(xi)
                + chi * (self.m3
                         + self.m4 * np.cos(xi) + self.m5 * np.sin(xi)))

    def pi(self, xi, chi):
        return np.exp(self.log_pi(xi, chi))

    def pi_cart(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.pi(np.arctan2(y, x), np.hypot(x, y))

    def beta_chi(self, xi):
        """Directional return to radial depth, beta_chi(xi) =
        m3 + m4 cos xi + m5 sin xi."""
        xi = np.asarray(xi, dtype=float)
        return self.m3 + self.m4 * np.cos(xi) + self.m5 * np.sin(xi)

    # ── gradient (polar, physical components) ─────────────────────

    def grad_log_pi(self, xi, chi):
        """Gradient of ln Pi in physical polar components:
        (d/dchi, (1/chi) d/dxi). The angular component is the physical
        one, so it diverges toward the origin by construction."""
        xi = np.asarray(xi, dtype=float)
        chi = np.asarray(chi, dtype=float)
        d_chi = self.beta_chi(xi)
        d_xi = (-self.m1 * np.sin(xi) + self.m2 * np.cos(xi)
                + chi * (-self.m4 * np.sin(xi) + self.m5 * np.cos(xi)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.where(chi > 0, d_xi / chi, np.nan)
        return d_chi, ang

    # ── derived summaries ─────────────────────────────────────────

    def depth_return_direction(self) -> tuple[float, float]:
        """(direction deg, amplitude) of the depth return harmonic."""
        return (float(np.degrees(np.arctan2(self.m5, self.m4)) % 360),
                float(np.hypot(self.m4, self.m5)))

    # ── bundle pricing ────────────────────────────────────────────

    def bundle_wage(self, bundles: pd.DataFrame,
                    wedge: pd.Series | None = None) -> pd.Series:
        """w_o = exp(eta_o) sum_t b_t Pi(r_t) per occupation.

        `bundles` as produced by model.data.load_bundles (columns
        onet_code, xi, chi, b). `wedge` is an optional log wage wedge
        eta_o indexed by onet_code (the measured non-spatial component;
        see scripts/05_family_wedge.py). Occupations without a wedge
        entry get eta = 0. The theory layer is wedge-free; pass wedge
        only in measurement/sensitivity runs."""
        pi_t = self.pi(bundles["xi"].to_numpy(), bundles["chi"].to_numpy())
        contrib = bundles["b"].to_numpy() * pi_t
        w = (pd.Series(contrib, index=bundles["onet_code"].to_numpy())
             .groupby(level=0).sum().rename("w_bundle"))
        if wedge is not None:
            w = w * np.exp(wedge.reindex(w.index).fillna(0.0))
        return w
