"""
model.technology
----------------
A technology K and the operated regime, eqs. (phi-K), (regime-shift),
(soft-switch), (displaced) of the paper.

    phi_K(r) = A_K exp(-1/2 (|r - p_K| / z_K)^2)

    capital bears the work where  s_K phi_K(r) > R / Pi(r)

    a(r) = sigmoid( (s_K phi_K(r) - R / Pi(r)) / tau )

    D_o  = sum_t b_{o,t} a(r_t)            (per-occupation displaced mass)

The four technology attributes are position p_K, reach z_K, amplitude A_K,
and character s_K in [0, 1] (1 = purely replacing, 0 = purely augmenting).
The economy-side parameters are the capital rental R and the within-cell
margin width tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .price_field import PriceField


@dataclass(frozen=True)
class Technology:
    xi_K: float          # angular position of the center
    chi_K: float         # radial position of the center
    z_K: float           # reach
    A_K: float = 1.0     # amplitude (effectiveness at the center)
    s_K: float = 1.0     # character: 1 replacing, 0 augmenting

    @property
    def p_K(self) -> tuple[float, float]:
        """Center in Cartesian coordinates."""
        return (self.chi_K * np.cos(self.xi_K),
                self.chi_K * np.sin(self.xi_K))

    # ── fields over the disk ──────────────────────────────────────

    def _dist(self, xi, chi):
        x = np.asarray(chi, dtype=float) * np.cos(np.asarray(xi, dtype=float))
        y = np.asarray(chi, dtype=float) * np.sin(np.asarray(xi, dtype=float))
        px, py = self.p_K
        return np.hypot(x - px, y - py)

    def phi(self, xi, chi):
        """Effectiveness field phi_K(r)."""
        d = self._dist(xi, chi)
        return self.A_K * np.exp(-0.5 * (d / self.z_K) ** 2)

    def grad_phi_norm(self, xi, chi):
        """|grad phi_K|(r) = (|r - p_K| / z_K^2) phi_K(r); peaks on the
        ring at distance z_K from the center (the seeding locus)."""
        d = self._dist(xi, chi)
        return d / self.z_K ** 2 * self.phi(xi, chi)

    # ── operated regime ───────────────────────────────────────────

    def operated_share(self, xi, chi, field: PriceField,
                       R: float, tau: float, log_wedge=0.0):
        """a(r): share of the location's micro-tasks borne by capital.

        `log_wedge` (scalar or array broadcastable over xi/chi) shifts
        the effective price of the labor performing the work,
        Pi_eff = exp(log_wedge) * Pi(r): a positive wedge makes the same
        location dearer and lowers the takeover threshold. The baseline
        model is wedge-free (log_wedge = 0)."""
        pi_eff = np.exp(log_wedge) * field.pi(xi, chi)
        margin = self.s_K * self.phi(xi, chi) - R / pi_eff
        return 1.0 / (1.0 + np.exp(-margin / tau))

    def displacement(self, bundles: pd.DataFrame, field: PriceField,
                     R: float, tau: float,
                     wedge: pd.Series | None = None) -> pd.DataFrame:
        """Per-occupation displaced mass D_o = sum_t b_t a(r_t) and the
        retained priced contribution sum_t b_t Pi_eff(r_t)(1 - a(r_t)).

        `bundles` as produced by model.data.load_bundles. `wedge` is an
        optional log wage wedge eta_o indexed by onet_code: it raises
        the effective price of the occupation's labor at every task it
        performs, which both shifts the takeover margin (dear work is
        taken first) and scales the priced contribution. Baseline runs
        pass wedge=None."""
        xi = bundles["xi"].to_numpy()
        chi = bundles["chi"].to_numpy()
        b = bundles["b"].to_numpy()
        if wedge is None:
            lw = 0.0
        else:
            lw = (wedge.reindex(bundles["onet_code"]).fillna(0.0)
                  .to_numpy())
        a = self.operated_share(xi, chi, field, R, tau, log_wedge=lw)
        pi_t = np.exp(lw) * field.pi(xi, chi)
        idx = bundles["onet_code"].to_numpy()
        out = pd.DataFrame({
            "D_o": pd.Series(b * a, index=idx).groupby(level=0).sum(),
            "w_retained": pd.Series(b * pi_t * (1 - a), index=idx)
                            .groupby(level=0).sum(),
            "w_pre": pd.Series(b * pi_t, index=idx).groupby(level=0).sum(),
        })
        out.index.name = "onet_code"
        return out
