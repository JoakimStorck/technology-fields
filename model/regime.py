"""
model.regime
------------
The downstream half of the model: reinstatement, the bundle operator, unbound
tasks, the two-factor task-work density, occupation value, and the labor share
(Secs. 3.5, 3.7, 3.8 of the paper). The takeover/displacement primitives live
in model.technology; the deficit gate in model.capability_field; the price
field in model.price_field. This module assembles them into the post-technology
comparative static at a given employment vector L.

Everything here takes employment L as an input and is therefore independent of
the worker-layer re-sorting kernel (eq. equilibrium). The re-sorting fixed point
that updates L is built separately, once its specification is settled; this
module supplies the value W_o it would iterate on.

The continuous fields (phi_K, Pi, q_k, the gradient ring, the seeded /
reinstated / unbound densities) are evaluated on a polar grid of the disk. The
bundles b_o are discrete measures on the occupation's own task positions, so
displacement and the strip term are computed on the tasks, while the refill,
capacity, and unbound mass are computed on the grid and integrated with the
polar area element.

Conventions enforced (and checked in scripts/11):
  - (1 - a) enters exactly once, inside the human part h_o.
  - the bundle operator is NOT renormalized; b_o^post integrates to
    1 - D_o + B_o/L_o.
  - bound mass sum_o B_o plus unbound mass integrate to gamma * Delta_Gamma_D.
  - three task-work measures are tracked: h_o (human), k_o (capital),
    h_o + k_o (total), with the operated share entering n only through k_o.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .capability_field import CapabilityField
from .price_field import PriceField
from .technology import Technology


# ─────────────────────────────────────────────────────────────────────
# Disk grid
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiskGrid:
    """Polar grid over the unit disk with cell centers and polar areas.
    The area element is chi d_chi d_xi, so a cell at radius chi_c with
    widths d_chi, d_xi has area chi_c * d_chi * d_xi."""
    xi: np.ndarray      # (n_cells,) angular centers
    chi: np.ndarray     # (n_cells,) radial centers
    area: np.ndarray    # (n_cells,) polar cell areas
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def build(cls, n_ang: int = 180, n_rad: int = 90) -> "DiskGrid":
        d_xi = 2 * np.pi / n_ang
        d_chi = 1.0 / n_rad
        ang = (np.arange(n_ang) + 0.5) * d_xi
        rad = (np.arange(n_rad) + 0.5) * d_chi
        XI, CHI = np.meshgrid(ang, rad, indexing="ij")
        xi = XI.ravel(); chi = CHI.ravel()
        area = chi * d_chi * d_xi
        return cls(xi=xi, chi=chi, area=area,
                   x=chi * np.cos(xi), y=chi * np.sin(xi))


# ─────────────────────────────────────────────────────────────────────
# The regime computation at a given employment vector
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RegimeInputs:
    """Frozen ingredients shared across regime evaluations."""
    bundles: pd.DataFrame          # onet_code, xi, chi, b (per task)
    occ: pd.DataFrame              # indexed by onet_code: xi, chi (centroid), q_{o,k} columns
    field: PriceField
    cap: CapabilityField
    grid: DiskGrid

    def occ_codes(self) -> np.ndarray:
        return self.occ.index.to_numpy()


def _ring_density(tech: Technology, grid: DiskGrid) -> np.ndarray:
    """Normalized gradient-ring density g_hat(r) = |grad phi_K| / int|grad phi_K|,
    peaking on the ring at distance z_K from p_K (the seeding locus)."""
    g = tech.grad_phi_norm(grid.xi, grid.chi)
    Z = np.sum(g * grid.area)
    return g / Z if Z > 0 else g


def _readiness(inp: RegimeInputs, ell: float, priced_only: bool = True):
    """e_o(r) = exp(-delta_o(r)/ell) on the grid, for every occupation.
    Returns an (n_occ, n_cells) array. delta uses the priced-cluster gate."""
    cap, grid = inp.cap, inp.grid
    keys = cap.v_gate.keys() if priced_only else cap.alpha.keys()
    q_grid = {k: cap.q(k, grid.xi, grid.chi) for k in keys}     # (n_cells,)
    codes = inp.occ_codes()
    delta = np.zeros((len(codes), grid.xi.size))
    for k in keys:
        q_o = inp.occ[k].to_numpy()[:, None]                     # (n_occ,1)
        delta += cap.v[k] * np.maximum(q_grid[k][None, :] - q_o, 0.0)
    return np.exp(-delta / ell)


NMIN = 0.01  # density floor: caps the marginal value beta*Pi*n^(beta-1) at empty
             # cells so empty regions are not infinitely valuable (the dynamic
             # layer's A10 fix, brought into the static comparative static).


def _fit(inp: RegimeInputs, ell: float, rho: float = 0.5, lam_over: float = 1.0,
         priced_only: bool = True):
    """Attachment primitive shared with the dynamic layer:
    FIT_o(r) = E_match_o(r) * exp(-d(r, mu_o)/rho).
    E_match is the SYMMETRIC capability match exp(-(under + lam_over*over)/ell) --
    penalising over-qualification as well as under-qualification, so an
    over-qualified generalist far from its core no longer attaches as if it were a
    good home -- and the locality factor ties reinstatement to geometric proximity
    to the occupation's centroid mu_o. lam_over = 0 and rho -> inf recover the
    one-sided, non-local _readiness.
    """
    cap, grid = inp.cap, inp.grid
    keys = cap.v_gate.keys() if priced_only else cap.alpha.keys()
    codes = inp.occ_codes()
    under = np.zeros((len(codes), grid.xi.size))
    over = np.zeros_like(under)
    for k in keys:
        q_o = inp.occ[k].to_numpy()[:, None]
        q_r = cap.q(k, grid.xi, grid.chi)[None, :]
        under += cap.v[k] * np.maximum(q_r - q_o, 0.0)
        over += cap.v[k] * np.maximum(q_o - q_r, 0.0)
    E = np.exp(-(under + lam_over * over) / ell)
    gx = grid.chi * np.cos(grid.xi)
    gy = grid.chi * np.sin(grid.xi)
    mux = inp.occ["chi"].to_numpy() * np.cos(inp.occ["xi"].to_numpy())
    muy = inp.occ["chi"].to_numpy() * np.sin(inp.occ["xi"].to_numpy())
    d = np.sqrt((gx[None, :] - mux[:, None]) ** 2 + (gy[None, :] - muy[:, None]) ** 2)
    return E * np.exp(-d / rho)


def regime(inp: RegimeInputs, tech: Technology, L: np.ndarray,
           R: float, tau: float, gamma: float, ell: float, beta: float,
           wedge: np.ndarray | None = None,
           eta: float = 1.0, survival: bool = False,
           rho: float = 0.5, lam_over: float = 1.0) -> dict:
    """Post-technology comparative static at employment L (aligned to
    inp.occ_codes()). Returns displacement, reinstatement, the bundle-operator
    wage change, the labor share, occupation value W_o, and densities, with the
    mass-accounting terms needed to verify the operator.

    eta and survival mirror model.equilibrium: the demand multiplier
    D(r) = (c(r)/Pi)^(1-eta) scales the value objects (W_o, labor share) but
    not the field-priced bundle wage change dW_bundle, which is the gross
    re-pricing of the bundle; the survival gate (1 - a(r)) selects which
    reinstatement seeds survive as human work. eta = 1, survival = False
    reproduces the cost-invariant, un-gated regime."""
    grid, field = inp.grid, inp.field
    codes = inp.occ_codes()
    L = np.asarray(L, float)
    lw = np.zeros(len(codes)) if wedge is None else np.asarray(wedge, float)

    # ── displacement on the discrete bundles: D_o = sum_t b_t a(r_t) ──
    bx = inp.bundles
    b_xi, b_chi, b_w = (bx["xi"].to_numpy(), bx["chi"].to_numpy(),
                        bx["b"].to_numpy())
    code_idx = pd.Index(codes)
    row_of = code_idx.get_indexer(bx["onet_code"].to_numpy())
    lw_task = lw[row_of]
    a_task = tech.operated_share(b_xi, b_chi, field, R, tau, log_wedge=lw_task)
    pi_task = field.pi(b_xi, b_chi)        # field price; the wedge enters only a
    D_o = np.bincount(row_of, weights=b_w * a_task, minlength=len(codes))
    # strip term of the wage change: - int Pi b_o a (valued at the field price)
    strip = np.bincount(row_of, weights=b_w * pi_task * a_task,
                        minlength=len(codes))

    # field-level operated share a(r) on the grid (no occupation wedge): the
    # survival gate (1 - a) and the demand multiplier D both read it.
    a_grid = tech.operated_share(grid.xi, grid.chi, field, R, tau)
    pi_cell = field.pi(grid.xi, grid.chi)
    # demand multiplier D = (c/Pi)^(1-eta), c/Pi = (1-a) + a R/(s_K phi_K Pi);
    # <= 1 where capital is adopted, = 1 where it is not (and where eta = 1).
    phi_grid = tech.phi(grid.xi, grid.chi)
    phi_tk = tech.phi(b_xi, b_chi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr_grid = np.where(phi_grid > 1e-9,
                           R / (tech.s_K * phi_grid * pi_cell), 1.0)
        cr_task = np.where(phi_tk > 1e-9,
                           R / (tech.s_K * phi_tk * pi_task), 1.0)
    D_grid = np.clip((1.0 - a_grid) + a_grid * cr_grid, 1e-9, 1.0) ** (1.0 - eta)
    D_task = np.clip((1.0 - a_task) + a_task * cr_task, 1e-9, 1.0) ** (1.0 - eta)
    surv = (1.0 - a_grid) if survival else np.ones_like(a_grid)  # price survival

    Delta_Gamma_D = float(np.sum(L * D_o))
    M = gamma * Delta_Gamma_D                                   # seeded mass

    # ── reinstatement on the grid ──
    g_hat = _ring_density(tech, grid)
    s = M * g_hat                                              # seeded density
    e = _fit(inp, ell, rho, lam_over)                        # (n_occ, n_cells)
    C = (L[:, None] * e).sum(axis=0)                          # capacity (n_cells)
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(C > 0, (L[:, None] * e) / C[None, :], 0.0)
    iota = (s * surv)[None, :] * Phi[None, :] * share         # (n_occ, n_cells)
    B_o = (iota * grid.area[None, :]).sum(axis=1)             # bound mass per occ
    u = s * surv * (1.0 - Phi)                               # unbound human density
    bound_mass = float((iota * grid.area[None, :]).sum())
    unbound_mass = float(np.sum(u * grid.area))

    # ── bundle-operator wage change: Delta w = -strip + (1/L) int Pi iota ──
    pi_grid = field.pi(grid.xi, grid.chi)[None, :]    # field price (wedge only in a)
    refill = (pi_grid * iota * grid.area[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dW_bundle = np.where(L > 0, -strip + refill / L, -strip)

    # ── two-factor densities on the grid ──
    # pre-technology labor density n0(r) = sum_o L_o b_o(r), binned to cells
    cell_of = _cell_index(grid, b_xi, b_chi)
    n0 = np.bincount(cell_of, weights=L[row_of] * b_w,
                     minlength=grid.xi.size) / grid.area
    iota_tot = iota.sum(axis=0)                               # sum_o iota_o(r)
    n = n0 + iota_tot                                         # eq. taskwork-density
    # human task-work H(r) = sum_o L_o h_o = sum_o L_o b_o (1-a) + sum_o iota
    # strip on the grid via binned (L b a), with a evaluated per task for fidelity
    La_binned = np.bincount(cell_of, weights=L[row_of] * b_w * a_task,
                            minlength=grid.xi.size) / grid.area
    H = (n0 - La_binned) + iota_tot
    with np.errstate(divide="ignore", invalid="ignore"):
        nb1 = np.maximum(n, NMIN) ** (beta - 1.0)   # NMIN floor (no divergence)
    # value objects carry the demand multiplier D; densities and dW_bundle do not
    num = np.sum(D_grid * pi_cell * H * nb1 * grid.area)
    den = np.sum(D_grid * pi_cell * (n ** beta) * grid.area)
    labor_share = float(num / den) if den > 0 else np.nan

    # ── occupation value W_o = beta int h_o Pi n^{beta-1} (eq. occ-value) ──
    # strip on tasks (task-exact price), reinstated on the grid via the
    # per-worker iota_o/L_o; matches model.equilibrium.density_and_value to
    # machine precision.
    strip_val = beta * np.bincount(
        row_of, weights=b_w * (1.0 - a_task) * D_task * pi_task * nb1[cell_of],
        minlength=len(codes))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_worker_iota = np.where(L[:, None] > 0, iota / L[:, None], 0.0)
    reinstated_val = beta * (per_worker_iota
                             * (D_grid * pi_cell * nb1 * grid.area)[None, :]).sum(axis=1)
    W_o = strip_val + reinstated_val

    return {
        "codes": codes, "L": L,
        "D_o": D_o, "B_o": B_o, "dW_bundle": dW_bundle,
        "Delta_Gamma_D": Delta_Gamma_D, "M": M,
        "bound_mass": bound_mass, "unbound_mass": unbound_mass,
        "labor_share": labor_share, "W_o": W_o,
        "n": n, "n0": n0, "H": H, "u": u, "iota_tot": iota_tot,
    }


def _cell_index(grid: DiskGrid, xi, chi) -> np.ndarray:
    """Map (xi, chi) points to flat grid-cell indices matching DiskGrid.build
    (angle-major ordering)."""
    n_ang = int(round(2 * np.pi / (np.diff(np.unique(grid.xi))[0])))
    n_rad = grid.xi.size // n_ang
    ai = np.clip((xi % (2 * np.pi)) / (2 * np.pi) * n_ang, 0, n_ang - 1).astype(int)
    ri = np.clip(np.asarray(chi) * n_rad, 0, n_rad - 1).astype(int)
    return ai * n_rad + ri
