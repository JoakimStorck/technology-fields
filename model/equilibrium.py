"""
model.equilibrium
-----------------
The worker layer: occupation employment L_o as the fixed point of a logit
re-sorting over the post-technology occupation values (Sec. 3.8, eq.
equilibrium). The value W_o is supplied by the task layer (model.regime) and
depends on the employment-dependent density n(r), so the allocation is a fixed
point L = F(L), solved by damped Picard with a multi-start uniqueness check.

Re-sorting kernel (reading A, origin-destination; under test, not asserted as
the correct reading of eq. equilibrium). A worker at origin o' chooses a
destination o by a logit over value net of the mobility cost c d(o, o'):

    P(o | o') = softmax_o( (W_o + alpha_o - c d(o, o')) / kappa ),
    d(o, o') = ||mu_o^0 - mu_{o'}^0||,
    L_o = sum_{o'} L_{o'}^0 P(o | o').

The occupation constants alpha_o absorb unmodelled amenities and rationalize
the observed employment L0 as the fixed point of the ZERO-FIELD sorting map
(no technology: a = 0, M = 0). Without them the observed allocation is not a
rest point of the logit, and every solved equilibrium carries a baseline
drift that is not the technology's (the script-28 discovery: ~58 percent of
mass relocated under a 0.3 percent shock). With them, dL = L* - L0 is the
technology's own employment response. The constants are exactly identified
(n_occ - 1 free constants against n_occ - 1 marginal conditions; alpha is
defined up to an additive constant that cancels in the logit) and are
recovered by destination balancing of the kernel, `anchor_alpha` below.
alpha = 0 reproduces the unanchored pre-revision kernel.

Three modelling choices, settled with the author:
  - self-selection: the choice set includes o = o' (cost zero), so stickiness
    is endogenous in kappa rather than imposed by an L^0 term in the numerator;
  - frozen geography: d uses the PRE-technology centroids mu_o^0, the
    human-capital barrier a mover faces, not the rewritten bundles;
  - c and kappa are free, so the threshold result is checked for sign
    robustness across a (c, kappa) sweep (scripts/13), as the shape and
    L^tot scales are.

The kernel is isolated in `resort`; an alternative reading (single-reference
gravity, or stickiness via L^0) is a one-function swap.

Cost. The readiness e_o(r), the operated share a(r_t), displacement D_o, and
the gradient ring g_hat are all employment-independent and are computed once.
Inside the fixed point only the capacity C(r) = sum_o L_o e_o(r), the density,
and the value reduce to two matrix-vector products with the precomputed
readiness matrix, so a solve is cheap enough to sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import RegimeInputs, _ring_density, _readiness, _fit, NMIN


@dataclass
class EqOutcome:
    L: np.ndarray
    W: np.ndarray
    converged: bool
    iters: int
    residual: float
    n: np.ndarray
    C: np.ndarray


class Equilibrium:
    """Pre-computes the employment-independent structure for one technology and
    economy, then solves L = F(L)."""

    def __init__(self, inp: RegimeInputs, tech, R: float, tau: float,
                 gamma: float, ell: float, beta: float,
                 wedge: np.ndarray | None = None,
                 eta: float = 1.0, survival: bool = False,
                 rho: float = 0.5, lam_over: float = 1.0):
        # eta: demand elasticity. Competitive pricing passes the automation cost
        #   saving to consumers; isoelastic demand scales place revenue by the
        #   multiplier D(r) = (c(r)/Pi)^(1-eta) on the price field, where the unit
        #   cost is c(r) = (1-a)Pi + a R/(s_K phi_K). eta = 1 (unit elastic) gives
        #   D = 1 and reproduces the cost-invariant model; eta > 1 (elastic) draws
        #   labour into cheapened directions; eta < 1 (inelastic) releases it.
        #   survival: if True, reinstatement seeds survive as human work only by
        #   the price gate (1 - a(r)); if False, the un-gated model.
        self.inp, self.tech, self.beta, self.gamma = inp, tech, beta, gamma
        self.eta, self.survival = eta, survival
        grid, field = inp.grid, inp.field
        codes = inp.occ_codes()
        self.codes = codes
        self.n_occ = len(codes)

        # frozen pre-technology geography: pairwise centroid distance
        mx = (inp.occ["chi"] * np.cos(inp.occ["xi"])).to_numpy()
        my = (inp.occ["chi"] * np.sin(inp.occ["xi"])).to_numpy()
        self.mu = np.c_[mx, my]
        self.d = np.sqrt(((self.mu[:, None, :] - self.mu[None, :, :]) ** 2)
                         .sum(-1))                      # (o', o) symmetric

        # the wedge enters ONLY the takeover margin (Sec. 3.6 / start note):
        # dearer families cross sooner. Valuation and the share use the field.
        lw = np.zeros(self.n_occ) if wedge is None else np.asarray(wedge, float)

        bx = inp.bundles
        self.b_xi = bx["xi"].to_numpy(); self.b_chi = bx["chi"].to_numpy()
        self.b_w = bx["b"].to_numpy()
        self.row_of = pd.Index(codes).get_indexer(bx["onet_code"].to_numpy())
        lw_task = lw[self.row_of]
        self.a_task = tech.operated_share(self.b_xi, self.b_chi, field, R, tau,
                                          log_wedge=lw_task)
        self.D_o = np.bincount(self.row_of, weights=self.b_w * self.a_task,
                               minlength=self.n_occ)
        self.pi_task = field.pi(self.b_xi, self.b_chi)          # field price

        # grid pieces
        self.area = grid.area
        self.pi_cell = field.pi(grid.xi, grid.chi)
        self.g_hat = _ring_density(tech, grid)                  # ring shape
        self.e = _fit(inp, ell, rho, lam_over)                  # (n_occ, n_cells)
        self.cell_of = _cell_index(grid, self.b_xi, self.b_chi)

        # field-level operated share a(r) on the grid (no occupation wedge):
        # used for the survival gate (1 - a) and the unit cost c(r).
        self.a_grid = tech.operated_share(grid.xi, grid.chi, field, R, tau)
        # demand multiplier D(r) = (c(r)/Pi)^(1-eta), competitive passthrough of
        # the automation cost saving under isoelastic demand. Unit cost ratio
        # c/Pi = (1-a) + a (R / (s_K phi_K Pi)); <= 1 wherever capital is adopted
        # (the gate s_K phi_K > R/Pi makes the capital term < 1), = 1 where it is
        # not. eta = 1 -> D = 1 everywhere (cost-invariant model).
        phi_grid = tech.phi(grid.xi, grid.chi)
        with np.errstate(divide="ignore", invalid="ignore"):
            cratio_grid = np.where(phi_grid > 1e-9,
                                   R / (tech.s_K * phi_grid * self.pi_cell), 1.0)
        psi_grid = np.clip((1.0 - self.a_grid) + self.a_grid * cratio_grid,
                           1e-9, 1.0)
        self.D_grid = psi_grid ** (1.0 - eta)

        # task-level pieces for the W_o strip term, with its own D(task)
        phi_task = tech.phi(self.b_xi, self.b_chi)
        with np.errstate(divide="ignore", invalid="ignore"):
            cratio_task = np.where(phi_task > 1e-9,
                                   R / (tech.s_K * phi_task * self.pi_task), 1.0)
        psi_task = np.clip((1.0 - self.a_task) + self.a_task * cratio_task,
                           1e-9, 1.0)
        self.D_task = psi_task ** (1.0 - eta)
        self.strip_w = self.b_w * (1.0 - self.a_task) * self.pi_task
        self.strip_wD = self.strip_w * self.D_task

        self.L0 = None     # set by caller (employment shares)
        self.alpha = np.zeros(self.n_occ)   # anchoring constants; zero =
                                            # unanchored kernel (anchor_alpha)

    # ── value at a given employment ──────────────────────────────────
    def density_and_value(self, L):
        beta = self.beta
        M = self.gamma * float(np.sum(L * self.D_o))            # seeded mass
        s = M * self.g_hat
        surv = (1.0 - self.a_grid) if self.survival else 1.0    # price survival
        C = L @ self.e                                          # (n_cells,)
        Phi = np.where(C > 0, C / (1.0 + C), 0.0)
        iota_tot = s * surv * Phi                               # bound human mass
        n0 = (np.bincount(self.cell_of, weights=L[self.row_of] * self.b_w,
                          minlength=self.area.size) / self.area)
        n = n0 + iota_tot

        with np.errstate(divide="ignore", invalid="ignore"):
            nb1 = np.maximum(n, NMIN) ** (beta - 1.0)
        # W_o strip: beta sum_t b(1-a) D(task) Pi n^{beta-1}
        strip_val = beta * np.bincount(
            self.row_of, weights=self.strip_wD * nb1[self.cell_of],
            minlength=self.n_occ)
        # W_o reinstated: beta e @ gvec, gvec = s surv Phi/C D(r) Pi n^{b-1} area
        with np.errstate(divide="ignore", invalid="ignore"):
            gvec = (np.where(C > 0, s * surv * Phi / C, 0.0)
                    * self.D_grid * self.pi_cell * nb1 * self.area)
        W = strip_val + beta * (self.e @ gvec)
        return n, C, W

    # ── re-sorting kernel (reading A, anchored) ──────────────────────
    def resort(self, W, c, kappa):
        U = (W[None, :] + self.alpha[None, :]
             - c * self.d) / kappa                     # origins x destinations
        U -= U.max(axis=1, keepdims=True)
        P = np.exp(U)
        P /= P.sum(axis=1, keepdims=True)
        return self.L0 @ P

    # ── fixed point ──────────────────────────────────────────────────
    def solve(self, c, kappa, lam=0.5, tol=1e-10, maxit=2000,
              L_init=None) -> EqOutcome:
        L = self.L0.copy() if L_init is None else L_init.copy()
        res, it = np.inf, 0
        for it in range(1, maxit + 1):
            n, C, W = self.density_and_value(L)
            Ln = self.resort(W, c, kappa)
            res = float(np.abs(Ln - L).sum())
            L = (1 - lam) * L + lam * Ln
            if res < tol:
                break
        n, C, W = self.density_and_value(L)
        return EqOutcome(L=L, W=W, converged=res < tol, iters=it,
                         residual=res, n=n, C=C)

    def multistart(self, c, kappa, n_random=3, seed=0):
        """Solve from L^0, uniform, and random starts; return the L^0 solution
        and the max pairwise sup-distance between converged solutions."""
        rng = np.random.default_rng(seed)
        starts = [self.L0.copy(), np.full(self.n_occ, 1.0 / self.n_occ)]
        for _ in range(n_random):
            r = rng.random(self.n_occ); starts.append(r / r.sum())
        sols = [self.solve(c, kappa, L_init=s) for s in starts]
        Ls = np.array([o.L for o in sols])
        spread = float(np.max(np.abs(Ls[:, None, :] - Ls[None, :, :]).sum(-1)))
        return sols[0], spread, all(o.converged for o in sols)

    # ── descriptive statistics at a given employment ─────────────────
    def labor_share(self, L):
        """Lambda = int D Pi H n^{beta-1} / int D Pi n^beta, the human task-work
        share of factor income, with the demand multiplier D(r) weighting each
        place's revenue. At eta = 1, D = 1 and this is the cost-invariant share."""
        beta = self.beta
        M = self.gamma * float(np.sum(L * self.D_o))
        s = M * self.g_hat
        surv = (1.0 - self.a_grid) if self.survival else 1.0
        C = L @ self.e
        Phi = np.where(C > 0, C / (1.0 + C), 0.0)
        iota_tot = s * surv * Phi
        n0 = (np.bincount(self.cell_of, weights=L[self.row_of] * self.b_w,
                          minlength=self.area.size) / self.area)
        n = n0 + iota_tot
        # human task-work H = (1-a) n0 + bound human reinstatement, with the
        # strip evaluated per task (a_task, wedge-bearing) and binned, so it
        # matches the W_o strip and regime.py rather than the cell-centre a_grid.
        La_binned = (np.bincount(self.cell_of,
                                 weights=L[self.row_of] * self.b_w * self.a_task,
                                 minlength=self.area.size) / self.area)
        H = (n0 - La_binned) + iota_tot
        with np.errstate(divide="ignore", invalid="ignore"):
            nb1 = np.maximum(n, NMIN) ** (beta - 1.0)
        w = self.D_grid * self.pi_cell
        num = float(np.sum(w * H * nb1 * self.area))
        den = float(np.sum(w * (n ** beta) * self.area))
        return num / den if den > 0 else np.nan

    def pretech_value(self, L):
        """Pre-technology occupation value W_o^0 = beta int b_o Pi n0^{beta-1}:
        no takeover, no reinstatement, A = 1. The baseline wage the post-shock
        value is measured against."""
        beta = self.beta
        n0 = (np.bincount(self.cell_of, weights=L[self.row_of] * self.b_w,
                          minlength=self.area.size) / self.area)
        with np.errstate(divide="ignore", invalid="ignore"):
            nb1 = np.where(n0 > 0, n0 ** (beta - 1.0), 0.0)
        return beta * np.bincount(
            self.row_of, weights=self.b_w * self.pi_task * nb1[self.cell_of],
            minlength=self.n_occ)

    def zero_field_value(self, L):
        """W_o of the zero-field economy at employment L: the exact a = 0,
        M = 0 limit of density_and_value, WITH its NMIN density floor. This
        is the value the sorting map iterates on when the technology is
        switched off, and therefore the anchoring reference for alpha_o.
        It is technology-free (no field object is read) and differs from
        pretech_value only in the floor, which pretech_value omits."""
        beta = self.beta
        n0 = (np.bincount(self.cell_of, weights=L[self.row_of] * self.b_w,
                          minlength=self.area.size) / self.area)
        with np.errstate(divide="ignore", invalid="ignore"):
            nb1 = np.maximum(n0, NMIN) ** (beta - 1.0)
        return beta * np.bincount(
            self.row_of, weights=self.b_w * self.pi_task * nb1[self.cell_of],
            minlength=self.n_occ)


def anchor_alpha(W0, d, L0, c, kappa, tol=1e-13, maxit=10000):
    """Occupation constants alpha_o that make the observed L0 the fixed
    point of the zero-field sorting map at value W0 = zero_field_value(L0).

    Write v_o = exp(alpha_o / kappa) and K[o', o] = exp((W0_o - c d)/kappa).
    The requirement is that the destination marginals of diag(L0) P(v),
    with P the row-normalisation of K diag(v), equal L0. That is the
    destination-constrained gravity calibration -- a one-sided Sinkhorn
    balancing:

        v  <-  v * L0 / (L0^T P(v)),

    normalised to geometric mean one (alpha up to an additive constant,
    which cancels in the logit). For strictly positive K, which the
    exponential guarantees, the balanced v exists and is unique up to
    scale (Sinkhorn's theorem, one-sided case), and the iteration
    converges globally. The convergence criterion IS the fixed-point
    property: the L1 gap between the implied marginals and L0.

    Exactly identified: n_occ - 1 free constants against n_occ - 1
    marginal conditions (shares sum to one on both sides).

    Raises RuntimeError if the tolerance is not reached; certified runs
    must not proceed on a broken anchor."""
    W0 = np.asarray(W0, float)
    L0 = np.asarray(L0, float)
    U = (W0[None, :] - c * d) / kappa
    U -= U.max(axis=1, keepdims=True)
    K = np.exp(U)
    v = np.ones_like(L0)
    err = np.inf
    for _ in range(maxit):
        P = K * v[None, :]
        P /= P.sum(axis=1, keepdims=True)
        m = L0 @ P
        err = float(np.abs(m - L0).sum())
        if err < tol:
            break
        v *= L0 / np.maximum(m, 1e-300)
        v /= np.exp(np.mean(np.log(v)))
    if err >= tol:
        raise RuntimeError(
            f"anchor_alpha: balancing did not converge (L1 marginal gap "
            f"{err:.2e} after {maxit} iterations, tol {tol:.0e})")
    return kappa * np.log(v)


def _cell_index(grid, xi, chi) -> np.ndarray:
    n_ang = int(round(2 * np.pi / (np.diff(np.unique(grid.xi))[0])))
    n_rad = grid.xi.size // n_ang
    ai = np.clip((np.asarray(xi) % (2 * np.pi)) / (2 * np.pi) * n_ang,
                 0, n_ang - 1).astype(int)
    ri = np.clip(np.asarray(chi) * n_rad, 0, n_rad - 1).astype(int)
    return ai * n_rad + ri
