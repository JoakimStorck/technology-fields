"""
_interface.py
-------------
The static layer as a frozen input to the dynamic work.

Every numbered experiment script (d01, d02, ...) consumes the static paper's
calibration through this module and through nothing else: no d-script imports
scripts/_setup.py or constructs an Equilibrium directly. The contract is one
object, StaticLayer, holding

  - the frozen geometry and occupation inputs (inp, L0, occ),
  - the calibrated technology field (tech) and its unit-amplitude shape
    (g0_grid, g0_task) for maturation trajectories A_K(t),
  - the companion paper's economy parameters (R, tau, beta, gamma, ell,
    rho, lam_over -- Table `economy-parameters` of the static manuscript),
  - a survival-gated Equilibrium with the shared attachment primitive
    (eq.e = model.regime._fit = the companion's eq. `attachment`),
  - the mobility reference (kappa, c) computed by the companion's rule
    (kappa = one SD of baseline occupation value; the median move costs
    one kappa), evaluated at A_K = 0.

Mobility-reference evaluation state. The companion's scripts evaluate the
same rule with the Equilibrium constructed at the CALIBRATED technology
(a > 0 enters the strip value), giving kappa 11.61, c 22.58 (the static
table's 11.6/22.6). This layer evaluates it at the pre-shock baseline
A_K = 0 -- the state the dynamics start from -- giving kappa 11.84,
c 23.02 (the dynamic manuscript's 11.8/23.0). Same rule, different
evaluation state; the difference is two per cent and moves no reported
number at its stated precision (checked for the d08 fixed point). Every
dynamic-paper script consumes THIS reference, so the dynamic layer is
internally consistent; the static scripts are untouched, since changing
their kappa would perturb every static re-sort result.

If the dynamic layer moves to a dedicated repository, this module is the cut
line: load_static_layer() is reimplemented as a reader of a serialized
calibration artifact exported by the static pipeline, and the API stays.

Result-bearing scripts write figures, tables, and a plain-text summary to
experiment/results/ (RESULTS below), and assert their own baseline numbers so
a drifted calibration fails loudly instead of silently changing the paper.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.equilibrium import Equilibrium
from model.regime import _cell_index
from model.technology import Technology

RESULTS = REPO / "experiment" / "results"


class AdjustedField:
    """The estimated price field with a cellwise log-adjustment applied:
    Pi(r) = Pi_0(r) exp(dlnPi(r)). Level from data, change from the shock --
    the price object of the GE feedback (d07-d09), shared here so the three
    scripts carry one definition."""

    def __init__(self, base, dlnPi, grid):
        self._base, self._dlnPi, self._grid = base, dlnPi, grid

    def pi(self, xi, chi):
        cells = _cell_index(self._grid, np.asarray(xi, float), np.asarray(chi, float))
        return self._base.pi(xi, chi) * np.exp(self._dlnPi[cells])

    def __getattr__(self, name):
        return getattr(self._base, name)

# Companion-paper economy parameters (static Table `economy-parameters`).
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
RHO, LAM_OVER = 0.5, 1.0


class StaticLayer:
    """Plain container (not a dataclass: this module is loaded via importlib
    spec by scripts, where dataclass machinery breaks)."""

    def __init__(self, inp, L0, occ, tech, ell, eq, g0_grid, g0_task, kappa, c):
        self.inp, self.L0, self.occ = inp, L0, occ
        self.tech, self.ell, self.eq = tech, ell, eq
        self.g0_grid, self.g0_task = g0_grid, g0_task
        self.kappa, self.c = kappa, c
        self.R, self.tau, self.beta, self.gamma = R, TAU, BETA, GAMMA
        self.rho, self.lam_over = RHO, LAM_OVER

    def set_maturity(self, A_K: float) -> np.ndarray:
        """Update the A_K-dependent arrays of eq in place (s_K = 1, eta = 1)
        and return the operated share on the grid."""
        eq = self.eq
        a_grid = 1.0 / (1.0 + np.exp(-(A_K * self.g0_grid - self.R / eq.pi_cell) / self.tau))
        a_task = 1.0 / (1.0 + np.exp(-(A_K * self.g0_task - self.R / eq.pi_task) / self.tau))
        eq.a_grid = a_grid
        eq.a_task = a_task
        eq.D_o = np.bincount(eq.row_of, weights=eq.b_w * a_task, minlength=eq.n_occ)
        return a_grid


_CACHE: dict[str, StaticLayer] = {}


def load_static_layer(cached: bool = True) -> StaticLayer:
    """Build (once) and return the frozen static layer."""
    if cached and "layer" in _CACHE:
        return _CACHE["layer"]
    spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts" / "_setup.py")
    _setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_setup)

    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None, survival=True)
    eq.L0 = L0
    unit = Technology(xi_K=tech.xi_K, chi_K=tech.chi_K, z_K=tech.z_K, A_K=1.0, s_K=1.0)
    g0_grid = unit.phi(inp.grid.xi, inp.grid.chi)
    g0_task = unit.phi(eq.b_xi, eq.b_chi)

    layer = StaticLayer(inp=inp, L0=L0, occ=occ, tech=tech, ell=ell, eq=eq,
                        g0_grid=g0_grid, g0_task=g0_task, kappa=0.0, c=0.0)
    # Mobility reference at the pre-technology baseline (A_K = 0), by the
    # companion's rule: kappa = SD of baseline occupation value W0; the median
    # move costs one kappa.
    layer.set_maturity(0.0)
    _, _, W0 = eq.density_and_value(L0)
    layer.kappa = float(np.std(W0))
    layer.c = layer.kappa / float(np.median(eq.d[eq.d > 0]))

    if cached:
        _CACHE["layer"] = layer
    return layer


def write_summary(name: str, lines: list[str]) -> Path:
    """Write a plain-text summary to experiment/results/ and echo it."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}_summary.txt"
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"wrote {path}")
    return path
