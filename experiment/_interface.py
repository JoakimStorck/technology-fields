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
  - the anchored mobility reference (kappa, c, alpha), the committed
    static rule scripts/_setup.anchor_reference shared verbatim with the
    static pipeline: kappa = one SD of the ZERO-FIELD occupation value
    (technology-free), the median move costs one kappa, and the occupation
    constants alpha_o balance the logit kernel so that the observed L0 is
    the fixed point of the zero-field sorting map
    (model.equilibrium.anchor_alpha). eq.alpha is set, so every solve or
    resort on the layer's Equilibrium runs the anchored kernel.

Anchoring. The static manuscript states the requirement the constants
meet: without alpha_o the observed allocation is not a rest point of the
logit, and a solved path mixes the technology's effect with a baseline
relocation that has nothing to do with the technology. The pre-revision
layer carried no alpha and computed kappa from density_and_value after
set_maturity(0.0) -- a mixed state, since set_maturity did not refresh the
strip weights baked at the calibrated technology -- giving kappa 11.84,
c 23.02 against the anchored zero-field 16.37, 31.83 (the static table's
16.4/31.8). d00_zero_field_anchor.py guards the anchor and documents the
drift the pre-revision kernel produced.

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

    def __init__(self, inp, L0, occ, tech, ell, eq, g0_grid, g0_task, kappa, c,
                 alpha=None):
        self.inp, self.L0, self.occ = inp, L0, occ
        self.tech, self.ell, self.eq = tech, ell, eq
        self.g0_grid, self.g0_task = g0_grid, g0_task
        self.kappa, self.c, self.alpha = kappa, c, alpha
        self.R, self.tau, self.beta, self.gamma = R, TAU, BETA, GAMMA
        self.rho, self.lam_over = RHO, LAM_OVER

    def set_maturity(self, A_K: float) -> np.ndarray:
        """Update the A_K-dependent arrays of eq in place (s_K = 1, eta = 1)
        and return the operated share on the grid. The strip weights are
        refreshed with the takeover, so density_and_value is internally
        consistent at every maturity (the pre-revision version left strip_wD
        baked at the calibrated technology; at the layer's eta = 1 the demand
        multiplier D_task is identically one, so strip_wD = strip_w * D_task
        is exact)."""
        eq = self.eq
        a_grid = 1.0 / (1.0 + np.exp(-(A_K * self.g0_grid - self.R / eq.pi_cell) / self.tau))
        a_task = 1.0 / (1.0 + np.exp(-(A_K * self.g0_task - self.R / eq.pi_task) / self.tau))
        eq.a_grid = a_grid
        eq.a_task = a_task
        eq.D_o = np.bincount(eq.row_of, weights=eq.b_w * a_task, minlength=eq.n_occ)
        eq.strip_w = eq.b_w * (1.0 - a_task) * eq.pi_task
        eq.strip_wD = eq.strip_w * eq.D_task
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
    # Anchored mobility-and-alpha reference: the committed static rule,
    # shared verbatim with the static pipeline (scripts/_setup). kappa = one
    # SD of the zero-field occupation value; the median move costs one kappa;
    # alpha_o makes the observed L0 the fixed point of the zero-field sorting
    # map. The rule is technology-free (eq.zero_field_value reads no field
    # object), so the reference coincides with the pre-shock state A_K = 0
    # that the dynamics start from, and one (c, kappa, alpha) serves the era.
    c, kappa, _, alpha = _setup.anchor_reference(eq, L0)
    layer.kappa, layer.c, layer.alpha = kappa, c, alpha
    eq.alpha = alpha
    layer.set_maturity(0.0)          # leave eq in the pre-shock state

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
