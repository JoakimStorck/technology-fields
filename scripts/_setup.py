"""
_setup.py
---------
Shared construction of the regime/equilibrium inputs for the regime scripts
(12-14): the task bundles, occupation centroids and cluster intensities,
pre-technology employment shares, the calibrated AI technology, the family
wage wedge, and interpretable scales for the readiness ell and the mobility
parameters (c, kappa). Reads only from data/ and results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.capability_field import CapabilityField, PRICED
from model.data import load_bundles
from model.price_field import PriceField
from model.regime import DiskGrid, RegimeInputs
from model.technology import Technology

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
CLUSTER_FILE = DATA / "occupation_cluster_intensity.csv"
WAGE_FILE = DATA / "national_M2023_dl.xlsx"
WEDGE_FILE = RESULTS / "family_wage_wedge.csv"


def _soc(s):
    return s.astype(str).str.replace(r"\..*", "", regex=True).str.strip()


def build_inputs(n_ang: int = 120, n_rad: int = 40):
    """RegimeInputs plus pre-technology employment shares L0 (sum 1) and the
    occupation table (indexed by onet_code, carrying Job Family, centroid,
    cluster intensities, and L0_share)."""
    cap = CapabilityField.from_results()
    field = PriceField.from_results()
    bundles = load_bundles()

    occ = pd.read_csv(OCC_FILE, usecols=["onet_code", "xi", "chi", "Job Family",
                                         "Title"])
    occ = occ.merge(pd.read_csv(CLUSTER_FILE), on="onet_code", how="inner")

    wages = pd.read_excel(WAGE_FILE, usecols=[8, 11])
    wages.columns = ["OCC_CODE", "TOT_EMP"]
    wages["OCC_CODE"] = _soc(wages["OCC_CODE"])
    wages["TOT_EMP"] = pd.to_numeric(wages["TOT_EMP"], errors="coerce")
    emp = wages.groupby("OCC_CODE")["TOT_EMP"].first()
    occ["OCC_CODE"] = _soc(occ["onet_code"])
    occ["L0_emp"] = occ["OCC_CODE"].map(emp)
    occ = occ.dropna(subset=["L0_emp", "xi", "chi", "S1", "S2", "A1", "A2"])
    occ = occ[occ["L0_emp"] > 0].copy()
    # employment shares (L^tot = 1): makes the saturating attachment Phi(C)
    # = C/(1+C) meaningful (C must be O(1)).
    occ["L0"] = occ["L0_emp"] / occ["L0_emp"].sum()

    bundles = bundles[bundles["onet_code"].isin(occ["onet_code"])].copy()
    occ = occ.set_index("onet_code")
    grid = DiskGrid.build(n_ang=n_ang, n_rad=n_rad)
    inp = RegimeInputs(bundles=bundles, occ=occ, field=field, cap=cap, grid=grid)
    return inp, occ["L0"].to_numpy(), occ


def load_tech() -> Technology:
    """The calibrated AI technology (primary, unweighted fit; script 08)."""
    pr = pd.read_csv(RESULTS / "technology_calibration.csv")
    p = pr.loc[pr["fit"] == "unweighted"].iloc[0]
    return Technology(xi_K=float(p["xi_K_rad"]), chi_K=float(p["chi_K"]),
                      z_K=float(p["z_K"]), A_K=float(p["A_K"]), s_K=1.0)


def interpretable_ell(inp: RegimeInputs) -> float:
    """ell so that a deficit of one within-direction SD of the priced-capability
    index gives readiness e^{-1}."""
    cap, occ = inp.cap, inp.occ
    idx = sum(cap.v[k] * occ[k].to_numpy() for k in PRICED)
    sec = np.floor(((np.degrees(occ["xi"].to_numpy()) + 22.5) % 360) / 45).astype(int)
    sds = [idx[sec == s].std() for s in range(8) if (sec == s).sum() >= 5]
    return float(np.mean(sds))


def load_wedge(occ) -> np.ndarray:
    """Per-occupation log wage wedge eta_o = eta_{g(o)} (family mean), aligned
    to occ.index. Occupations whose family has no wedge get 0."""
    wedge = pd.read_csv(WEDGE_FILE).set_index("Job Family")["eta_g"]
    return occ["Job Family"].map(wedge).fillna(0.0).to_numpy()


def mobility_reference(W0: np.ndarray, d: np.ndarray):
    """Interpretable (c, kappa): kappa = SD of baseline value W_o (one SD of
    value is one logit unit); c = kappa / median within-subsystem move, so a
    typical move costs about one logit unit -- mobility damped but not dead."""
    kappa = float(np.std(W0))
    dmed = float(np.median(d[d > 0]))
    c = kappa / dmed
    return c, kappa, dmed
