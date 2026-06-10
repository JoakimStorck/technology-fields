"""
model.data
----------
Shared data access. Reads exclusively from data/ in this repository
(frozen by scripts/00_freeze_inputs.py; provenance in data/MANIFEST.json).

Two objects are provided:

  load_mincer_sample()  The Paper 1 Mincer sample: occupation polar
                        coordinates merged with BLS OEWS May 2023 median
                        hourly wages, restricted to positive H_MEDIAN and
                        non-missing rle_mean. Pins N = 785 and replicates
                        Paper 1, Table 3 exactly.

  load_family_map()     onet_code -> O*NET Job Family, for the measurement
                        layer (the theory layer is family-free; families
                        enter only through diagnostics and the wage wedge).

  load_bundles()        Task bundles b_o: every task with polar position
                        (xi, chi) and importance weight b normalized to sum
                        to one within each occupation (weights = O*NET task
                        ratings 'rt'). This is the empirical realization of
                        the bundle measure b_o(r) in the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
TASK_FILE = DATA / "task_embeddings_polar_scaled.csv"
WAGE_FILE = DATA / "national_M2023_dl.xlsx"
RLE_FILE = DATA / "occupation_rle.csv"


def _soc(code: pd.Series) -> pd.Series:
    """O*NET-SOC code -> detailed SOC (strip the .xx suffix)."""
    return code.astype(str).str.replace(r"\..*", "", regex=True).str.strip()


def load_wages() -> pd.DataFrame:
    """BLS OEWS May 2023 national file -> OCC_CODE, TOT_EMP, H_MEDIAN."""
    wages = pd.read_excel(WAGE_FILE, usecols=[8, 9, 11, 22])
    wages.columns = ["OCC_CODE", "OCC_TITLE", "TOT_EMP", "H_MEDIAN"]
    wages["OCC_CODE"] = _soc(wages["OCC_CODE"])
    wages["H_MEDIAN"] = pd.to_numeric(wages["H_MEDIAN"], errors="coerce")
    wages["TOT_EMP"] = pd.to_numeric(wages["TOT_EMP"], errors="coerce")
    return wages


def load_mincer_sample() -> pd.DataFrame:
    """Occupation-level estimation sample (N = 785), with ln_wage and the
    polar regressors used by the wage-field estimation."""
    occ = pd.read_csv(OCC_FILE).copy()
    occ["OCC_CODE"] = _soc(occ["onet_code"])
    df = occ.merge(
        load_wages()[["OCC_CODE", "TOT_EMP", "H_MEDIAN"]],
        on="OCC_CODE", how="left",
    )
    df = df.merge(pd.read_csv(RLE_FILE), on="onet_code", how="left")
    df = df.dropna(subset=["H_MEDIAN", "xi", "chi", "rle_mean"])
    df = df.loc[df["H_MEDIAN"] > 0].copy()

    df["ln_wage"] = np.log(df["H_MEDIAN"])
    df["cos_xi"] = np.cos(df["xi"])
    df["sin_xi"] = np.sin(df["xi"])
    df["chi_cos"] = df["chi"] * df["cos_xi"]
    df["chi_sin"] = df["chi"] * df["sin_xi"]
    df["chi_cos2"] = df["chi"] * np.cos(2 * df["xi"])
    df["chi_sin2"] = df["chi"] * np.sin(2 * df["xi"])
    return df.reset_index(drop=True)


def load_family_map() -> pd.Series:
    """onet_code -> Job Family (O*NET classification)."""
    occ = pd.read_csv(OCC_FILE, usecols=["onet_code", "Job Family"])
    return occ.set_index("onet_code")["Job Family"]


def load_bundles() -> pd.DataFrame:
    """Task-level bundles: one row per task with columns
    onet_code, Task ID, xi, chi, b (importance weight, sums to one per
    occupation)."""
    t = pd.read_csv(TASK_FILE, usecols=["onet_code", "Task ID", "xi", "chi", "rt"])
    t = t.dropna(subset=["xi", "chi", "rt"]).copy()
    t["b"] = t["rt"] / t.groupby("onet_code")["rt"].transform("sum")
    return t
