"""
23_webb_fields.py
-----------------
Calibrates the three technology fields of Webb (2020) -- robots, software, AI --
against the occupational task disk, and places Webb's AI field beside the
Eloundou et al. (2024) AI field on identical footing.

Motivation. The cognitive field of the paper is fitted to the Eloundou task-
level LLM exposure (scripts/08). Webb (2020) provides an independent family of
occupational exposure measures for THREE technologies -- industrial robots,
software, and AI -- built by the same task-matching logic: verb-noun pairs from
patent text are matched to O*NET task descriptions and aggregated to occupations
(Webb 2020; Montobbio et al. 2024 give a robotics-specific alternative). Webb's
AI measure is patent-based where Eloundou's is capability-based, so calibrating
both and comparing their centres is a cross-measure check on where the AI field
sits. Webb's robot measure is the industrial-automation counterpart used as the
placebo field in the paper's identification section, and calibrating it here
puts that field on the same data footing as the AI field rather than placing it
by hand.

Method. The same Gaussian model and squared-error loss as scripts/08, with two
deliberate changes. First, Webb's measures are occupation-level, so the fit is
run on the OCCUPATION substrate, one point per O*NET-SOC occupation at its
bundle centroid (xi, chi), rather than on the 17k task points; for a like-for-
like comparison the Eloundou surface is re-aggregated to the same occupation
substrate (task beta averaged by task weight rt) and refitted here, while the
headline task-level Eloundou calibration remains the one in
results/technology_calibration.csv. Second, the centre is constrained to the
occupied disk (chi_K <= the occupational support radius) by polar
reparametrisation, so a field cannot be placed where no work exists; 08's
Cartesian box does not impose this, and it only binds for surfaces that are
monotone gradients rather than localised peaks. Each field is fitted unweighted,
which calibrates the technology's spatial form (its centre p_K and reach z_K);
the economic level is set downstream by R, tau, s_K and is not calibrated here.

AWAITING INPUT (the Webb side stays skipped until the file is present):
  Webb's raw-score file (preferred): keyed by onetsoccode with continuous
      columns ai_score / software_score / robot_score (Webb's data page, a Stata
      file; the raw scores are concentrated, so the fitted reach reflects a
      technology's true spread rather than a flattened percentile rank). Place as
      data/final_df_out.dta or data/webb2020_exposure.csv; a Stata file with any
      extension is read directly, and onetsoccode joins the disk without a
      crosswalk.
  Webb's percentile file (fallback): keyed by occ1990dd with pct_ai /
      pct_software / pct_robot; then data/onet_to_occ1990dd.dta (a Stata
      crosswalk onetsoccode<->occ1990dd) carries it onto the O*NET-SOC disk.
      Percentiles are uniform by construction and overstate a technology's reach.
  Amplitude A_K absorbs the scale, so centre and reach are scale-free; the
  inputs are hashed into data/MANIFEST.json under external_inputs. Run --smoke
  to exercise the pipeline on a synthetic corpus (stamped NOT A RESULT).

Reads:
    data/occupation_embeddings_polar_scaled.csv   occupation coordinates
    data/task_embeddings_polar_scaled.csv         task weights rt (Eloundou agg)
    data/onet_task_exposure.csv                   Eloundou task beta
    data/webb2020_exposure.csv                    Webb occupation exposure (input)
Writes:
    results/webb_calibration.csv           calibrated (p_K, z_K, A_K, fit) per field
    results/webb_calibration_summary.txt
    results/webb_field_comparison.png      fitted fields on the disk, side by side
    results/webb_points_and_rings.png      exposure points with the fitted centre and z_K ring, per field
    data/MANIFEST.json                     external_inputs entry for the Webb file

Usage:
    python scripts/23_webb_fields.py            # needs data/webb2020_exposure.csv
    python scripts/23_webb_fields.py --smoke    # synthetic corpus, stamped SMOKE
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

OCC_FILE = DATA / "occupation_embeddings_polar_scaled.csv"
TASK_FILE = DATA / "task_embeddings_polar_scaled.csv"
EXPOSURE_FILE = DATA / "onet_task_exposure.csv"
def _first_existing(*paths: Path) -> Path:
    """First path that exists, else the first (canonical) name."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# The exposure file (Webb's raw-score file preferred, the percentile file as a
# fallback) and its crosswalk, under canonical or original download names.
WEBB_FILE = _first_existing(DATA / "webb2020_exposure.dta",
                            DATA / "webb2020_exposure.csv",
                            DATA / "final_df_out.dta",
                            DATA / "final_df_out_dta.txt",
                            DATA / "exposure_by_occ1990dd_lswt2010.csv")
WEBB_XWALK = _first_existing(DATA / "onet_to_occ1990dd.dta",       # Stata crosswalk
                             DATA / "onet_to_occ1990dd_dta.txt")
MANIFEST = DATA / "MANIFEST.json"

# Webb file schema (edit to match the downloaded file; auto-detection is used
# when a named column is absent). Webb's public file is keyed by occ1990dd with
# percentile columns, and is crosswalked to O*NET-SOC through WEBB_XWALK; a file
# already keyed by O*NET-SOC or SOC is used directly, without the crosswalk.
WEBB_COLS = {"robot": "pct_robot", "software": "pct_software", "ai": "pct_ai"}

# Order and colours for the three Webb fields plus the Eloundou reference.
# Chronological order of the automation waves, oldest first: industrial robots,
# software/computerisation, machine learning, then language models. The summary
# table, the figures and the csv all follow this order.
FIELD_ORDER = ["webb_robot", "webb_software", "webb_ai", "eloundou"]
FIELD_LABEL = {"eloundou": "Eloundou AI (occ.)", "webb_ai": "Webb AI",
               "webb_software": "Webb software", "webb_robot": "Webb robots"}
FIELD_COLOR = {"eloundou": "#1f77b4", "webb_ai": "#2ca02c",
               "webb_software": "#9467bd", "webb_robot": "#d62728"}

# Reuse the exact calibration of scripts/08 (single source of truth for method).
_spec = importlib.util.spec_from_file_location(
    "calib08", Path(__file__).parent / "08_calibrate_technology.py")
calib08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib08)


# ─────────────────────────────────────────────────────────────────────
# Keys and data
# ─────────────────────────────────────────────────────────────────────

def _soc6(code: pd.Series) -> pd.Series:
    """Normalise an O*NET-SOC (XX-XXXX.XX) or SOC (XX-XXXX) code to 6-digit
    SOC XX-XXXX. Webb is joined to the geometry on this key; several O*NET-SOC
    occupations sharing a SOC receive the same Webb value (broadcast)."""
    s = code.astype(str).str.strip()
    return s.str.extract(r"(\d{2}-\d{4})", expand=False).fillna(s)


def load_occupations() -> pd.DataFrame:
    occ = pd.read_csv(OCC_FILE, usecols=["onet_code", "xi", "chi"])
    occ["soc6"] = _soc6(occ["onet_code"])
    occ["x"] = occ["chi"] * np.cos(occ["xi"])
    occ["y"] = occ["chi"] * np.sin(occ["xi"])
    return occ


def eloundou_by_occupation() -> pd.Series:
    """Aggregate task-level Eloundou beta to the occupation, weighting each task
    by its within-occupation weight rt (the same weight scripts/08 uses to build
    the employment share b). Returns a Series indexed by onet_code."""
    t = pd.read_csv(TASK_FILE, usecols=["onet_code", "Task ID", "rt"])
    t["task_id"] = t["Task ID"].astype(int)
    e = pd.read_csv(EXPOSURE_FILE)
    df = t.merge(e[["task_id", "phi"]], on="task_id", how="inner")
    # rt-weighted occupation mean = sum(phi*rt)/sum(rt) per occupation;
    # vectorised, so no dependence on the pandas groupby-apply signature.
    num = (df["phi"] * df["rt"]).groupby(df["onet_code"]).sum()
    den = df["rt"].groupby(df["onet_code"]).sum()
    agg = (num / den).rename("eloundou")
    return agg


def _detect(cols, want, fallback):
    """Case-insensitive column pick: exact/`want`, else first column whose name
    contains `fallback`, else None."""
    low = {c.lower(): c for c in cols}
    if want.lower() in low:
        return low[want.lower()]
    for c in cols:
        if fallback in c.lower():
            return c
    return None


def _read_table(path: Path) -> pd.DataFrame:
    """Read a Stata (.dta, or a Stata file with any extension) or CSV table."""
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:11] == b"<stata_dta>" or path.suffix.lower() == ".dta":
        return pd.read_stata(path)
    return pd.read_csv(path)


def _pick_tech(cols, tech: str):
    """Column for a technology, tried in priority order so that a substring like
    'ai' in 'agg_pairs' cannot win over 'ai_score'/'pct_ai'. Returns None if no
    column matches."""
    low = {c.lower(): c for c in cols}
    for cand in (f"{tech}_score", f"pct_{tech}", f"{tech}_pctile",
                 f"{tech}_exposure", tech):
        if cand in low:
            return low[cand]
    for c in cols:                       # last resort: name starts/ends with tech
        cl = c.lower()
        if cl.startswith(tech) or cl.endswith(tech) or cl.endswith(f"_{tech}"):
            return c
    return None


def load_webb() -> pd.DataFrame | None:
    """Read the Webb exposure file and return a frame indexed by O*NET-SOC
    (onet_code) with columns robot/software/ai. Webb's raw-score file is keyed
    by onetsoccode (used directly); his percentile file is keyed by occ1990dd
    and is crosswalked to O*NET-SOC through WEBB_XWALK, broadcasting each
    occ1990dd value to the O*NET-SOC occupations sharing it. Returns None if the
    file is absent."""
    if not WEBB_FILE.exists():
        return None
    w = _read_table(WEBB_FILE)
    tech_col = {t: _pick_tech(w.columns, t) for t in WEBB_COLS}
    tech_col = {t: c for t, c in tech_col.items() if c is not None}
    if not tech_col:
        raise ValueError(f"no exposure columns found in {WEBB_FILE.name}; "
                         f"columns: {list(w.columns)}")

    onet_col = _detect(w.columns, "onetsoccode", "onet")
    if onet_col and w[onet_col].astype(str).str.contains(r"\d\d-\d{4}").any():
        out = pd.DataFrame({"onet_code": w[onet_col].astype(str).str.strip()})
        for t, c in tech_col.items():
            out[t] = pd.to_numeric(w[c], errors="coerce")
    elif "occ1990dd" in w.columns:
        if not WEBB_XWALK.exists():
            raise FileNotFoundError(
                f"{WEBB_FILE.name} is keyed by occ1990dd; the crosswalk "
                f"{WEBB_XWALK.name} (onetsoccode<->occ1990dd) is required. "
                f"Place it in data/ (Dorn / Webb).")
        xw = _read_table(WEBB_XWALK)
        oc = _detect(xw.columns, "onetsoccode", "onet")
        xw = xw[[oc, "occ1990dd"]].rename(columns={oc: "onet_code"}).dropna()
        xw["occ1990dd"] = xw["occ1990dd"].astype(int)
        w = w.copy()
        w["occ1990dd"] = w["occ1990dd"].astype(int)
        m = xw.merge(w[["occ1990dd"] + list(tech_col.values())],
                     on="occ1990dd", how="left")
        out = pd.DataFrame({"onet_code": m["onet_code"].astype(str).str.strip()})
        for t, c in tech_col.items():
            out[t] = pd.to_numeric(m[c], errors="coerce")
    else:
        raise ValueError(
            f"{WEBB_FILE.name} has neither an O*NET-SOC/SOC column nor "
            f"occ1990dd; columns: {list(w.columns)}")
    out = out.dropna(how="all", subset=list(tech_col))
    return out.groupby("onet_code").mean(numeric_only=True)


def synthetic_webb(occ: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """SMOKE ONLY. A synthetic Webb-shaped surface: robots peaking in the
    technical-physical west, AI in the analytical north-east, software broad.
    NOT A RESULT -- exercises the pipeline when the real file is absent."""
    rng = np.random.default_rng(seed)
    x, y = occ["x"].to_numpy(), occ["y"].to_numpy()

    def bump(cx, cy, z):
        return np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / z ** 2)

    df = pd.DataFrame({"onet_code": occ["onet_code"].to_numpy()})
    df["robot"] = bump(-0.45, 0.0, 0.30) + 0.05 * rng.standard_normal(len(x))
    df["ai"] = bump(0.30, 0.28, 0.55) + 0.05 * rng.standard_normal(len(x))
    df["software"] = bump(0.10, 0.20, 0.75) + 0.05 * rng.standard_normal(len(x))
    return df.set_index("onet_code")[["robot", "software", "ai"]].clip(lower=0)


def build_surfaces(smoke: bool) -> tuple[dict[str, pd.DataFrame], str]:
    """Return {field_name: occupation frame with x, y, xi, chi, phi} for every
    field that has data, and a provenance tag."""
    occ = load_occupations()
    elo = eloundou_by_occupation()
    occ = occ.merge(elo, left_on="onet_code", right_index=True, how="left")

    if smoke:
        webb = synthetic_webb(occ)
        tag = "SMOKE (synthetic corpus -- NOT A RESULT)"
    else:
        webb = load_webb()
        tag = f"Webb (2020), frozen in {WEBB_FILE.name}"
    if webb is not None:
        occ = occ.merge(webb.add_prefix("webb_"), left_on="onet_code",
                        right_index=True, how="left")

    surfaces: dict[str, pd.DataFrame] = {}
    for name in FIELD_ORDER:
        if name not in occ.columns:
            continue
        sub = occ[["xi", "chi", "x", "y", name]].dropna().copy()
        sub = sub.rename(columns={name: "phi"})
        if len(sub) >= 30:
            surfaces[name] = sub
    return surfaces, tag


# ─────────────────────────────────────────────────────────────────────
# Fit (delegated to scripts/08) + comparison
# ─────────────────────────────────────────────────────────────────────

def _calibrate_interior(x, y, b, chi_max):
    """Fit (chi_K, xi_K, z_K, A_K) with the centre constrained to the occupied
    disk, chi_K <= chi_max, by polar reparametrisation. Same Gaussian model and
    squared-error loss as scripts/08; the only change is that the centre is
    forced to lie where occupations exist, which 08's Cartesian box does not
    guarantee. A coarse profiled-A_K grid seeds a trust-region solve."""
    best = None
    for chiK in np.linspace(0.0, chi_max, 16):
        for xiK in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
            px, py = chiK * np.cos(xiK), chiK * np.sin(xiK)
            for zK in np.linspace(0.10, 1.20, 12):
                g = np.exp(-0.5 * ((x - px) ** 2 + (y - py) ** 2) / zK ** 2)
                den = np.sum(g * g)
                A = np.sum(g * b) / den if den > 0 else 0.0
                if A <= 0:
                    continue
                sse = np.sum((A * g - b) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, chiK, xiK, zK, A)
    _, chiK, xiK, zK, A = best

    def resid(p):
        cK, xK, zk, a = p
        px, py = cK * np.cos(xK), cK * np.sin(xK)
        g = np.exp(-0.5 * ((x - px) ** 2 + (y - py) ** 2) / zk ** 2)
        return a * g - b

    res = least_squares(
        resid, [chiK, xiK, zK, A], method="trf",
        bounds=([0.0, -2 * np.pi, 1e-3, 0.0], [chi_max, 2 * np.pi, 2.0, 5.0]))
    cK, xK, zk, a = res.x
    return cK * np.cos(xK), cK * np.sin(xK), zk, a, res


def fit_field(df: pd.DataFrame, chi_max: float) -> dict:
    """Calibrate one Gaussian field on the occupation substrate, unweighted,
    with the centre constrained to the occupied disk (chi_K <= chi_max). Same
    Gaussian model and loss as scripts/08. Each surface is scaled to [0,1]
    before the fit (Webb ships percentiles, Eloundou ships beta in [0,1]);
    centre and reach are invariant to a positive rescaling, and A_K is then
    reported in max-normalised units."""
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    b = df["phi"].to_numpy().astype(float)
    bmax = b.max()
    bn = b / bmax if bmax > 0 else b
    dfn = df.assign(phi=bn)
    w = np.ones(len(df))
    px, py, zK, A_K, res = _calibrate_interior(x, y, bn, chi_max)
    chiK = float(np.hypot(px, py))
    xiK = float(np.degrees(np.arctan2(py, px)) % 360)
    r2 = calib08.weighted_r2(x, y, bn, w, px, py, zK, A_K)
    sf = calib08.surface_fit(dfn, px, py, zK, A_K)
    at_bound = chiK >= 0.995 * chi_max
    return dict(px=px, py=py, xi_K_deg=xiK, chi_K=chiK, z_K=zK, A_K=A_K,
                n=len(df), r2_task=float(r2), r2_surface=sf["r2_surface"],
                eta2_between_cell=sf["eta2"], at_bound=at_bound)


def compare(a: dict, b: dict) -> dict:
    """Head-to-head geometry of two fitted fields (a vs b)."""
    dang = ((a["xi_K_deg"] - b["xi_K_deg"] + 180) % 360) - 180
    dcent = float(np.hypot(a["px"] - b["px"], a["py"] - b["py"]))
    return dict(d_angle_deg=dang, d_chi=a["chi_K"] - b["chi_K"],
                d_center=dcent, z_ratio=a["z_K"] / b["z_K"])


# ─────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────

def _disk(ax):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1.0))
    ax.set_aspect("equal"); ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([]); ax.set_yticks([])
    for ang, lab in [(0, "human-centred\n(east)"), (90, "analytical (north)"),
                     (180, "technical-physical\n(west)"), (270, "service (south)")]:
        ax.text(1.12 * np.cos(np.radians(ang)), 1.12 * np.sin(np.radians(ang)),
                lab, ha="center", va="center", fontsize=7, color="0.4")


def figure_comparison(occ_xy, fits: dict, tag: str, out: Path):
    """Left: every field's centre and z_K ring on the occupation cloud.
    Right: the fitted Gaussian of each field as filled contours, small multiples.
    """
    names = [n for n in FIELD_ORDER if n in fits]
    grid = np.linspace(-1, 1, 241)
    X, Y = np.meshgrid(grid, grid)
    inside = np.hypot(X, Y) <= 1.0

    ncol = len(names)
    fig = plt.figure(figsize=(4.6 + 2.7 * ncol, 5.2))
    gs = fig.add_gridspec(1, 1 + ncol, width_ratios=[1.5] + [1] * ncol)

    axO = fig.add_subplot(gs[0])
    _disk(axO)
    axO.scatter(occ_xy[0], occ_xy[1], s=5, color="0.85", zorder=1)
    for n in names:
        f = fits[n]
        c = FIELD_COLOR[n]
        axO.add_patch(plt.Circle((f["px"], f["py"]), f["z_K"], fill=False,
                                 color=c, lw=1.6, ls="--", zorder=3))
        axO.plot(f["px"], f["py"], "x", color=c, ms=9, mew=2.2, zorder=4,
                 label=f"{FIELD_LABEL[n]}: "
                       f"$\\xi$={f['xi_K_deg']:.0f}$^\\circ$ "
                       f"$\\chi$={f['chi_K']:.2f} $z$={f['z_K']:.2f}")
    axO.legend(loc="lower left", frameon=False, fontsize=7.2,
               bbox_to_anchor=(-0.05, -0.02))
    axO.set_title("Field centres and $z_K$ rings", fontsize=10)

    for k, n in enumerate(names):
        ax = fig.add_subplot(gs[1 + k])
        _disk(ax)
        f = fits[n]
        Z = np.where(inside, f["A_K"] * np.exp(
            -0.5 * ((X - f["px"]) ** 2 + (Y - f["py"]) ** 2) / f["z_K"] ** 2),
            np.nan)
        ax.contourf(X, Y, Z, levels=18, cmap="viridis", zorder=0)
        ax.add_patch(plt.Circle((f["px"], f["py"]), f["z_K"], fill=False,
                                color="white", lw=1.1, ls="--", zorder=3))
        ax.plot(f["px"], f["py"], "w+", ms=9, mew=1.8, zorder=4)
        ax.set_title(f"{FIELD_LABEL[n]}\n$R^2_{{\\mathrm{{surf}}}}$="
                     f"{f['r2_surface']:.2f}", fontsize=9)

    fig.suptitle(f"Technology fields in chronological order (oldest first) "
                 f"\u2014 {tag}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_points_rings(surfaces: dict, fits: dict, out: Path):
    """One panel per field: the occupation data points coloured by exposure,
    with the fitted centre p_K (x) and the z_K ring drawn on top, so the fitted
    field circle can be judged directly against the data it was fitted to."""
    names = [n for n in FIELD_ORDER if n in surfaces]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4.6))
    if len(names) == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        _disk(ax)
        d = surfaces[n]
        sc = ax.scatter(d["x"], d["y"], c=d["phi"], s=13, cmap="viridis",
                        alpha=0.85, edgecolors="none", zorder=2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.66)
        cb.set_label("percentile" if n.startswith("webb_") else "beta",
                     fontsize=8)
        f = fits[n]
        ax.add_patch(plt.Circle((f["px"], f["py"]), f["z_K"], fill=False,
                                color="crimson", lw=1.8, ls="--", zorder=4))
        ax.plot(f["px"], f["py"], "x", color="crimson", ms=11, mew=2.4,
                zorder=5)
        tag = "  (centre pinned)" if f.get("at_bound") else ""
        ax.set_title(f"{FIELD_LABEL[n]} (n={len(d)})\n"
                     f"$\\xi$={f['xi_K_deg']:.0f}$^\\circ$ "
                     f"$\\chi$={f['chi_K']:.2f} $z$={f['z_K']:.2f}{tag}",
                     fontsize=9)
    fig.suptitle("Technology fields in chronological order (oldest first): "
                 "exposure points with the fitted centre and $z_K$ ring",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_provenance() -> None:
    if not WEBB_FILE.exists():
        return
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    ext = manifest.get("external_inputs", {})
    ext["webb2020_exposure.csv"] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ("Webb (2020), \"The Impact of Artificial Intelligence on the "
                   "Labor Market\" (SSRN 3482150); occupational exposure to AI, "
                   "software, and robots from patent-task text matching, keyed "
                   "by occ1990dd (percentile columns)."),
        "measure": "occupational exposure percentile to robots / software / AI",
        "derived": False,
        "sha256_frozen": _sha256(WEBB_FILE),
    }
    if WEBB_XWALK.exists():
        ext["onet_to_occ1990dd.dta"] = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "onetsoccode <-> occ1990dd crosswalk (Dorn / Webb)",
            "measure": "occupation code crosswalk",
            "derived": False,
            "sha256_frozen": _sha256(WEBB_XWALK),
        }
    manifest["external_inputs"] = ext
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="run on a synthetic corpus (NOT A RESULT)")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    surfaces, tag = build_surfaces(args.smoke)
    if "eloundou" not in surfaces:
        sys.exit("Eloundou occupation surface unavailable; check data/ inputs.")
    have_webb = any(n.startswith("webb_") for n in surfaces)
    if not have_webb and not args.smoke:
        print(f"No Webb file at {WEBB_FILE}. Place it (see AWAITING INPUT in the "
              f"module docstring) or run with --smoke. Skipping the Webb side.")

    if not args.smoke:
        record_provenance()

    occ = load_occupations()
    chi_support = float(occ["chi"].max())
    fits = {n: fit_field(df, chi_support) for n, df in surfaces.items()}

    def _idnote(f: dict) -> str:
        return ("centre at the occupied edge" if f["at_bound"] else "localised")

    lines = ["Webb (2020) field calibration on the occupation substrate.",
             f"  provenance: {tag}",
             "  fit: unweighted Gaussian (scripts/08 model), centre constrained "
             "to the occupied disk; each surface scaled to [0,1] before the fit.",
             "  z_K is reach as a fraction of centre-to-rim (rim chi = 1); "
             f"occupations reach chi = {chi_support:.2f}.", ""]
    hdr = (f"  {'field':<20}{'n':>5}{'xi_K':>9}{'chi_K':>8}{'z_K':>8}"
           f"{'A_K':>8}{'R2_surf':>9}  note")
    lines += [hdr, "  " + "-" * (len(hdr) - 2)]
    for n in FIELD_ORDER:
        if n not in fits:
            continue
        f = fits[n]
        lines.append(f"  {FIELD_LABEL[n]:<20}{f['n']:>5}"
                     f"{f['xi_K_deg']:>8.1f}\u00b0{f['chi_K']:>8.3f}"
                     f"{f['z_K']:>8.3f}{f['A_K']:>8.3f}{f['r2_surface']:>9.3f}"
                     f"  {_idnote(f)}")
    lines.append("")

    if "webb_ai" in fits:
        c = compare(fits["webb_ai"], fits["eloundou"])
        pinned = fits["webb_ai"]["at_bound"]
        lines += [
            "Webb AI vs Eloundou AI (same substrate, like-for-like):",
            f"  angular separation of centres  {c['d_angle_deg']:+.1f} deg",
            f"  radial separation chi_K        {c['d_chi']:+.3f}",
            f"  centre-to-centre distance      {c['d_center']:.3f} disk units",
            f"  reach ratio z_K(Webb)/z_K(Elo) {c['z_ratio']:.2f}",
        ]
        if pinned:
            lines += [
                "  The Webb AI centre pins at the occupied edge. The patent-based "
                "AI surface is a broad",
                "  gradient toward high-skill work with no interior peak, so the "
                "constrained fit sits it on the",
                "  rim of the occupied disk: the two measures place AI exposure "
                "differently. Eloundou gives a",
                "  localised north-east peak, Webb AI does not co-locate with it. "
                "This matches the known weak",
                "  correlation between Webb's patent measure and capability-based "
                "AI measures.",
            ]
        else:
            lines += [
                "  Close centres cross-validate the AI field's location across "
                "two independent exposure",
                "  constructions; the separation above is how far the two "
                "measures agree.",
            ]
        lines.append("")

    for n in ("webb_robot", "webb_software"):
        if n in fits:
            f = fits[n]
            lines.append(
                f"{FIELD_LABEL[n]}: centre xi_K = {f['xi_K_deg']:.1f} deg, "
                f"chi_K = {f['chi_K']:.3f}, z_K = {f['z_K']:.3f}, "
                f"R2_surf = {f['r2_surface']:.2f} ({_idnote(f)}).")
    if "webb_robot" in fits:
        lines += [
            "  The robot field is the data-calibrated location of the "
            "industrial-automation field used",
            "  as the placebo in the identification section, in place of the "
            "hand placement.",
        ]
    lines.append("")

    rows = [{"field": n, **{k: fits[n][k] for k in
             ("n", "xi_K_deg", "chi_K", "z_K", "A_K", "px", "py",
              "r2_task", "r2_surface", "eta2_between_cell", "at_bound")},
             "provenance": tag}
            for n in FIELD_ORDER if n in fits]
    pd.DataFrame(rows).to_csv(RESULTS / "webb_calibration.csv", index=False)

    figure_comparison((occ["x"].to_numpy(), occ["y"].to_numpy()), fits, tag,
                      RESULTS / "webb_field_comparison.png")
    figure_points_rings(surfaces, fits, RESULTS / "webb_points_and_rings.png")

    (RESULTS / "webb_calibration_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {RESULTS / 'webb_calibration.csv'}")
    print(f"wrote {RESULTS / 'webb_field_comparison.png'}")
    print(f"wrote {RESULTS / 'webb_points_and_rings.png'}")
    if args.smoke:
        print("\n*** SMOKE RUN: outputs are synthetic and NOT A RESULT. ***")


if __name__ == "__main__":
    main()
