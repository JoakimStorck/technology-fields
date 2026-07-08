"""
recover_projection.py  (infrastructure; not an analysis producer)
-----------------------------------------------------------------
Recovers the frozen encoder->disk projection basis and writes
data/geometry_projection.npz for scripts/21_startup_seeding.py.

Pipeline (geometry-of-work, 2_pca_projection.ipynb): the diagnostic established
SCALER_MODE="none", so no per-task pre-PCA scaling is applied. The whole map is
then linear -- centre on the mean, SVD-project onto the top axes (unweighted_pca),
and an orthogonal Procrustes rotation of (pc1, pc2) to a fixed compass (Cell 8) --
so the composite from the raw embedding to the frozen coordinates is affine:

    [pc1, pc2] = e @ W + b .

We recover (W, b) by least squares from the raw embeddings and the frozen (pc1,
pc2). Because the stored embeddings are float32 and the frozen coordinates were
produced by a float32 SVD, the reproduction floor is ~1e-4, not machine
precision; acceptance is therefore on the MEDIAN residual with the tail
diagnosed (near-origin tasks inflate the angle). A handful of scaler recipes are
also tried and reported, to confirm "none" is best. New text is projected with
the same affine map, pc = X_new @ W + b.

INPUTS -- copy from the reference run's exports/ into --raw-dir
(default data/_geometry_raw/):
    task_embeddings.npy                raw task embeddings (N, D), float32
    task_embeddings_fingerprint.json   row order (Task IDs) for the .npy
    tasks_for_pca_base.csv             rle_mean, rt (only to confirm no scaling)
    run_config.json, radial_scale.json (optional)
Frozen task coordinates: data/task_embeddings_polar_scaled.csv.

OUTPUT:
    data/geometry_projection.npz   W (D,2), b (2,), preprocess, scaler_mode,
                                   radial_scale, source; prints SHA-256 + MANIFEST.

Usage:
    python scripts/recover_projection.py [--raw-dir data/_geometry_raw]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
POLAR = DATA / "task_embeddings_polar_scaled.csv"
OUT = DATA / "geometry_projection.npz"

MED_TOL = 1e-3   # accept if the best recipe's MEDIAN residual is below this
                 # (float32 SVD floor is ~1e-4; a real mismatch would be ~1e-2)
RECIPES = [("none", ""), ("rle", "log"), ("rt", "log")]


def _radial_scale_in_repo() -> float:
    occ = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv",
                      usecols=["r", "chi"])
    k = (occ["chi"] / occ["r"]).to_numpy()
    assert k.std() < 1e-9, "in-repo radial scale is not constant"
    return float(np.median(k))


def _ids_for_npy(raw: Path, n_rows: int):
    fp = raw / "task_embeddings_fingerprint.json"
    if fp.exists():
        d = json.loads(fp.read_text())
        for key in ("task_ids", "ids", "row_ids", "order", "task_id", "Task ID"):
            if key in d and len(d[key]) == n_rows:
                return np.asarray(d[key])
    base = raw / "tasks_for_pca_base.csv"
    if base.exists():
        df = pd.read_csv(base)
        for col in ("Task ID", "task_id", "TaskID"):
            if col in df.columns and len(df) == n_rows:
                return df[col].to_numpy()
    return None


def _scale_factors(values, fn, floor=0.1):
    v = np.where(np.isfinite(np.asarray(values, float)), values, 0.0)
    f = {"linear": v, "log": np.log1p(v),
         "sqrt": np.sqrt(np.maximum(v, 0.0))}[fn]
    f = np.maximum(f, float(floor))
    return f / f.mean() if f.mean() > 0 else f


def _affine_fit(X, pc):
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, pc, rcond=None)
    res = np.abs(A @ coef - pc).max(axis=1)
    return coef[:-1], coef[-1], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DATA / "_geometry_raw"))
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    npy = raw / "task_embeddings.npy"
    if not npy.exists():
        sys.exit(f"missing {npy}. Copy the reference run's exports/ files.")

    E = np.load(npy).astype(np.float64)
    N, D = E.shape
    frozen = pd.read_csv(POLAR, usecols=["Task ID", "pc1", "pc2", "xi", "chi"])
    ids = _ids_for_npy(raw, N)
    if ids is None:
        sys.exit("no Task-ID order for the .npy; copy the fingerprint / base csv.")
    order = pd.Series(np.arange(N), index=pd.Index(ids, name="Task ID"))
    frozen = frozen[frozen["Task ID"].isin(order.index)].copy()
    rows = order.reindex(frozen["Task ID"]).to_numpy()
    E = E[rows]
    pc = frozen[["pc1", "pc2"]].to_numpy()
    chi = frozen["chi"].to_numpy()
    print(f"aligned {len(frozen)}/{N} rows; float32 embeddings; row-norm median "
          f"{np.median(np.linalg.norm(E, axis=1)):.4f}")

    base = pd.read_csv(raw / "tasks_for_pca_base.csv").set_index(
        next(c for c in ("Task ID", "task_id", "TaskID")
             if c in pd.read_csv(raw / "tasks_for_pca_base.csv").columns))

    fits = {}
    print("\nscaler check (affine fit residual, median / max):")
    for mode, fn in RECIPES:
        if mode == "none":
            Xs = E
        elif mode in base.columns:
            Xs = E * _scale_factors(base[mode].reindex(frozen["Task ID"]).to_numpy(),
                                    fn)[:, None]
        else:
            continue
        W, b, res = _affine_fit(Xs, pc)
        fits[(mode, fn)] = (W, b, res)
        tag = mode if mode == "none" else f"{mode}/{fn}"
        print(f"  {tag:10s} median {np.median(res):.2e}  max {res.max():.2e}")

    (mode, fn), (W, b, res) = min(fits.items(), key=lambda kv: np.median(kv[1][2]))
    med = float(np.median(res))
    print(f"\nchosen scaler: {mode}  (median residual {med:.2e})")
    if med > MED_TOL:
        sys.exit(f"best median residual {med:.2e} exceeds {MED_TOL:g}: the "
                 "pipeline used a step not captured here. Paste Cell 1b values.")

    # float32-floor confirmation: are the worst rows near the origin?
    q = np.percentile(res, [50, 90, 99, 100])
    worst = np.argsort(res)[::-1][:max(1, len(res) // 100)]
    print(f"residual pct: p50 {q[0]:.2e}  p90 {q[1]:.2e}  p99 {q[2]:.2e}  "
          f"max {q[3]:.2e}")
    print(f"worst 1% of rows: median chi {np.median(chi[worst]):.3f} "
          f"(overall median chi {np.median(chi):.3f}) -- small chi => the tail "
          "is near-origin angle noise, harmless for placement")

    k = _radial_scale_in_repo()
    pch = (E if mode == "none" else E * _scale_factors(
        base[mode].reindex(frozen["Task ID"]).to_numpy(), fn)[:, None]) @ W + b
    chi_hat = k * np.hypot(pch[:, 0], pch[:, 1])
    res_chi = float(np.median(np.abs(chi_hat - chi)))
    print(f"radius reproduction: median |dchi| {res_chi:.2e}")

    preprocess = "l2" if abs(np.median(np.linalg.norm(E, axis=1)) - 1.0) < 1e-3 \
        else ""
    np.savez(OUT, W=W.astype(np.float64), b=b.astype(np.float64),
             preprocess=np.str_(preprocess), scaler_mode=np.str_(mode),
             scaler_fn=np.str_(fn), radial_scale=np.float64(k),
             source="geometry-of-work reference run; SCALER_MODE=none; affine "
                    "map e->pc recovered by lstsq, float32 floor; new text pc = e @ W + b")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"\nwrote {OUT}  ({OUT.stat().st_size} bytes)\nsha256 {sha}")
    print("\nMANIFEST snippet (external_inputs):")
    print(json.dumps({"geometry_projection.npz": {
        "source": "geometry-of-work reference run exports/task_embeddings.npy",
        "recipe": f"affine lstsq of frozen (pc1,pc2) on raw embeddings; "
                  f"scaler={mode}; float32 floor median {med:.1e}; pc = e @ W + b",
        "sha256": sha}}, indent=2))


if __name__ == "__main__":
    main()
