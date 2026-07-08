"""
21_startup_seeding.py
---------------------
Locates the AI startup ecosystem in the occupational task geometry and reads its
position against the model's three fields: the operated share a(r) (where capital
displaces incumbent work), the seeding ring |grad phi_K| (where automation seeds
new task content), and the unbound field u(r) (seeded work no existing occupation
can bind). Two questions:

  1. AISE incidence. The occupation-indexed AISE score (Fenoaltea et al. 2026,
     PNAS Nexus) measures where startup products bite existing occupations. Does
     that revealed targeting coincide with the model's incidence field a(r)?
  2. Startup placement. Where does the startups' own footprint -- the product
     text embedded in the frozen geometry -- sit on the disk relative to a(r),
     the seeding ring, and u(r)?

Q1 is answered here (AISE joined at occupation centroids). Q2 is opened here (the
startups are embedded and placed) and carried by producer 22, which measures the
field enrichment (zeta / u / a) at the startup locations against a disk null --
the quantitative form of "where the startup work sits".

The model side is deterministic and is computed here regardless of the startup
data. On the committed equilibrium (R=18, tau=0.08, gamma=0.5, beta=0.5, ell from
the SD rule) the mass-weighted median distance-to-p_K is ~0.39 for a(r) and ~0.79
for u(r); u(r) sits on/beyond the z_K=0.583 seeding ring; the area-weighted cosine
cos(a,u) ~ 0.24. a(r) tracks exposure (cos(a,phi)~0.86); u(r) tracks the gradient
ring (cos(u,ring)~0.89). The startup side needs two external inputs (see AWAITING
INPUTS); embeddings are cached to results/ so re-runs need no API.

FINDINGS:
  Q1  AISE (revealed occupation-level targeting) coincides with the model
      incidence field a(r): AISE-weighted median distance-to-p_K ~0.42 vs a(r)
      ~0.39, and Spearman(AISE, a at centroid) ~ +0.34. Revealed targeting is a
      core field, distinct from the unbound periphery.
  Q2  The 809 embedded AI startups (and 37 robotics) are written to
      results/startup_seeding_startups.csv; their position relative to the
      seeding ring, the incidence core and u(r) is quantified in producer 22.

AWAITING INPUTS (the startup side stays skipped until both are present):
  data/geometry_projection.npz   the frozen projection basis from the
      geometry-of-work reference run
      (encoder_run_tag in data/MANIFEST.json). The encoder->(pc1,pc2) map is
      affine, so it is recovered by least squares from the raw task embeddings
      and the frozen coordinates (tools/recover_projection.py in
      geometry-of-work) and validated to machine precision. Arrays:
        W          (D, 2)   [pc1, pc2] = preprocess(e) @ W + b   (D=3072)
        b          (2,)     intercept
        preprocess str0     "" or "l2" (per-vector normalisation applied to e
                            before the map; match the encoder_run_tag pipeline)
      atan2(pc2, pc1) and chi = RADIAL_SCALE * hypot(pc1, pc2) then reproduce
      the committed disk. Vendor via the recovery helper and record its SHA-256
      in data/MANIFEST.json.
  data/startups_ycombinator.csv  one row per startup:
        id            unique key
        text          the product/ hiring text to embed (the same text fed to
                      the AISE labeller for a like-for-like comparison; hiring
                      text is the better OWN-WORK signal if available)
        is_ai         1/0 (restrict to AI startups, as AISE does)
        is_robotics   1/0 (optional, for the RSE contrast)
  data/aise_by_onet.csv          optional, enables P1 (Fenoaltea et al. 2026,
      AISE data repo, CC BY 4.0), keyed by full O*NET-SOC:
        onet_code     O*NET-SOC code (e.g. 11-1011.00)
        aise          AI Startup Exposure in [0, 1]
        rse           Robotic Startup Exposure (optional second field)
        job_zone, complementarity  covariates (optional)
  Embedding uses OpenAI text-embedding-3-large (the encoder_run_tag encoder);
  set OPENAI_API_KEY. Run `--smoke` to exercise the pipeline on a synthetic
  corpus (clearly stamped NOT A RESULT) when the real corpus is absent.

Reads: data/ (frozen inputs), results/technology_calibration.csv,
       results/{price,capability}_field_coefficients via model.*.from_results().
Writes:
    results/startup_seeding_model.csv        radial profiles a/u/ring/phi vs dist-to-p_K
    results/startup_seeding_summary.txt
    results/startup_seeding.png              the radial-separation figure (model side)
    results/startup_seeding_startups.csv     (startup side, when inputs present)

Usage:
    python scripts/21_startup_seeding.py            # model side only if corpus absent
    python scripts/21_startup_seeding.py --smoke    # + synthetic-corpus machinery check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.equilibrium import Equilibrium
from model.regime import regime

_spec = importlib.util.spec_from_file_location(
    "_setup", Path(__file__).parent / "_setup.py")
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5

PROJ_FILE = DATA / "geometry_projection.npz"
CORPUS_FILE = DATA / "startups_ycombinator.csv"
AISE_FILE = DATA / "aise_by_onet.csv"


# ─────────────────────────────────────────────────────────────────────
# Frozen radial scale, recovered in-repo (chi = RADIAL_SCALE * r, exact)
# ─────────────────────────────────────────────────────────────────────
def radial_scale() -> float:
    occ = pd.read_csv(DATA / "occupation_embeddings_polar_scaled.csv",
                      usecols=["r", "chi"])
    k = (occ["chi"] / occ["r"]).to_numpy()
    assert k.std() < 1e-9, "radial scale is not constant; frozen files changed"
    return float(np.median(k))


# ─────────────────────────────────────────────────────────────────────
# Grid diagnostics shared by both sides
# ─────────────────────────────────────────────────────────────────────
def mw_median_dist(w, d, area) -> float:
    """Mass-weighted median of a density w over distance d, area element area."""
    w = np.maximum(np.asarray(w, float), 0.0) * area
    order = np.argsort(d)
    cw = np.cumsum(w[order]) / max(w.sum(), 1e-30)
    return float(np.interp(0.5, cw, d[order]))


def cos_aw(p, q, area) -> float:
    p = np.maximum(p, 0.0) * area
    q = np.maximum(q, 0.0) * area
    denom = np.sqrt((p * p).sum() * (q * q).sum())
    return float((p * q).sum() / denom) if denom > 0 else 0.0


def radial_profile(w, d, area, edges):
    bi = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    num = np.bincount(bi, weights=w * area, minlength=len(edges) - 1)
    den = np.bincount(bi, weights=area, minlength=len(edges) - 1)
    return np.where(den > 0, num / den, 0.0)


# ─────────────────────────────────────────────────────────────────────
# Model side: a(r), u(r), the seeding ring, and their radial separation
# ─────────────────────────────────────────────────────────────────────
def model_side():
    inp, L0, occ = _setup.build_inputs()
    tech = _setup.load_tech()
    ell = _setup.interpretable_ell(inp)

    eq = Equilibrium(inp, tech, R, TAU, GAMMA, ell, BETA, wedge=None,
                     survival=True)
    eq.L0 = L0
    _, _, W0 = eq.density_and_value(L0)
    c, kappa, _ = _setup.mobility_reference(W0, eq.d)
    out = eq.solve(c, kappa)
    diag = regime(inp, tech, out.L, R, TAU, GAMMA, ell, BETA, wedge=None,
                  survival=True)

    g = inp.grid
    px, py = tech.p_K
    d = np.hypot(g.x - px, g.y - py)
    a = tech.operated_share(g.xi, g.chi, inp.field, R, TAU)
    phi = tech.phi(g.xi, g.chi)
    ring = tech.grad_phi_norm(g.xi, g.chi)
    u = diag["u"]

    unbound_share = float((u * g.area).sum() / diag["M"])
    # frozen-baseline guard: must reproduce the committed candidate map (68%)
    assert abs(unbound_share - 0.68) < 0.01, (
        f"unbound share {unbound_share:.3f} != committed 0.68; machinery drifted")

    stats = {name: (mw_median_dist(w, d, g.area), float(g.x[np.argmax(w)]),
                    float(g.y[np.argmax(w)]),
                    float(np.hypot(g.x[np.argmax(w)] - px,
                                   g.y[np.argmax(w)] - py)))
             for name, w in [("phi", phi), ("a", a), ("ring", ring), ("u", u)]}

    edges = np.linspace(0, d.max(), 21)
    prof = {name: radial_profile(w, d, g.area, edges)
            for name, w in [("a", a), ("u", u), ("ring", ring), ("phi", phi)]}
    centers = 0.5 * (edges[:-1] + edges[1:])

    model = dict(tech=tech, grid=g, d=d, a=a, u=u, ring=ring, phi=phi,
                 unbound_share=unbound_share, stats=stats, prof=prof,
                 centers=centers, occ=occ, inp=inp, R=R, TAU=TAU)

    pd.DataFrame({"dist_to_pK": centers, **prof}).to_csv(
        RESULTS / "startup_seeding_model.csv", index=False)
    return model


def model_figure(model):
    c = model["centers"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, ls, lab in [("a", "-", "a(r) incidence / target"),
                          ("ring", ":", "|grad phi| seeding ring"),
                          ("u", "-", "u(r) unbound / candidate")]:
        p = model["prof"][name]
        p = p / p.max() if p.max() > 0 else p
        ax.plot(c, p, ls, lw=2, label=lab)
    ax.axvline(model["tech"].z_K, color="0.4", lw=1, ls="--")
    ax.text(model["tech"].z_K, 1.02, "$z_K$", ha="center", fontsize=9, color="0.3")
    ax.set_xlabel("distance from technology centre $p_K$ (disk units)")
    ax.set_ylabel("area-normalised density (peak = 1)")
    ax.set_title("Where work is taken vs where new work is unbound")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "startup_seeding.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Startup side: embed the corpus into the frozen geometry, test P1-P4
# ─────────────────────────────────────────────────────────────────────
def load_projection():
    z = np.load(PROJ_FILE, allow_pickle=True)
    pre = str(z["preprocess"]) if "preprocess" in z else ""
    return z["W"].astype(float), z["b"].astype(float), pre


def _load_dotenv(path=REPO_ROOT / ".env"):
    """Populate os.environ from a local .env (KEY=VALUE lines), without
    overriding vars already set in the environment. No python-dotenv dependency.
    The value is never printed or logged."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def embed_openai(texts, model_name="text-embedding-3-large"):
    """Embed with the encoder_run_tag encoder. Reads OPENAI_API_KEY from the
    environment or a local .env in the repo root."""
    _load_dotenv()
    from openai import OpenAI
    client = OpenAI()
    out = []
    for i in range(0, len(texts), 256):
        chunk = [t.replace("\n", " ") for t in texts[i:i + 256]]
        resp = client.embeddings.create(model=model_name, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, float)


def _embed_cached(texts):
    """Embed, caching to results/ keyed by a hash of the corpus so run_all
    re-runs need no API call. Delete the cache to force a fresh embed."""
    import hashlib
    cache = RESULTS / "startup_embeddings.npz"
    h = hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if str(z["hash"]) == h and int(z["emb"].shape[0]) == len(texts):
                print(f"  [embeddings: cache hit, {len(texts)} vectors from "
                      f"{cache.name}; no API call]")
                return z["emb"]
        except Exception:
            pass
    print(f"  [embeddings: cache miss, calling OpenAI for {len(texts)} texts]")
    emb = embed_openai(texts)
    np.savez(cache, emb=emb.astype(np.float32), hash=np.str_(h))
    return emb


def project_to_disk(emb, W, b, preprocess, k):
    e = np.asarray(emb, float)
    if preprocess == "l2":
        e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
    pc = e @ W + b                              # (n, 2), affine map
    xi = np.arctan2(pc[:, 1], pc[:, 0]) % (2 * np.pi)
    r = np.hypot(pc[:, 0], pc[:, 1])
    chi = k * r
    return xi, chi


def density_on_grid(xi, chi, grid, bw=0.06, weights=None):
    """Area-normalised Gaussian KDE of points (xi, chi) on the disk grid,
    optionally weighting each point (e.g. by its AISE score)."""
    px = chi * np.cos(xi); py = chi * np.sin(xi)
    gx, gy = grid.x[:, None], grid.y[:, None]
    d2 = (gx - px[None, :]) ** 2 + (gy - py[None, :]) ** 2
    k = np.exp(-0.5 * d2 / bw ** 2)
    if weights is not None:
        k = k * np.asarray(weights, float)[None, :]
    kde = k.sum(axis=1)
    Z = (kde * grid.area).sum()
    return kde / Z if Z > 0 else kde


def startup_side(model, smoke=False):
    lines = []
    k = radial_scale()
    grid, d, area = model["grid"], model["d"], model["grid"].area
    px, py = model["tech"].p_K
    have_proj, have_corpus = PROJ_FILE.exists(), CORPUS_FILE.exists()

    if not (have_proj and have_corpus) and not smoke:
        lines += [
            "STARTUP SIDE: SKIPPED (awaiting inputs).",
            f"  projection basis {PROJ_FILE.name}: "
            f"{'present' if have_proj else 'MISSING'}",
            f"  startup corpus   {CORPUS_FILE.name}: "
            f"{'present' if have_corpus else 'MISSING'}",
            "  Vendor both (see module docstring, AWAITING INPUTS), then re-run.",
            "  The model-side predictions above are frozen and independent of "
            "this step.",
        ]
        return lines

    if smoke and not (have_proj and have_corpus):
        # Synthetic machinery check: NOT A RESULT. Random locations, no embedding.
        rng = np.random.default_rng(0)
        n = 400
        xi = rng.uniform(0, 2 * np.pi, n)
        chi = np.clip(rng.normal(model["tech"].chi_K, 0.25, n), 0.02, 1.0)
        lines += ["STARTUP SIDE: SMOKE RUN (synthetic corpus -- NOT A RESULT)."]
    else:
        W, b, pre = load_projection()
        corpus = pd.read_csv(CORPUS_FILE)
        corpus = corpus[corpus.get("is_ai", 1) == 1].copy()
        emb = _embed_cached(corpus["text"].astype(str).tolist())
        xi, chi = project_to_disk(emb, W, b, pre, k)
        corpus["xi"], corpus["chi"] = xi, chi
        # positions file omits the third-party `text` column so it can be
        # tracked (coordinates only); the corpus with text stays out of git
        pos_cols = [c for c in corpus.columns if c != "text"]
        corpus[pos_cols].to_csv(RESULTS / "startup_seeding_startups.csv",
                                index=False)
        lines += [f"STARTUP SIDE: {len(corpus)} AI startups embedded and "
                  f"projected into the frozen geometry."]

    text_field = density_on_grid(xi, chi, grid)
    a, u = model["a"], model["u"]

    med_text = mw_median_dist(text_field, d, area)
    med_a = model["stats"]["a"][0]
    med_u = model["stats"]["u"][0]
    z_K = model["tech"].z_K
    cos_ta = cos_aw(text_field, a, area)
    cos_tu = cos_aw(text_field, u, area)
    text_peak_dist = float(np.hypot(grid.x[np.argmax(text_field)] - px,
                                    grid.y[np.argmax(text_field)] - py))

    lines += [
        f"  mass-weighted median dist-to-p_K:  text {med_text:.3f}  "
        f"a(r) {med_a:.3f}  u(r) {med_u:.3f}  (z_K {z_K:.3f})",
        f"  area-weighted cosine:  cos(text,a) {cos_ta:.3f}  "
        f"cos(text,u) {cos_tu:.3f}",
        f"  text peak dist-to-p_K {text_peak_dist:.3f}",
    ]

    if AISE_FILE.exists():
        occ = model["occ"].reset_index()
        aise = pd.read_csv(AISE_FILE)          # keyed by full O*NET-SOC onet_code
        m = occ.merge(aise, on="onet_code", how="inner")
        wt = m["aise"].to_numpy()
        target_field = density_on_grid(m["xi"].to_numpy(), m["chi"].to_numpy(),
                                       grid, weights=wt)
        med_target = mw_median_dist(target_field, d, area)
        a_at_occ = model["tech"].operated_share(m["xi"].to_numpy(),
                                                m["chi"].to_numpy(),
                                                model["inp"].field, R, TAU)
        rho = pd.Series(wt).corr(pd.Series(a_at_occ), method="spearman")
        p1 = (abs(med_target - med_a) <= 0.15) and (rho >= 0.3)
        lines += [
            f"  AISE median dist-to-p_K {med_target:.3f}; "
            f"Spearman(AISE, a at centroid) {rho:+.3f}",
            f"  Q1 AISE-is-incidence: {'yes' if p1 else 'no'}",
        ]
        offset_ok = (med_text - med_target) >= 0.5 * z_K
        lines += [f"  startup text vs AISE, radial offset "
                  f"{med_text - med_target:+.3f} (z_K {z_K:.3f})"]
    else:
        offset_ok = (med_text - med_a) >= 0.5 * z_K
        lines += [f"  startup text vs a(r), radial offset "
                  f"{med_text - med_a:+.3f} (z_K {z_K:.3f})"]

    lines += [
        f"  cos gap (u - a) {cos_tu - cos_ta:+.3f};  "
        f"text peak dist-to-p_K {text_peak_dist:.3f}",
        "  Field enrichment at the startup locations is quantified in "
        "producer 22.",
    ]
    if smoke and not (have_proj and have_corpus):
        lines += ["  (SMOKE: synthetic points; positions not a result.)"]
    return lines


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="exercise the startup pipeline on a synthetic corpus")
    args = ap.parse_args()

    model = model_side()
    model_figure(model)
    s = model["stats"]

    L = [
        "Startup seeding experiment (see module docstring).",
        f"  calibrated field: xi_K {np.degrees(model['tech'].xi_K):.1f} deg, "
        f"chi_K {model['tech'].chi_K:.3f}, z_K {model['tech'].z_K:.3f}, "
        f"A_K {model['tech'].A_K:.3f}",
        f"  economy R {R}, tau {TAU}, gamma {GAMMA}, beta {BETA}; "
        f"grid {model['grid'].xi.size} cells",
        f"  unbound share {model['unbound_share']:.3f} "
        f"(reproduces committed 0.68)",
        "",
        "MODEL FIELDS (frozen):",
        f"  mass-weighted median dist-to-p_K:  "
        f"phi {s['phi'][0]:.3f}  a(r) {s['a'][0]:.3f}  "
        f"ring {s['ring'][0]:.3f}  u(r) {s['u'][0]:.3f}  "
        f"(z_K {model['tech'].z_K:.3f})",
        f"  radial separation u - a = {s['u'][0] - s['a'][0]:+.3f} "
        f"(~{(s['u'][0] - s['a'][0]) / model['tech'].z_K:.2f} z_K)",
        f"  area-weighted cosine  cos(a,u) "
        f"{cos_aw(model['a'], model['u'], model['grid'].area):.3f}  "
        f"cos(a,phi) {cos_aw(model['a'], model['phi'], model['grid'].area):.3f}  "
        f"cos(u,ring) {cos_aw(model['u'], model['ring'], model['grid'].area):.3f}",
        "  a(r) is a core field on the technology; u(r) is peripheral, on/beyond "
        "the z_K ring.",
        "  The startups' placement relative to these fields is quantified in "
        "producer 22.",
        "",
    ]
    L += startup_side(model, smoke=args.smoke)

    txt = "\n".join(L) + "\n"
    (RESULTS / "startup_seeding_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
