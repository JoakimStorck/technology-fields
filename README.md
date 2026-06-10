# technology-fields

Model implementation and empirical tests for **"Technology fields, automation
and task reinstatement"** (Storck & Andersson, working paper). Builds on the
polar task-space geometry of *The polar geometry of work*
([geometry-of-work](https://github.com/JoakimStorck/geometry-of-work)).

## Data contract

This repository is **self-contained**: everything required to fit the model
is frozen under `data/`, with provenance (source repository, commit, encoder
run tag, per-file SHA-256, derivation recipes) recorded in
`data/MANIFEST.json`. Analysis scripts read exclusively from `data/`.

The only script that touches a geometry-of-work checkout is
`scripts/00_freeze_inputs.py`, which re-vendors the inputs and rewrites the
manifest. Run it again only when upstream Paper 1 data changes; the manifest
diff then makes the data update explicit in version control.

## Scripts

## Code layout

Reusable model code lives in `model/` (the task-layer objects of the paper:
price field, bundles, technology fields, operated regime); `scripts/` are
numbered, runnable analyses that orchestrate it and write to `results/`.

| Module | Contents |
|---|---|
| `model/data.py` | Frozen-data access: the Mincer sample (N = 785) and task bundles b_o. |
| `model/price_field.py` | The price of skill Pi(r) (eq. 1): construction from estimated coefficients, evaluation, gradient, bundle pricing. |
| `model/technology.py` | Technology fields phi_K (position, reach, amplitude, character), the operated share a(r), displacement D_o. |

| Script | Purpose |
|---|---|
| `scripts/00_freeze_inputs.py` | Vendors required Paper 1 inputs into `data/` and writes `data/MANIFEST.json`. Needs a local geometry-of-work checkout (`--geometry-root`, default: sibling directory). |
| `scripts/01_wage_field.py` | Litmus test for the price-of-skill field (Paper 3, eq. 1): estimates ln Π(ξ, χ) = m₀ + m₁cos ξ + m₂sin ξ + χ(m₃ + m₄cos ξ + m₅sin ξ) on the Paper 1 Mincer sample (N = 785), with replication of Table 3, second-harmonic sufficiency test, and employment-weighted robustness. |
| `scripts/02_price_field.py` | Builds Pi(r) from the estimated coefficients; validates the bundle wage equation against BLS wages (Jensen-gap analysis); demonstrates the operated regime with an illustrative technology (price-ordered takeover, displacement ranking). Produces the price-field and operated-share maps. |

Outputs are written to `results/`.

## Setup

```
pip install -r requirements.txt
python scripts/01_wage_field.py    # estimates the wage field (results/wage_field_*.csv)
python scripts/02_price_field.py   # builds Pi(r), validates bundle wages, regime demo
```
