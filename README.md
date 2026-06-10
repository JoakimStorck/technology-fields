# technology-fields

Model implementation and empirical tests for **"Technology fields, automation
and task reinstatement"** (Storck & Andersson, working paper). Builds on the
polar task-space geometry of *The polar geometry of work*
([geometry-of-work](https://github.com/JoakimStorck/geometry-of-work)),
which is expected as a sibling checkout:

```
parent/
  geometry-of-work/   # Paper 1: data, coordinates, wage analysis
  technology-fields/  # this repo
```

All scripts accept `--geometry-root PATH` to override the default sibling
location.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/01_wage_field.py` | Litmus test for the price-of-skill field (Paper 3, eq. 1): estimates ln Π(ξ, χ) = m₀ + m₁cos ξ + m₂sin ξ + χ(m₃ + m₄cos ξ + m₅sin ξ) on the Paper 1 Mincer sample (N = 785), with replication of Table 3, second-harmonic sufficiency test, and employment-weighted robustness. |

Outputs are written to `results/`.

## Setup

```
pip install -r requirements.txt
python scripts/01_wage_field.py
```
