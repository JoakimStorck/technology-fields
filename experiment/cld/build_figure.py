#!/usr/bin/env python3
"""Build manuscript Figure 2: the code-audited stock-and-flow diagram.

Thin consumer of the ``stocktake`` package (extracted from this repo's
former ``experiment/cld/``). It audits the dynamic labour-market model in
``run_dynamic.py`` against the hand-declared figure in ``concept_map.toml``
and emits the Forrester-notation figure and audit trail into ``results/``.

The figure is code-audited, not hand-placed: its layout is derived by
stocktake's principled swim-lane engine. An adverse audit exits non-zero,
so a drifted figure fails the pipeline loudly.
"""
import sys
from pathlib import Path

from stocktake import StocktakeError, build

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent

try:
    report = build(
        EXPERIMENT / "run_dynamic.py",
        HERE / "concept_map.toml",
        EXPERIMENT / "results",
        render=True,
    )
except StocktakeError as exc:
    print(f"figure build failed: {exc}", file=sys.stderr)
    sys.exit(1)

print(report.summary())
