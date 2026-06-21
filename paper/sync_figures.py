#!/usr/bin/env python3
"""
sync_figures.py
---------------
Copy the current version of every figure the paper references from the
generating directories into paper/, so the manuscript compiles against the
latest pipeline output. It scans all paper/*.tex (the manuscripts and any
\\input section files), collects every \\includegraphics target, resolves each
against the source directories in priority order, and copies it next to the
.tex.

Source directories (priority order):
    results/        deterministic outputs of the static pipeline (scripts 01-13)
    experiment/     dynamic-paper figures (tempo_*, calendar_*)

Report categories:
    refreshed    a newer/different file was copied into paper/
    up-to-date   paper/ already held identical bytes
    paper-only   referenced, no source-dir copy, but a file already sits in
                 paper/ (a hand-made figure; left untouched)
    MISSING      referenced but found in no source dir and not in paper/

A figure reference that omits its extension is matched against .png then .pdf.
LaTeX comments (% ...) are ignored. Subdirectory references (e.g.
figures/x.png) are preserved under paper/.

Usage:
    python paper/sync_figures.py            # copy
    python paper/sync_figures.py --check     # report only; copy nothing
    python paper/sync_figures.py --strict    # exit non-zero if anything is MISSING
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
SOURCES = [ROOT / "results", ROOT / "experiment"]      # priority order
EXTS = [".png", ".pdf"]                                # tried when a ref omits the extension
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")


def referenced_figures() -> set[str]:
    refs: set[str] = set()
    for tex in sorted(PAPER.glob("*.tex")):
        for line in tex.read_text(errors="ignore").splitlines():
            line = COMMENT_RE.sub("", line)            # drop LaTeX comments
            for m in INCLUDE_RE.finditer(line):
                refs.add(m.group(1).strip())
    return refs


def candidate_names(ref: str) -> list[str]:
    return [ref] if Path(ref).suffix else [ref + e for e in EXTS]


def find_source(name: str) -> Path | None:
    for d in SOURCES:
        p = d / name
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only; copy nothing")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any referenced figure is MISSING")
    args = ap.parse_args()

    refreshed, uptodate, paper_only, missing = [], [], [], []
    for ref in sorted(referenced_figures()):
        src = name = None
        for cand in candidate_names(ref):
            src = find_source(cand)
            if src:
                name = cand
                break
        if not src:
            present = any((PAPER / c).exists() for c in candidate_names(ref))
            (paper_only if present else missing).append(ref)
            continue
        dst = PAPER / name
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            uptodate.append(name)
            continue
        if not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        refreshed.append(f"{name}  <-  {src.relative_to(ROOT)}")

    def block(title: str, items: list[str]) -> None:
        print(f"{title} ({len(items)})")
        for it in items:
            print(f"    {it}")

    block("would refresh" if args.check else "refreshed", refreshed)
    block("up-to-date", uptodate)
    if paper_only:
        block("paper-only (no source dir; left untouched)", paper_only)
    if missing:
        block("MISSING (referenced, no source, not in paper/)", missing)

    if args.strict and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
