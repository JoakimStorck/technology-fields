"""
fetch_startups.py  (infrastructure; not an analysis producer)
-------------------------------------------------------------
Builds data/startups_ycombinator.csv for scripts/21_startup_seeding.py from the
Y Combinator company directory, via the yc-oss daily mirror of YC's public
Algolia index (no key, no HTML scraping). Each record carries the same fields
Fenoaltea et al. (2026) use to build AISE/RSE: one_liner, long_description,
tags, batch.

AI and robotics flags use the SAME YC tag sets as Fenoaltea et al. (2026), so
the corpus is directly comparable to their AISE and RSE populations:
  is_ai        any of the 13 AI tags: AI, Artificial Intelligence, AI Assistant,
               AI-Powered Drug Discovery, AIOps, Conversational AI, ML, Machine
               Learning, Deep Learning, Deepfake Detection, Generative AI,
               AI-Enhanced Learning, Computer Vision.
  is_robotics  any of the 5 robotics tags: Robotics, Robotic Process Automation,
               Food Service Robots & Machines, Medical Robotics, Robotic Surgery.
               Tags only, as in their RSE; the AI overlap is observed, not
               imposed (they take all robotics-tagged and note most also carry an
               AI tag). No industry filter.

text = one_liner + ". " + long_description  (detailed descriptions, as in their
main analysis). Whitespace collapsed; empty rows dropped.

The corpus is a superset of their March-2024 population because YC keeps adding
companies; the tag DEFINITIONS are identical, so the method section can state
the population is constructed as in Fenoaltea et al. --max-year / --drop-batches
reproduce their vintage if wanted; the default is the latest directory.

Source: yc-oss/api (https://yc-oss.github.io/api/companies/all.json), a daily
GitHub Actions mirror of YC's public ycdc_public Algolia data. --source accepts
that URL (default) or a local path to all.json.

Legal: public data, research use. The descriptions are copyrighted text -- embed
and analyse, do not redistribute the raw corpus.

Usage:
    python scripts/fetch_startups.py                       # live yc-oss mirror
    python scripts/fetch_startups.py --source path/all.json
    python scripts/fetch_startups.py --max-year 2024 \
        --drop-batches "Summer 2024,Fall 2024"             # their vintage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "startups_ycombinator.csv"
LIVE = "https://yc-oss.github.io/api/companies/all.json"

AI_TAGS = {"ai", "artificial intelligence", "ai assistant",
           "ai-powered drug discovery", "aiops", "conversational ai", "ml",
           "machine learning", "deep learning", "deepfake detection",
           "generative ai", "ai-enhanced learning", "computer vision"}
ROBO_TAGS = {"robotics", "robotic process automation",
             "food service robots & machines", "medical robotics",
             "robotic surgery"}

_WS = re.compile(r"\s+")


def _load(source: str):
    if re.match(r"^https?://", source):
        with urllib.request.urlopen(source, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _batch_year(batch: str):
    for tok in (batch or "").split():
        if tok.isdigit():
            return int(tok)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=LIVE)
    ap.add_argument("--max-year", type=int, default=None,
                    help="keep only batches with year <= this (their vintage)")
    ap.add_argument("--drop-batches", default="",
                    help="comma-separated batch names to exclude")
    ap.add_argument("--control-n", type=int, default=0,
                    help="also write data/startups_control.csv with this many "
                         "companies carrying NEITHER tag set (seeded sample); "
                         "the empirical null for producer 22")
    ap.add_argument("--seed", type=int, default=22,
                    help="sampling seed for --control-n")
    args = ap.parse_args()

    drop = {b.strip() for b in args.drop_batches.split(",") if b.strip()}

    companies = _load(args.source)
    print(f"loaded {len(companies)} companies from {args.source}")

    rows, ctrl_rows = [], []
    for c in companies:
        tags = {str(t).lower() for t in (c.get("tags") or [])}
        is_ai = bool(tags & AI_TAGS)
        is_robo = bool(tags & ROBO_TAGS)       # tags only, as in Fenoaltea RSE
        batch = c.get("batch") or ""
        if batch in drop:
            continue
        if args.max_year is not None:
            y = _batch_year(batch)
            if y is not None and y > args.max_year:
                continue
        one_liner = (c.get("one_liner") or "").rstrip()
        for suffix in ("\u2026", "..."):
            if one_liner.endswith(suffix):
                one_liner = one_liner[:-len(suffix)].rstrip()
                break
        text = _WS.sub(" ", f"{one_liner}. "
                             f"{c.get('long_description') or ''}").strip(" .")
        if len(text) < 20:
            continue
        rec = {
            "id": c.get("id"), "name": c.get("name"), "batch": batch,
            "status": c.get("status"), "text": text,
            "is_ai": int(is_ai), "is_robotics": int(is_robo),
        }
        if is_ai or is_robo:
            rows.append(rec)
        else:
            ctrl_rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")
    print(f"  AI: {int(df['is_ai'].sum())}   robotics: "
          f"{int(df['is_robotics'].sum())}   both: "
          f"{int(((df['is_ai'] == 1) & (df['is_robotics'] == 1)).sum())}")
    print(f"  median text length {int(df['text'].str.len().median())} chars; "
          f"batch year range "
          f"{min(filter(None,(_batch_year(b) for b in df['batch'])), default='?')}"
          f"-{max(filter(None,(_batch_year(b) for b in df['batch'])), default='?')}")

    if args.control_n > 0:
        ctrl = pd.DataFrame(ctrl_rows).drop_duplicates(subset="id")
        if len(ctrl) > args.control_n:
            ctrl = ctrl.sample(args.control_n, random_state=args.seed)
        ctrl_out = OUT.parent / "startups_control.csv"
        ctrl.to_csv(ctrl_out, index=False)
        print(f"wrote {ctrl_out}  ({len(ctrl)} control rows, neither tag "
              f"set; seed {args.seed})")


if __name__ == "__main__":
    main()
