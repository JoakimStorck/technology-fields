"""
fetch_startups.py  (infrastructure; not an analysis producer)
-------------------------------------------------------------
Builds data/startups_ycombinator.csv for scripts/21_startup_seeding.py from the
Y Combinator company directory, via the yc-oss daily mirror of YC's public
Algolia index (no key, no HTML scraping). Each record already carries the same
fields the AISE authors used: one_liner, long_description, tags, industries,
batch.

AI and robotics flags follow YC's own tags:
  is_ai        the company carries the "AI" tag  (broaden with --ai-broad to the
               union AI / Generative AI / Machine Learning / Conversational AI /
               AIOps / ML / AI Assistant)
  is_robotics  a robotics tag, or the "Manufacturing and Robotics" industry

text = one_liner + ". " + long_description  (the same product text the AISE
labeller saw). Whitespace collapsed; empty rows dropped.

The corpus is a SUPERSET of the paper's March-2024 set because YC keeps adding
companies (the 2024-2026 AI wave in particular). Exact replication of their ~958
is neither possible nor needed -- the AISE OUTPUT is already vendored
(AISE_occupations_v1.csv); this corpus supplies the descriptions to EMBED. Use
--max-year 2024 --drop-batches "Summer 2024,Fall 2024" to approximate their
vintage.

Source: yc-oss/api (https://yc-oss.github.io/api/companies/all.json), a daily
GitHub Actions mirror of YC's public ycdc_public Algolia data. --source accepts
that URL (default) or a local path to all.json.

Legal: public data, research use. The descriptions are copyrighted text -- embed
and analyse, do not redistribute the raw corpus.

Usage:
    python scripts/fetch_startups.py                       # live yc-oss mirror
    python scripts/fetch_startups.py --source path/all.json
    python scripts/fetch_startups.py --ai-broad --max-year 2024 \
        --drop-batches "Summer 2024,Fall 2024"
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

AI_STRICT = {"ai"}
AI_BROAD = {"ai", "generative ai", "machine learning", "conversational ai",
            "aiops", "ml", "ai assistant", "swarm ai"}
ROBO_TAGS = {"robotics", "medical robotics", "robotic surgery",
             "swarm robotics", "robotic process automation",
             "food service robots and machines"}
ROBO_INDUSTRY = "Manufacturing and Robotics"

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
    ap.add_argument("--ai-broad", action="store_true",
                    help="use the union of AI-ish tags, not just the 'AI' tag")
    ap.add_argument("--max-year", type=int, default=None,
                    help="keep only batches with year <= this (paper vintage)")
    ap.add_argument("--drop-batches", default="",
                    help="comma-separated batch names to exclude")
    args = ap.parse_args()

    ai_set = AI_BROAD if args.ai_broad else AI_STRICT
    drop = {b.strip() for b in args.drop_batches.split(",") if b.strip()}

    companies = _load(args.source)
    print(f"loaded {len(companies)} companies from {args.source}")

    rows = []
    for c in companies:
        tags = {str(t).lower() for t in (c.get("tags") or [])}
        inds = set(c.get("industries") or [])
        is_ai = bool(tags & ai_set)
        is_robo = bool(tags & ROBO_TAGS) or (ROBO_INDUSTRY in inds)
        if not (is_ai or is_robo):
            continue
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
        rows.append({
            "id": c.get("id"), "name": c.get("name"), "batch": batch,
            "status": c.get("status"), "text": text,
            "is_ai": int(is_ai), "is_robotics": int(is_robo),
        })

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


if __name__ == "__main__":
    main()
