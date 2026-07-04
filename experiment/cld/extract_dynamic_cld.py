#!/usr/bin/env python3
"""
Audit and emit the dynamic model's structure figures.

Three extracted layers form the audit trail:
1. code edges: variable dependencies extracted from Python assignments
2. concept edges: code variables mapped to theoretical concepts
3. figure edges: concepts aggregated into manuscript-scale nodes

The figures themselves are hand-declared in concept_map.toml and
code-AUDITED, not code-generated: the compact CLD consists of the manual
figure-level edges, each checked against the extracted graph in
dynamic_figure_edges_audit.csv; the stock-and-flow figure (the manuscript's
transition-accounting figure) is declared in [stockflow], and every edge
must list concept-level witnesses that exist with AST support, or carry
identity = true with its defining mechanism. An unwitnessed edge is a hard
failure, so the figure cannot drift from the code silently.
"""

from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    sign: str = ""
    mechanism: str = ""
    source_type: str = "ast"


def normalise_name(name: str) -> str:
    aliases = {
        "eq.D_o": "D_o",
        "eq.a_task": "a_task",
        "np": "",
        "math": "",
    }
    return aliases.get(name, name)


def node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = node_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr

    if isinstance(node, ast.Subscript):
        return node_name(node.value)

    if isinstance(node, ast.Call):
        return node_name(node.func)

    if isinstance(node, ast.Tuple):
        return None

    return None


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}

    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for element in node.elts:
            names |= target_names(element)
        return names

    if isinstance(node, ast.Subscript):
        base = node_name(node.value)
        return {base} if base else set()

    if isinstance(node, ast.Attribute):
        name = node_name(node)
        return {name} if name else set()

    return set()


def dependency_names(node: ast.AST) -> set[str]:
    names: set[str] = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            name = node_name(child)
            if name:
                names.add(name)
        elif isinstance(child, ast.Subscript):
            name = node_name(child.value)
            if name:
                names.add(name)

    return {normalise_name(n) for n in names if normalise_name(n)}


def extract_code_edges(source_path: Path) -> list[Edge]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    edges: list[Edge] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: set[str] = set()
            for target in node.targets:
                targets |= target_names(target)

            deps = dependency_names(node.value)

        elif isinstance(node, ast.AnnAssign):
            targets = target_names(node.target)
            deps = dependency_names(node.value) if node.value else set()

        elif isinstance(node, ast.AugAssign):
            targets = target_names(node.target)
            deps = dependency_names(node.value)
            deps |= targets

        else:
            continue

        targets = {normalise_name(t) for t in targets if normalise_name(t)}
        deps = {d for d in deps if d not in targets}

        for target in sorted(targets):
            for dep in sorted(deps):
                edges.append(Edge(source=dep, target=target, source_type="ast"))

    return edges


def load_map(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def map_code_to_concepts(edges: list[Edge], config: dict) -> list[Edge]:
    variables = config.get("variables", {})
    mapped: list[Edge] = []

    for edge in edges:
        src = variables.get(edge.source)
        tgt = variables.get(edge.target)

        if src and tgt and src != tgt:
            mapped.append(
                Edge(
                    source=src,
                    target=tgt,
                    sign=edge.sign,
                    mechanism=edge.mechanism,
                    source_type=edge.source_type,
                )
            )

    return collapse_edges(mapped)


def aggregate_to_figure(edges: list[Edge], config: dict) -> list[Edge]:
    aggregation = config.get("aggregation", {})
    mapped: list[Edge] = []

    for edge in edges:
        src = aggregation.get(edge.source)
        tgt = aggregation.get(edge.target)

        if src and tgt and src != tgt:
            mapped.append(
                Edge(
                    source=src,
                    target=tgt,
                    sign=edge.sign,
                    mechanism=edge.mechanism,
                    source_type=edge.source_type,
                )
            )

    return collapse_edges(mapped)


def manual_edges(config: dict, figure_only: bool = False) -> list[Edge]:
    out: list[Edge] = []

    for item in config.get("manual_edges", []):
        if figure_only and not item.get("include_in_figure", False):
            continue

        out.append(
            Edge(
                source=item["from"],
                target=item["to"],
                sign=item.get("sign", ""),
                mechanism=item.get("mechanism", ""),
                source_type="manual",
            )
        )

    return out

def collapse_edges(edges: list[Edge]) -> list[Edge]:
    grouped: dict[tuple[str, str], list[Edge]] = {}

    for edge in edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)

    collapsed: list[Edge] = []

    for (source, target), group in grouped.items():
        manual = [e for e in group if e.source_type == "manual"]
        ast_edges = [e for e in group if e.source_type == "ast"]

        sign = ""
        mechanism = ""

        if manual:
            sign = manual[0].sign
            mechanism = manual[0].mechanism
        else:
            sign = group[0].sign
            mechanism = group[0].mechanism

        if manual and ast_edges:
            source_type = "ast+manual"
        elif manual:
            source_type = "manual"
        elif ast_edges:
            source_type = "ast"
        else:
            source_type = group[0].source_type

        collapsed.append(
            Edge(
                source=source,
                target=target,
                sign=sign,
                mechanism=mechanism,
                source_type=source_type,
            )
        )

    return sorted(collapsed, key=lambda e: (e.source, e.target, e.source_type))

def write_edges(path: Path, edges: list[Edge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "target", "sign", "source_type", "mechanism"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "sign": edge.sign,
                    "source_type": edge.source_type,
                    "mechanism": edge.mechanism,
                }
            )


def dot_label(node: str, labels: dict[str, str]) -> str:
    return labels.get(node, node.replace("_", " ").title())


def write_dot(path: Path, edges: list[Edge], labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = sorted({e.source for e in edges} | {e.target for e in edges})

    lines = [
        "digraph dynamic_cld {",
        "  graph [rankdir=LR, splines=true, overlap=false];",
        '  node [shape=box, style="rounded", fontsize=11];',
        "  edge [fontsize=10];",
        "",
    ]

    for node in nodes:
        label = dot_label(node, labels)
        lines.append(f'  "{node}" [label="{label}"];')

    lines.append("")

    for edge in edges:
        attrs = []
        if edge.sign:
            attrs.append(f'label="{edge.sign}"')
        if edge.sign == "-":
            attrs.append('arrowhead="tee"')
        attr_text = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'  "{edge.source}" -> "{edge.target}"{attr_text};')

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


SHAPE = {
    "stock": ('box', 'rounded,bold'),
    "flow": ('ellipse', ''),
    "regulator": ('box', 'rounded'),
    "source": ('box', 'rounded,dashed'),
    "aux": ('box', 'rounded'),
}


def emit_stockflow(config: dict, concept_edges: list[Edge], out_dir: Path) -> None:
    """Validate the [stockflow] declaration against the extracted concept
    graph and write the dot. Every edge needs AST-supported witnesses or an
    explicit identity marking; otherwise this raises."""
    sf = config.get("stockflow")
    if not sf:
        return

    supported = {(e.source, e.target) for e in concept_edges
                 if "ast" in e.source_type}

    audit_rows: list[dict] = []
    for edge in sf.get("edges", []):
        key = f"{edge['from']}->{edge['to']}"
        if edge.get("identity", False):
            audit_rows.append({"edge": key, "status": "identity",
                               "witnesses": "", "mechanism": edge.get("mechanism", "")})
            continue
        witnesses = edge.get("audit", [])
        if not witnesses:
            raise SystemExit(f"stockflow edge {key} has no audit witnesses "
                             f"and is not marked identity")
        missing = [w for w in witnesses
                   if tuple(w.split("->")) not in supported]
        if missing:
            raise SystemExit(f"stockflow edge {key}: no AST support for "
                             f"witness(es) {missing}; the figure no longer "
                             f"matches the code")
        audit_rows.append({"edge": key, "status": "ast",
                           "witnesses": "; ".join(witnesses),
                           "mechanism": edge.get("mechanism", "")})

    with (out_dir / "stockflow_audit.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["edge", "status",
                                               "witnesses", "mechanism"])
        writer.writeheader()
        writer.writerows(audit_rows)

    lines = [
        f"digraph {sf.get('name', 'stockflow')} {{",
        "  graph [rankdir=LR, splines=true, overlap=false, nodesep=0.45, ranksep=0.55];",
        "  node [fontsize=11, fontname=\"Helvetica\"];",
        "  edge [fontsize=10, fontname=\"Helvetica\"];",
        "",
    ]
    for node in sf.get("nodes", []):
        shape, style = SHAPE[node.get("kind", "aux")]
        style_attr = f', style="{style}"' if style else ""
        lines.append(f'  {node["id"]} [label="{node["label"]}", '
                     f'shape={shape}{style_attr}];')
    lines.append("")
    for edge in sf.get("edges", []):
        attrs = [f'label="{edge.get("sign", "")}"'] if edge.get("sign") else []
        if edge.get("sign") == "-":
            attrs.append('arrowhead="tee"')
        attr_text = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'  {edge["from"]} -> {edge["to"]}{attr_text};')
    lines.append("")
    lines.append("  { rank=same; U; bind; B; }")
    lines.append("}")
    name = sf.get("name", "stockflow")
    (out_dir / f"{name}.dot").write_text("\n".join(lines), encoding="utf-8")
    print(f"Stockflow: {len(audit_rows)} edges audited "
          f"({sum(1 for r in audit_rows if r['status'] == 'ast')} ast, "
          f"{sum(1 for r in audit_rows if r['status'] == 'identity')} identity)")


def try_render_dot(dot_path: Path) -> None:
    pdf_path = dot_path.with_suffix(".pdf")
    png_path = dot_path.with_suffix(".png")

    try:
        subprocess.run(
            ["dot", "-Tpdf", str(dot_path), "-o", str(pdf_path)],
            check=True,
        )
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
        )
    except FileNotFoundError:
        print("Graphviz 'dot' was not found. Wrote .dot file only.")
    except subprocess.CalledProcessError as exc:
        print(f"Graphviz rendering failed: {exc}")

def write_unmapped_report(path: Path, code_edges: list[Edge], config: dict) -> None:
    variables = config.get("variables", {})

    counts = Counter()
    for edge in code_edges:
        counts[edge.source] += 1
        counts[edge.target] += 1

    rows = [
        (name, count)
        for name, count in counts.items()
        if name not in variables
        and not name.startswith("np.")
        and name not in {"self", "dyn", "eq", "layer", "float", "int", "len", "sum", "min", "max", "list", "dict"}
    ]

    rows.sort(key=lambda x: (-x[1], x[0]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "count"])
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(REPO / "experiment" / "run_dynamic.py"),
        help="Path to the dynamic model Python file.",
    )
    parser.add_argument(
        "--map",
        default=str(REPO / "experiment" / "cld" / "concept_map.toml"),
        help="Path to the concept map.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "experiment" / "results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the .dot file to PDF and PNG if Graphviz is installed.",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    map_path = Path(args.map)
    out_dir = Path(args.out)

    config = load_map(map_path)

    code_edges = collapse_edges(extract_code_edges(source_path))

    write_unmapped_report(out_dir / "unmapped_variables.csv", code_edges, config)

    concept_edges_ast = map_code_to_concepts(code_edges, config)
    concept_edges = collapse_edges(concept_edges_ast + manual_edges(config, figure_only=False))

    figure_edges_ast = aggregate_to_figure(concept_edges_ast, config)
    figure_edges_audit = collapse_edges(figure_edges_ast + manual_edges(config, figure_only=True))
    figure_edges = collapse_edges(manual_edges(config, figure_only=True))

    write_edges(out_dir / "dynamic_code_edges.csv", code_edges)
    write_edges(out_dir / "dynamic_concept_edges.csv", concept_edges)
    write_edges(out_dir / "dynamic_figure_edges_audit.csv", figure_edges_audit)
    write_edges(out_dir / "dynamic_figure_edges.csv", figure_edges)

    labels = config.get("figure_nodes", {})
    write_dot(out_dir / "dynamic_figure.dot", figure_edges, labels)

    emit_stockflow(config, concept_edges_ast, out_dir)

    if args.render:
        try_render_dot(out_dir / "dynamic_figure.dot")
        sf = config.get("stockflow")
        if sf:
            try_render_dot(out_dir / f"{sf.get('name', 'stockflow')}.dot")

    print(f"Wrote {len(code_edges)} code edges")
    print(f"Wrote {len(concept_edges)} concept edges")
    print(f"Wrote {len(figure_edges)} figure edges")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
