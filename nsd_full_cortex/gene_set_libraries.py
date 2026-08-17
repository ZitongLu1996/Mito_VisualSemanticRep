"""Build the three gene-set libraries used in the manuscript enrichment."""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "derivatives" / "ahba_functional_enrichment" / "sources"
GO_OBO = SOURCES / "go-basic.obo"
GO_GAF = SOURCES / "goa_human.gaf.gz"
HODGE_XLSX = SOURCES / "hodge2019_supplementary_table2.xlsx"
MITOCARTA_GMX = (
    ROOT.parent / "ahba_analysis" / "mitocarta" / "Human.MitoPathways3.0.gmx"
)
MINIMUM_GENES = 10
MAXIMUM_GENES = 500


def parse_obo() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Read names and parent relationships for GO Biological Process terms."""
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current: dict[str, object] | None = None

    def finish(term: dict[str, object] | None) -> None:
        if not term or term.get("obsolete"):
            return
        term_id = term.get("id")
        if term_id and term.get("namespace") == "biological_process":
            names[str(term_id)] = str(term.get("name", term_id))
            parents[str(term_id)].update(term.get("parents", set()))

    with GO_OBO.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            if line == "[Term]":
                finish(current)
                current = {"parents": set(), "obsolete": False}
                continue
            if line.startswith("[") and line != "[Term]":
                finish(current)
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: GO:"):
                current["id"] = line.split("id: ", 1)[1]
            elif line.startswith("name: "):
                current["name"] = line.split("name: ", 1)[1]
            elif line.startswith("namespace: "):
                current["namespace"] = line.split("namespace: ", 1)[1]
            elif line.startswith("is_a: GO:"):
                current["parents"].add(line.split()[1])
            elif line.startswith("relationship: part_of GO:"):
                current["parents"].add(line.split()[2])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
    finish(current)
    valid = set(names)
    return names, {
        term: {parent for parent in linked if parent in valid}
        for term, linked in parents.items()
        if term in valid
    }


def ancestor_map(parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return all transitive ancestors of each GO term."""
    cache: dict[str, set[str]] = {}

    def visit(term: str, active: set[str]) -> set[str]:
        if term in cache:
            return cache[term]
        if term in active:
            return set()
        output: set[str] = set()
        for parent in parents.get(term, set()):
            output.add(parent)
            output.update(visit(parent, active | {term}))
        cache[term] = output
        return output

    for term in parents:
        visit(term, set())
    return cache


def build_go_bp_sets(
    universe: set[str],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Build propagated GO BP sets mapped to the analysis gene universe."""
    names, parents = parse_obo()
    ancestors = ancestor_map(parents)
    direct: dict[str, set[str]] = defaultdict(set)
    with gzip.open(GO_GAF, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[8] != "P" or "NOT" in fields[3].split("|"):
                continue
            symbol, term = fields[2].strip().upper(), fields[4].strip()
            if symbol in universe and term in names:
                direct[term].add(symbol)

    propagated: dict[str, set[str]] = defaultdict(set)
    for term, genes in direct.items():
        propagated[term].update(genes)
        for parent in ancestors.get(term, set()):
            propagated[parent].update(genes)

    gene_sets: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for term, genes in propagated.items():
        mapped = sorted(genes & universe)
        if MINIMUM_GENES <= len(mapped) <= MAXIMUM_GENES:
            label = f"{term} | {names[term]}"
            gene_sets[label] = mapped
            rows.append(
                {
                    "library": "GO Biological Process",
                    "gene_set": label,
                    "term_id": term,
                    "term_name": names[term],
                    "hierarchy_level": "GO BP",
                    "n_genes": len(mapped),
                    "genes": ";".join(mapped),
                }
            )
    return gene_sets, pd.DataFrame(rows)


def split_markers(value: object) -> set[str]:
    """Parse a marker-gene field from the Hodge supplementary table."""
    if pd.isna(value):
        return set()
    return {
        token.strip().upper()
        for token in re.split(r"[,|]", str(value))
        if token.strip() and token.strip().lower() != "nan"
    }


def build_hodge_cell_sets(
    universe: set[str],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Build adult human cortical cell-type marker sets from Hodge et al."""
    table = pd.read_excel(HODGE_XLSX)
    marker_columns = (
        "single_markers_vs_all",
        "level4_markers_vs_all",
        "combo_markers_vs_all",
    )
    table["_markers"] = [
        set().union(*(split_markers(row[column]) for column in marker_columns))
        for _, row in table.iterrows()
    ]

    candidates: list[dict[str, object]] = []
    for level in ("level1", "level2", "level3"):
        for cell_type, subset in table.groupby(level, sort=True):
            mapped = sorted(set().union(*subset["_markers"].tolist()) & universe)
            if MINIMUM_GENES <= len(mapped) <= MAXIMUM_GENES:
                hierarchy = level.replace("level", "Level ")
                label = f"{hierarchy} | {cell_type}"
                candidates.append(
                    {
                        "library": "Adult human cortical cell types",
                        "gene_set": label,
                        "term_id": label,
                        "term_name": str(cell_type),
                        "hierarchy_level": hierarchy,
                        "n_genes": len(mapped),
                        "genes": ";".join(mapped),
                    }
                )

    # Identical marker sets at adjacent hierarchy levels are tested once,
    # retaining the most specific label.
    deduplicated: dict[str, dict[str, object]] = {}
    for row in candidates:
        signature = str(row["genes"])
        previous = deduplicated.get(signature)
        if previous is None or row["hierarchy_level"] > previous["hierarchy_level"]:
            deduplicated[signature] = row
    rows = sorted(
        deduplicated.values(),
        key=lambda row: (row["hierarchy_level"], row["term_name"]),
    )
    return (
        {row["gene_set"]: str(row["genes"]).split(";") for row in rows},
        pd.DataFrame(rows),
    )


def build_mitopathway_sets(universe: set[str]) -> dict[str, list[str]]:
    """Build MitoCarta 3.0 MitoPathways mapped to the gene universe."""
    table = pd.read_csv(MITOCARTA_GMX, sep="\t", header=None, dtype=str)
    gene_sets: dict[str, list[str]] = {}
    for column in table:
        name = str(table.iloc[0, column]).strip()
        members = set(
            table.iloc[2:, column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
        mapped = sorted((members - {""}) & universe)
        if MINIMUM_GENES <= len(mapped) <= MAXIMUM_GENES:
            gene_sets[name] = mapped
    return gene_sets
