"""Competitive enrichment of the final AHBA genome-wide gene rankings.

The input score is the signed four-participant group t statistic obtained after
participant-wise donor-fixed-effect and PG1-adjusted gene correlations.  The
complete 16,008-gene universe is used; no gene-level significance threshold is
applied before enrichment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from run_ahba_genomewide_gene_analysis import fdr_bh
from gene_set_libraries import (
    build_go_bp_sets,
    build_hodge_cell_sets,
    build_mitopathway_sets,
)


ROOT = Path(__file__).resolve().parent
GENE_RESULTS = (
    ROOT
    / "derivatives"
    / "ahba_genomewide_gene_analysis"
    / "gene_group_results_100k.csv.gz"
)
OUTPUT = ROOT / "derivatives" / "ahba_enrichment"
VARIANCE_MAPS = ("unique_visual", "unique_semantic")
MINIMUM_GENES = 10
MAXIMUM_GENES = 500


def build_inventory(universe: set[str]) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Build the three manuscript gene-set libraries on one gene universe."""
    go_sets, go_inventory = build_go_bp_sets(universe)
    cell_sets, cell_inventory = build_hodge_cell_sets(universe)
    mito_sets = build_mitopathway_sets(universe)
    mito_inventory = pd.DataFrame(
        {
            "library": "MitoCarta 3.0 MitoPathways",
            "gene_set": list(mito_sets),
            "term_id": list(mito_sets),
            "term_name": list(mito_sets),
            "hierarchy_level": "MitoPathway",
            "n_genes": [len(mito_sets[name]) for name in mito_sets],
            "genes": [";".join(mito_sets[name]) for name in mito_sets],
        }
    )
    libraries = {
        "GO Biological Process": go_sets,
        "Adult human cortical cell types": cell_sets,
        "MitoCarta 3.0 MitoPathways": mito_sets,
    }
    inventory = pd.concat(
        [go_inventory, cell_inventory, mito_inventory], ignore_index=True
    )
    return libraries, inventory


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gene = pd.read_csv(GENE_RESULTS)
    gene = gene[gene["variance_map"].isin(VARIANCE_MAPS)].copy()
    if gene["gene"].nunique() != 16_008:
        raise RuntimeError("Expected the frozen 16,008-gene AHBA universe")

    universe = set(gene["gene"].astype(str).str.upper())
    libraries, inventory = build_inventory(universe)
    inventory.to_csv(
        OUTPUT / "gene_set_inventory.csv.gz", index=False, compression="gzip"
    )

    rows: list[dict[str, object]] = []
    for (analysis, variance_map), panel in gene.groupby(
        ["analysis", "variance_map"], sort=False
    ):
        score = (
            panel.assign(gene=panel["gene"].astype(str).str.upper())
            .set_index("gene")["group_t"]
            .dropna()
        )
        score_index = score.index.to_numpy(str)
        score_values = score.to_numpy(float)
        for library, gene_sets in libraries.items():
            metadata = inventory[inventory["library"].eq(library)].set_index("gene_set")
            for gene_set, listed_members in gene_sets.items():
                members = set(listed_members) & set(score_index)
                if not MINIMUM_GENES <= len(members) <= MAXIMUM_GENES:
                    continue
                member_mask = np.isin(score_index, np.asarray(sorted(members), str))
                x = score_values[member_mask]
                y = score_values[~member_mask]
                test = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                    method="asymptotic",
                    use_continuity=True,
                )
                u = float(test.statistic)
                rank_biserial = 2.0 * u / (len(x) * len(y)) - 1.0
                rows.append(
                    {
                        "library": library,
                        "analysis": analysis,
                        "variance_map": variance_map,
                        "gene_set": gene_set,
                        "term_id": metadata.loc[gene_set, "term_id"],
                        "term_name": metadata.loc[gene_set, "term_name"],
                        "hierarchy_level": metadata.loc[gene_set, "hierarchy_level"],
                        "n_genes": len(x),
                        "n_background_genes": len(y),
                        "mannwhitney_u": u,
                        "rank_biserial_effect": rank_biserial,
                        "median_group_gene_t_members": float(np.median(x)),
                        "median_group_gene_t_background": float(np.median(y)),
                        "mannwhitney_p_two_sided": float(test.pvalue),
                    }
                )

    result = pd.DataFrame(rows)
    q_column = "mannwhitney_q_bh_within_library_panel"
    result[q_column] = result.groupby(
        ["library", "analysis", "variance_map"], group_keys=False
    )["mannwhitney_p_two_sided"].transform(lambda values: fdr_bh(values.to_numpy()))
    result["significant_q05"] = result[q_column] < 0.05
    result = result.sort_values(
        [
            "library",
            "analysis",
            "variance_map",
            q_column,
            "mannwhitney_p_two_sided",
            "gene_set",
        ]
    )
    result.to_csv(
        OUTPUT / "gene_set_mannwhitney_bh_results.csv.gz",
        index=False,
        compression="gzip",
    )

    settings = {
        "gene_level_source": str(GENE_RESULTS),
        "gene_universe": int(gene["gene"].nunique()),
        "variance_maps": list(VARIANCE_MAPS),
        "ranking_statistic": (
            "signed one-sample t statistic across four participant Fisher-z "
            "partial correlations (AHBA donor fixed effects and PG1 adjusted)"
        ),
        "test": "two-sided competitive Mann-Whitney U",
        "effect_size": "rank-biserial correlation",
        "gene_set_size": f"{MINIMUM_GENES}-{MAXIMUM_GENES} mapped genes",
        "fdr": "BH separately within each library, encoding model and variance map",
        "n_gene_sets": {
            library: int(len(sets)) for library, sets in libraries.items()
        },
    }
    (OUTPUT / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(result.groupby(["library", "analysis", "variance_map"])["significant_q05"]
          .agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
