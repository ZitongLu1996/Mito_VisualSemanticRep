"""Presentation-level redundancy reduction for significant GO BP terms.

Statistical results are not recomputed. Significant terms are greedily grouped
by mapped-gene overlap, and the strongest term in each cluster is retained as
the representative. Clustering is performed separately for UV and US so that
themes remain specific to the functional contrast being presented.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "derivatives" / "ahba_enrichment"
QCOL = "mannwhitney_q_bh_within_library_panel"
PCOL = "mannwhitney_p_two_sided"
JACCARD_THRESHOLD = 0.25
OVERLAP_COEFFICIENT_THRESHOLD = 0.50


def overlap(a, b):
    intersection = len(a & b)
    if not intersection:
        return 0.0, 0.0
    return intersection / len(a | b), intersection / min(len(a), len(b))


def main():
    results = pd.read_csv(OUT / "gene_set_mannwhitney_bh_results.csv.gz")
    inventory = pd.read_csv(OUT / "gene_set_inventory.csv.gz")
    go = results[results.library.eq("GO Biological Process")].copy()
    genes = (inventory[inventory.library.eq("GO Biological Process")]
             .set_index("gene_set").genes.map(lambda x: set(str(x).split(";"))))

    membership_rows = []
    representative_rows = []
    for variance_map, panel_all in go.groupby("variance_map", sort=False):
        significant = panel_all[panel_all.significant_q05]
        term_summary = (significant.groupby("gene_set")
            .agg(best_q=(QCOL, "min"), best_p=(PCOL, "min"),
                 max_abs_effect=("rank_biserial_effect", lambda x: np.abs(x).max()))
            .sort_values(["best_q", "best_p", "max_abs_effect"],
                         ascending=[True, True, False]))

        clusters = []
        for term, stats in term_summary.iterrows():
            strongest = significant[
                significant.gene_set.eq(term)
            ].sort_values([QCOL, PCOL]).iloc[0]
            direction = 1 if strongest.rank_biserial_effect >= 0 else -1
            assigned = None
            best_similarity = (-1.0, -1.0)
            for i, cluster in enumerate(clusters):
                if cluster["direction"] != direction:
                    continue
                jaccard, coefficient = overlap(genes[term], genes[cluster["representative"]])
                if (jaccard >= JACCARD_THRESHOLD or
                        coefficient >= OVERLAP_COEFFICIENT_THRESHOLD):
                    if (jaccard, coefficient) > best_similarity:
                        assigned = i
                        best_similarity = (jaccard, coefficient)
            if assigned is None:
                clusters.append({"representative": term, "direction": direction,
                                 "members": [(term, 1.0, 1.0)]})
            else:
                clusters[assigned]["members"].append(
                    (term, best_similarity[0], best_similarity[1]))

        for cluster_id, cluster in enumerate(clusters, start=1):
            representative = cluster["representative"]
            member_names = [x[0] for x in cluster["members"]]
            for term, jaccard, coefficient in cluster["members"]:
                membership_rows.append({
                    "variance_map": variance_map,
                    "cluster_id": cluster_id,
                    "representative_gene_set": representative,
                    "representative_term_name": representative.split(" | ", 1)[-1],
                    "direction": "positive" if cluster["direction"] > 0 else "negative",
                    "member_gene_set": term,
                    "member_term_name": term.split(" | ", 1)[-1],
                    "jaccard_to_representative": jaccard,
                    "overlap_coefficient_to_representative": coefficient,
                    "n_terms_in_cluster": len(member_names),
                })
            rep = panel_all[panel_all.gene_set.eq(representative)].copy()
            rep["cluster_id"] = cluster_id
            rep["representative_term_name"] = representative.split(" | ", 1)[-1]
            rep["n_terms_in_cluster"] = len(member_names)
            rep["cluster_member_terms"] = ";".join(member_names)
            representative_rows.append(rep)

    membership = pd.DataFrame(membership_rows)
    representatives = pd.concat(representative_rows, ignore_index=True)
    membership.to_csv(OUT / "go_bp_redundancy_clusters.csv.gz", index=False,
                      compression="gzip")
    representatives.to_csv(OUT / "go_bp_representative_results.csv.gz", index=False,
                           compression="gzip")
    print(membership.groupby("variance_map").agg(
        significant_terms=("member_gene_set", "nunique"),
        representative_themes=("cluster_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
