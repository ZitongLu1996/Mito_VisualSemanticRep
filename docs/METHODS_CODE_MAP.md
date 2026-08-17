# Manuscript Methods–code map

| Manuscript component | Primary code |
|---|---|
| NSD participant/image design and shared1000 split | `nsd_full_cortex/prepare_design.py` |
| Three-repeat beta averaging on non-medial-wall fsaverage cortex | `nsd_full_cortex/prepare_encoding_significance_betas.py` |
| DINOv2, CORnet-S, MiniLM and MPNet features | `nsd_full_cortex/extract_features.py` |
| Nested PCA, vertex-wise ridge selection and convex stacking | `nsd_full_cortex/run_encoding.py`, `nsd_full_cortex/run_encoding_significance.py`, `vsvariance/` |
| UV, US and Shared variance partitioning | `vsvariance/analysis.py` |
| Paired test-image bootstrap and bilateral BH-FDR masks | `nsd_full_cortex/bootstrap_joint_encoding.py`, `nsd_full_cortex/make_subject_encoding_masks.py` |
| Variance-map numerical quality control | `nsd_full_cortex/qc_variance_maps.py` |
| MitoD/MRC and PG1 surface preparation | `mitochondrial_analysis/run_mito_variance_analysis.py` |
| Participant-level partial correlations and 100,000 Moran spatial permutations | `nsd_full_cortex/run_subject_masked_mito_analysis.py` |
| AHBA probe/sample preprocessing and subject-specific site matching | `nsd_full_cortex/run_ahba_genomewide_gene_analysis.py` |
| Gene-wise donor+PG1 partial correlations and four-participant spatial inference | `nsd_full_cortex/run_ahba_genomewide_gene_analysis.py` |
| GO BP, adult cortical cell-type and MitoCarta gene-set libraries | `nsd_full_cortex/gene_set_libraries.py` |
| Final complete-ranking Mann–Whitney enrichment | `nsd_full_cortex/run_ahba_mannwhitney_enrichment.py` |
| GO BP display-level redundancy reduction | `nsd_full_cortex/reduce_go_bp_redundancy.py` |
| Manuscript figures | `notebooks/figure1.ipynb`, `notebooks/figure2.ipynb`, `notebooks/figure3.ipynb` |

The repository excludes the discarded joint MitoD–MRC conditional model,
noise-ceiling normalization, whole-cortex spin tests, direct vertex shuffling,
participant sign-flip inference and four-of-four transcriptomic conjunctions.
