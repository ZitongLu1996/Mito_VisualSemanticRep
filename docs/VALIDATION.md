# Validation record

The curated repository was checked before release as follows.

## Numerical tests

`pytest -q` passes four tests covering:

- equivalence of the SVD ridge implementation to scikit-learn Ridge;
- non-negative, sum-to-one convex stacking weights;
- the UV + US + Shared = Joint variance-partition identity; and
- complete, non-overlapping outer-fold validation coverage.

## Notebook integrity

All code cells in the three output-free figure notebooks compile. The public
copies retain the plotting expressions and parameter values from the final
manuscript notebooks. Only narrative comments, analysis-root discovery and the
replacement of the obsolete AHBA derivative path were changed.

## Enrichment equivalence

The consolidated gene-set loader reproduced the final libraries exactly:

- 4,949 GO Biological Process sets;
- 13 adult human cortical cell-type marker sets; and
- 72 MitoCarta 3.0 MitoPathways.

The final enrichment entry point produced 20,136 model-by-map-by-gene-set rows.
Against the frozen manuscript results, the maximum absolute difference was
zero for rank-biserial effect size, two-sided Mann-Whitney p value and
within-library/model/map BH-FDR q value.
