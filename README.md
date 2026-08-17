# Distinct mitochondrial phenotypes align with visual and semantic representations across human cortex

This repository contains the analysis code accompanying the manuscript
**“Distinct mitochondrial phenotypes align with visual and semantic
representations across human cortex.”** It links participant-specific 7-T fMRI
encoding-model variance maps from the Natural Scenes Dataset (NSD) to the
MitoBrainMap and Allen Human Brain Atlas (AHBA).

## Analysis overview

1. Average the three NSD beta estimates for each image.
2. Reserve the official 1,000 images viewed by all four complete NSD
   participants as a fixed test set; use the remaining participant-specific
   images for training.
3. Fit two parallel image-to-brain encoding analyses:
   DINOv2–MiniLM and CORnet-S–MPNet.
4. Partition held-out predictive variance into unique visual (UV), unique
   semantic (US) and shared components.
5. Define a participant- and model-specific cortical mask from 10,000 paired
   test-image bootstrap samples and bilateral BH-FDR correction.
6. Relate the variance maps separately to MitoD and MRC after adjustment for
   the Margulies principal gradient (PG1), using participant-level effects and
   100,000 mask-specific Moran spectral permutations.
7. Match left-cortical AHBA samples independently to every participant/model
   mask, estimate donor- and PG1-adjusted gene associations, and perform
   four-participant group inference with 100,000 spatial permutations.
8. Test GO Biological Process, adult human cortical cell-type and MitoCarta
   MitoPathway gene sets with complete-ranking competitive Mann–Whitney tests
   and panel-wise BH-FDR correction.

## Repository structure

```text
.
├── config/                    # Frozen manuscript analysis settings
├── docs/                      # Data instructions and Methods–code crosswalk
├── mitochondrial_analysis/   # MitoD/MRC and PG1 surface preparation
├── notebooks/                # Clean, output-free Figure 1–3 notebooks
├── nsd_full_cortex/          # Main data, encoding and statistical scripts
├── tests/                     # Numerical unit tests
├── tools/                     # Caption-table preparation
├── vsvariance/                # Ridge, nested PCA and stacking implementation
├── environment.yml
├── requirements.txt
└── run_pipeline.py
```

See [`docs/METHODS_CODE_MAP.md`](docs/METHODS_CODE_MAP.md) for a direct mapping
between manuscript subsections and scripts. See [`docs/DATA.md`](docs/DATA.md)
for data locations and external resources. Release checks are recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## Installation

The analyses were run in Python. A CUDA-capable GPU is strongly recommended for
feature extraction and the 10,000-image-bootstrap computations.

Using conda:

```bash
conda env create -f environment.yml
conda activate mito-representations
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Connectome Workbench (`wb_command`) is also required by neuromaps surface
transformations and must be available on `PATH`.

## Reproducing the workflow

After placing the external resources described in `docs/DATA.md`, stages can
be run separately:

```bash
python run_pipeline.py --stage data
python run_pipeline.py --stage features --device cuda
python run_pipeline.py --stage encoding
python run_pipeline.py --stage mitochondrial
python run_pipeline.py --stage transcriptomic
python run_pipeline.py --stage enrichment
```

Running all stages is possible with `python run_pipeline.py --stage all`, but
the full workflow downloads a large dataset and includes multiple 100,000-fold
spatial-permutation analyses. Individual scripts are resumable where their
outputs already exist, but the computationally intensive stages should be run
on a workstation or cluster with sufficient memory and storage.

## Statistical scope

- The two feature combinations are parallel robustness analyses and are never
  pooled as independent observations.
- Group inference is based on four participant-level effects, not on a
  participant-averaged cortical map.
- MitoD and MRC are analyzed separately after PG1 adjustment. The repository
  intentionally excludes the discarded joint MitoD–MRC conditional model.
- The primary molecular inference preserves cortical spatial autocorrelation
  through participant- and mask-specific Moran spectral surrogates.
- AHBA sites are selected independently for each participant/model mask; a
  four-participant conjunction mask is not used.
- Ranked enrichment uses the complete 16,008-gene universe. GO term clustering
  is display-level redundancy reduction and does not change p or q values.

## Citation

If you use this code or the accompanying results, please cite the preprint:

> Lu, Z. & Wang, Y. Distinct mitochondrial phenotypes align with visual and
> semantic representations across human cortex. *bioRxiv*
> 2026.08.07.743627 (2026). https://doi.org/10.64898/2026.08.07.743627

```bibtex
@article{lu2026mitochondrial,
  author  = {Lu, Z. and Wang, Y.},
  title   = {Distinct mitochondrial phenotypes align with visual and semantic representations across human cortex},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.07.743627},
  url     = {https://doi.org/10.64898/2026.08.07.743627}
}
```

## Data and code availability

Raw NSD, MitoBrainMap, AHBA, Gene Ontology, Hodge marker and MitoCarta resources
remain subject to their original licenses and are not redistributed here.
Generated derivatives are excluded by `.gitignore`. The exact software and
analysis parameters used for the manuscript are recorded in
`config/analysis_config.json`.
