# Data and external resources

This repository intentionally excludes raw neuroimaging data, stimulus images,
postmortem expression matrices, pretrained model weights and large derivatives.
The following resources are required to reproduce the analyses.

## Natural Scenes Dataset

The downloader in `nsd_full_cortex/download_nsd.py` retrieves the public NSD
experiment metadata, fsaverage single-trial beta files and stimulus HDF5 file
for `subj01`, `subj02`, `subj05` and `subj07` from the official public S3
bucket. The complete download is large (hundreds of beta files); verify local
storage before running it.

Expected data root after download:

```text
nsd_full_cortex/data/
├── nsddata/experiments/nsd/
├── nsddata_betas/ppdata/subjXX/fsaverage/betas_fithrf_GLMdenoise_RR/
└── nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
```

NSD: <https://naturalscenesdataset.org/>

## COCO captions

Download the official COCO 2017 annotation archive and provide
`captions_train2017.json` and `captions_val2017.json`. Build the five-caption
mapping used by the encoding models with:

```bash
python tools/prepare_nsd_captions.py \
  --nsd-stim-info nsd_full_cortex/data/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  --coco-train-captions /path/to/captions_train2017.json \
  --coco-val-captions /path/to/captions_val2017.json \
  --output-json metadata/nsd_captions.json
```

COCO annotations: <https://cocodataset.org/#download>

## MitoBrainMap

Obtain the brain-wide MitoD and MRC NIfTI maps released with Mosharov et al.
(2025) and place them at:

```text
mitochondrial_analysis/source_maps/mosharov2025/MitoD.nii.gz
mitochondrial_analysis/source_maps/mosharov2025/MRC.nii.gz
```

The maps are available through the source article and NeuroVault collection
16418: <https://neurovault.org/collections/16418/>.

`mitochondrial_analysis/run_mito_variance_analysis.py --prepare-surfaces`
also retrieves the Margulies principal gradient through neuromaps and projects
PG1, MitoD and MRC to fsaverage10k.

## Allen Human Brain Atlas

`run_ahba_genomewide_gene_analysis.py` uses `abagen.datasets.fetch_microarray`
to obtain microarray data for all six AHBA donors. The script expects writable
locations under:

```text
ahba_analysis/abagen_data/
ahba_analysis/atlas_data/
```

AHBA: <https://human.brain-map.org/>

The MNI152-to-fsaverage10k regfusion coordinates required for sample-to-surface
matching should be available at:

```text
ahba_analysis/atlas_data/atlases/regfusion/
tpl-MNI152_space-fsaverage_den-10k_hemi-L_regfusion.txt
```

They are distributed by neuromaps.

## Gene-set resources

The combined enrichment script builds its libraries from these files:

```text
nsd_full_cortex/derivatives/ahba_functional_enrichment/sources/go-basic.obo
nsd_full_cortex/derivatives/ahba_functional_enrichment/sources/goa_human.gaf.gz
nsd_full_cortex/derivatives/ahba_functional_enrichment/sources/hodge2019_supplementary_table2.xlsx
ahba_analysis/mitocarta/Human.MitoPathways3.0.gmx
```

- Gene Ontology: <https://geneontology.org/docs/download-ontology/>
- GO human annotations: <https://current.geneontology.org/products/pages/downloads.html>
- Hodge et al. adult human MTG markers: Supplementary Table 2 of the source study
- MitoCarta 3.0 MitoPathways: <https://www.broadinstitute.org/mitocarta/mitocarta30-inventory-mammalian-mitochondrial-proteins-and-pathways>

Consult each source's license and terms before redistributing derived or source
files.

