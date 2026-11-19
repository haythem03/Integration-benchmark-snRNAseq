# Integration benchmark (snRNA-seq)

Simple pipeline to compare batch-integration methods on a single-nucleus RNA-seq AnnData object using [scIB](https://scib.readthedocs.io/) metrics.

## What it does

1. Loads your `.h5ad`
2. Runs the methods you enable in `config.yaml`
3. Scores each method with scIB
4. Writes a metrics CSV and a few summary figures

Methods covered: PCA (unintegrated baseline), Harmony, Scanorama, ComBat, BBKNN, scVI, scANVI, scPoli, Seurat v3 (CCA), Seurat v4 (RPCA), CSS Pearson, CSS Spearman.

Overall score = `0.6 * bio + 0.4 * batch`.

## Setup

Python side (scanpy / scvi / etc.):

```bash
pip install scanpy anndata scib harmonypy scanorama bbknn scvi-tools scarches pyyaml seaborn
```

R side (only if you keep Seurat / CSS on):

```r
install.packages(c("Seurat", "Matrix"))
# CSS:
# remotes::install_github("quadbio/simspec")
```

## Usage

1. Edit `config.yaml`
   - set `input_h5ad`, `batch_key`, `label_key`, `counts_layer`
   - flip methods on/off
   - optionally set `max_cells` for a quick dry run
2. Run:

```bash
python run_scib_benchmark.py
# or
python run_scib_benchmark.py --config /path/to/config.yaml
```

## Outputs

Under `output_dir` (default `./results`):

| Path | Content |
|------|---------|
| `scib_metrics.csv` | All metrics + overall score |
| `figures/overall_ranking.png` | Barplot of overall scores |
| `figures/metrics_heatmap.png` | Per-metric heatmap |
| `figures/master_scorecard.png` | Bio / batch / overall scorecard |
| `checkpoints/` | Per-method embedding checkpoints |

## Notes

- Input object needs raw counts in `layers[counts_layer]`, plus batch and cell-type columns in `.obs`.
- scANVI warm-starts from the scVI model, so scVI is run first if needed.
- Existing checkpoints are reused; delete `checkpoints/` if you want a clean rerun.
- Seurat / CSS are handled by `run_r_methods.R` and only run when those methods are enabled.
