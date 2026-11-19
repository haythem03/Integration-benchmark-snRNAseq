#!/usr/bin/env Rscript
# Seurat / CSS integrations. Called by run_scib_benchmark.py.
# Args: <work_dir> <batch_key> [methods_csv]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_r_methods.R <work_dir> <batch_key> [methods_csv]")
}

work_dir  <- args[[1]]
batch_key <- args[[2]]
wanted <- if (length(args) >= 3) {
  strsplit(args[[3]], ",", fixed = TRUE)[[1]]
} else {
  c("seurat_v3", "seurat_v4", "css_pearson", "css_spearman")
}

setwd(work_dir)
library(Seurat)
library(Matrix)

load_seurat <- function(normalize = FALSE) {
  counts_matrix <- readMM("counts.mtx")
  genes <- read.csv("genes.csv", header = FALSE)$V1
  metadata <- read.csv("metadata.csv", row.names = 1)
  rownames(counts_matrix) <- genes
  colnames(counts_matrix) <- rownames(metadata)
  seu <- CreateSeuratObject(counts = counts_matrix, meta.data = metadata)
  VariableFeatures(seu) <- rownames(seu)
  if (normalize) {
    seu <- NormalizeData(seu, verbose = FALSE)
    seu <- ScaleData(seu, verbose = FALSE)
    seu <- RunPCA(seu, npcs = 50, verbose = FALSE)
  }
  seu
}

need_seurat <- any(c("seurat_v3", "seurat_v4") %in% wanted)
need_css <- any(c("css_pearson", "css_spearman") %in% wanted)

if (need_seurat) {
  cat("Loading Seurat object...\n")
  seu_master <- load_seurat(normalize = FALSE)
  obj.list <- SplitObject(seu_master, split.by = batch_key)

  if ("seurat_v3" %in% wanted) {
    cat("Seurat v3 (CCA)...\n")
    obj.list <- lapply(obj.list, function(x) {
      x <- NormalizeData(x, verbose = FALSE)
      VariableFeatures(x) <- rownames(x)
      x
    })
    anchors <- FindIntegrationAnchors(object.list = obj.list, dims = 1:30, reduction = "cca")
    seu <- IntegrateData(anchorset = anchors, dims = 1:30)
    seu <- ScaleData(seu, verbose = FALSE)
    seu <- RunPCA(seu, npcs = 50, verbose = FALSE)
    write.csv(Embeddings(seu, "pca"), "seurat_v3_embeddings.csv", quote = FALSE)
    rm(anchors, seu); gc()
  }

  if ("seurat_v4" %in% wanted) {
    cat("Seurat v4 (RPCA)...\n")
    obj.list <- lapply(obj.list, function(x) {
      x <- ScaleData(x, verbose = FALSE)
      x <- RunPCA(x, features = rownames(x), verbose = FALSE)
      x
    })
    anchors <- FindIntegrationAnchors(object.list = obj.list, dims = 1:30, reduction = "rpca")
    seu <- IntegrateData(anchorset = anchors, dims = 1:30)
    seu <- ScaleData(seu, verbose = FALSE)
    seu <- RunPCA(seu, npcs = 50, verbose = FALSE)
    write.csv(Embeddings(seu, "pca"), "seurat_v4_embeddings.csv", quote = FALSE)
    rm(anchors, seu); gc()
  }

  rm(obj.list, seu_master); gc()
}

if (need_css) {
  if (!requireNamespace("simspec", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    remotes::install_github("quadbio/simspec")
  }
  library(simspec)

  cat("Loading Seurat object for CSS...\n")
  seu_master <- load_seurat(normalize = TRUE)

  if ("css_pearson" %in% wanted) {
    cat("CSS (Pearson)...\n")
    seu <- cluster_sim_spectrum(seu_master, label_tag = batch_key, corr_method = "pearson")
    write.csv(Embeddings(seu, "css"), "css_pearson_embeddings.csv", quote = FALSE)
    rm(seu); gc()
  }

  if ("css_spearman" %in% wanted) {
    cat("CSS (Spearman)...\n")
    seu <- cluster_sim_spectrum(seu_master, label_tag = batch_key, corr_method = "spearman")
    write.csv(Embeddings(seu, "css"), "css_spearman_embeddings.csv", quote = FALSE)
    rm(seu); gc()
  }
}

cat("R methods done.\n")
