#!/usr/bin/env python3
"""
Run integration methods on an snRNA-seq AnnData, score them with scIB,
and write a metrics table + a few summary plots.

  python run_scib_benchmark.py
  python run_scib_benchmark.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
import seaborn as sns
import yaml

try:
    import scib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scib", "-q"])
    import scib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scib_benchmark")

HERE = Path(__file__).resolve().parent
R_METHODS = {"seurat_v3", "seurat_v4", "css_pearson", "css_spearman"}
BIO = ["ARI", "NMI", "ASW_celltype", "iso_label_F1", "iso_label_ASW", "cLISI"]
BATCH = ["ASW_batch", "graph_connectivity", "iLISI", "kBET", "PCR"]


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    methods = cfg.get("methods") or {}
    cfg["enabled"] = [m for m, on in methods.items() if on]
    if not cfg["enabled"]:
        raise ValueError("No methods enabled in config.yaml")
    return cfg


def ensure_dirs(out: Path) -> Path:
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return ckpt


def save_ckpt(path: Path, emb: np.ndarray, obs: pd.DataFrame, adata_graph, *, keep_X=None):
    if keep_X is not None:
        ckpt = ad.AnnData(X=keep_X, obs=obs.copy())
    else:
        ckpt = ad.AnnData(obs=obs.copy())
    ckpt.obsm["X_emb"] = np.asarray(emb)
    ckpt.obsp["connectivities"] = adata_graph.obsp["connectivities"].copy()
    ckpt.obsp["distances"] = adata_graph.obsp["distances"].copy()
    ckpt.uns["neighbors"] = adata_graph.uns["neighbors"].copy()
    ckpt.write_h5ad(path, compression="gzip")
    del ckpt
    gc.collect()


def prepare_adata(cfg: dict) -> ad.AnnData:
    path = cfg["input_h5ad"]
    log.info("Loading %s", path)
    adata = sc.read_h5ad(path)

    batch_key = cfg["batch_key"]
    label_key = cfg["label_key"]
    layer = cfg.get("counts_layer") or "counts"

    for col in (batch_key, label_key):
        if col not in adata.obs:
            raise KeyError(f"obs['{col}'] missing. Available: {list(adata.obs.columns)}")
    if layer not in adata.layers:
        raise KeyError(f"layers['{layer}'] missing. Available: {list(adata.layers.keys())}")

    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = adata.obs[label_key].astype("category")

    max_cells = cfg.get("max_cells")
    if max_cells is not None and adata.n_obs > int(max_cells):
        log.info("Subsampling %d -> %d cells", adata.n_obs, int(max_cells))
        sc.pp.subsample(adata, n_obs=int(max_cells), random_state=cfg.get("random_state", 42))

    log.info("HVG selection (n_top_genes=%d)", cfg["n_top_genes"])
    sc.pp.highly_variable_genes(
        adata,
        layer=layer,
        flavor="seurat_v3",
        n_top_genes=int(cfg["n_top_genes"]),
        batch_key=batch_key,
        subset=True,
    )
    log.info("Integration input: %s", adata.shape)
    return adata


def run_pca(adata, cfg, ckpt_dir, batch_key):
    a = adata.copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=cfg["n_pcs"], svd_solver="arpack", random_state=cfg["random_state"])
    sc.pp.neighbors(a, n_neighbors=cfg["n_neighbors"], n_pcs=cfg["n_pcs"], random_state=cfg["random_state"])
    save_ckpt(
        ckpt_dir / "checkpoint_pca.h5ad",
        a.obsm["X_pca"],
        a.obs,
        a,
        keep_X=adata.layers[cfg["counts_layer"]].copy(),
    )


def run_harmony(adata, cfg, ckpt_dir, batch_key):
    import harmonypy as hm

    a = adata.copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=cfg["n_pcs"], svd_solver="arpack", random_state=cfg["random_state"])
    ho = hm.run_harmony(
        a.obsm["X_pca"], a.obs, vars_use=[batch_key],
        max_iter_harmony=30, random_state=cfg["random_state"], verbose=False,
    )
    emb = np.asarray(ho.Z_corr)
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_harmony.h5ad", emb, a.obs, a)


def run_scanorama(adata, cfg, ckpt_dir, batch_key):
    import scanorama

    a = adata.copy()
    batches = a.obs[batch_key].cat.categories.tolist()
    parts = [a[a.obs[batch_key] == b].copy() for b in batches]
    scanorama.integrate_scanpy(parts, dimred=cfg["n_pcs"], verbose=False)
    emb_map = {bc: sub.obsm["X_scanorama"][i] for sub in parts for i, bc in enumerate(sub.obs_names)}
    emb = np.vstack([emb_map[bc] for bc in a.obs_names])
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_scanorama.h5ad", emb, a.obs, a)


def run_combat(adata, cfg, ckpt_dir, batch_key):
    a = adata.copy()
    a.X = a.X.toarray().astype(np.float32) if sp.issparse(a.X) else np.asarray(a.X, dtype=np.float32)
    sc.pp.combat(a, key=batch_key)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=cfg["n_pcs"], svd_solver="arpack", random_state=cfg["random_state"])
    sc.pp.neighbors(a, n_neighbors=cfg["n_neighbors"], n_pcs=cfg["n_pcs"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_combat.h5ad", a.obsm["X_pca"], a.obs, a)


def run_bbknn(adata, cfg, ckpt_dir, batch_key):
    import bbknn

    a = adata.copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=cfg["n_pcs"], svd_solver="arpack", random_state=cfg["random_state"])
    bbknn.bbknn(a, batch_key=batch_key, n_pcs=cfg["n_pcs"])
    save_ckpt(ckpt_dir / "checkpoint_bbknn.h5ad", a.obsm["X_pca"], a.obs, a)


def _counts_as_X(adata, layer):
    a = adata.copy()
    x = a.layers[layer].copy()
    a.X = x.tocsr().astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)
    return a


def run_scvi(adata, cfg, ckpt_dir, batch_key):
    import scvi
    import torch

    a = _counts_as_X(adata, cfg["counts_layer"])
    scvi.model.SCVI.setup_anndata(a, layer=None, batch_key=batch_key)
    model = scvi.model.SCVI(a, n_latent=cfg["n_latent"], n_layers=2, n_hidden=256, gene_likelihood="nb")
    use_gpu = torch.cuda.is_available()
    model.train(
        max_epochs=cfg["max_epochs"], batch_size=2048, early_stopping=True,
        accelerator="gpu" if use_gpu else "cpu", devices=1 if use_gpu else "auto",
    )
    model.save(ckpt_dir / "scvi_model", overwrite=True)
    emb = model.get_latent_representation(a)
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_scvi.h5ad", emb, a.obs, a)


def run_scanvi(adata, cfg, ckpt_dir, batch_key):
    import scvi
    import torch

    label_key = cfg["label_key"]
    a = _counts_as_X(adata, cfg["counts_layer"])
    scvi.model.SCVI.setup_anndata(a, layer=None, batch_key=batch_key)
    parent = scvi.model.SCVI.load(ckpt_dir / "scvi_model", adata=a)
    model = scvi.model.SCANVI.from_scvi_model(parent, unlabeled_category="Unknown", labels_key=label_key)
    use_gpu = torch.cuda.is_available()
    model.train(
        max_epochs=cfg["max_epochs"], batch_size=2048, early_stopping=True,
        accelerator="gpu" if use_gpu else "cpu", devices=1 if use_gpu else "auto",
    )
    emb = model.get_latent_representation(a)
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_scanvi.h5ad", emb, a.obs, a)


def run_scpoli(adata, cfg, ckpt_dir, batch_key):
    if not hasattr(ad, "read"):
        ad.read = ad.read_h5ad
    from scarches.models import scPoli

    a = adata.copy()
    model = scPoli(
        adata=a,
        condition_keys=batch_key,
        cell_type_keys=cfg["label_key"],
        hidden_layer_sizes=[int(np.sqrt(a.n_obs))],
        latent_dim=cfg["n_latent"],
        embedding_dims=10,
        recon_loss="nb",
    )
    model.train(
        n_epochs=cfg.get("scpoli_epochs", 100),
        pretraining_epochs=80,
        eta=10,
        use_early_stopping=True,
    )
    emb = np.asarray(model.get_latent(a))
    if emb.shape[0] != a.n_obs and emb.shape[1] == a.n_obs:
        emb = emb.T
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
    save_ckpt(ckpt_dir / "checkpoint_scpoli.h5ad", emb, a.obs, a)


PY_RUNNERS = {
    "pca": run_pca,
    "harmony": run_harmony,
    "scanorama": run_scanorama,
    "combat": run_combat,
    "bbknn": run_bbknn,
    "scvi": run_scvi,
    "scanvi": run_scanvi,
    "scpoli": run_scpoli,
}


def export_for_r(adata, cfg, ckpt_dir: Path):
    layer = cfg["counts_layer"]
    scipy.io.mmwrite(ckpt_dir / "counts.mtx", adata.layers[layer].T)
    adata.obs.to_csv(ckpt_dir / "metadata.csv")
    pd.DataFrame(adata.var_names).to_csv(ckpt_dir / "genes.csv", index=False, header=False)


def run_r_methods(cfg, ckpt_dir: Path, enabled: list[str]):
    r_wanted = [m for m in enabled if m in R_METHODS]
    if not r_wanted:
        return
    rscript = HERE / "run_r_methods.R"
    cmd = ["Rscript", str(rscript), str(ckpt_dir), cfg["batch_key"], ",".join(r_wanted)]
    log.info("Running R methods: %s", r_wanted)
    subprocess.check_call(cmd)

    ref = ad.read_h5ad(ckpt_dir / "checkpoint_pca.h5ad", backed="r")
    obs = ref.obs.copy()
    ref.file.close()

    for method in r_wanted:
        csv_path = ckpt_dir / f"{method}_embeddings.csv"
        if not csv_path.exists():
            log.warning("Missing R embedding for %s", method)
            continue
        emb = pd.read_csv(csv_path, index_col=0).reindex(obs.index).values
        tmp = ad.AnnData(obs=obs.copy())
        tmp.obsm["X_emb"] = emb
        sc.pp.neighbors(tmp, use_rep="X_emb", n_neighbors=cfg["n_neighbors"], random_state=cfg["random_state"])
        save_ckpt(ckpt_dir / f"checkpoint_{method}.h5ad", emb, obs, tmp)
        del tmp
        gc.collect()


def overall_score(row: dict) -> float:
    bio = np.nanmean([row.get(k, np.nan) for k in BIO])
    batch = np.nanmean([row.get(k, np.nan) for k in BATCH])
    return float(0.6 * bio + 0.4 * batch)


def score_method(method: str, ckpt_path: Path, adata_pre: ad.AnnData, cfg: dict) -> dict:
    label_key = cfg["label_key"]
    batch_key = cfg["batch_key"]
    ckpt = ad.read_h5ad(ckpt_path)
    embed = "X_emb"
    r = {"method": method}

    scib.metrics.cluster_optimal_resolution(
        ckpt, label_key=label_key, cluster_key="leiden_scib",
        cluster_function=sc.tl.leiden, metric=scib.metrics.nmi,
    )
    r["ARI"] = scib.me.ari(ckpt, cluster_key="leiden_scib", label_key=label_key)
    r["NMI"] = scib.me.nmi(ckpt, cluster_key="leiden_scib", label_key=label_key)

    if embed in ckpt.obsm:
        r["ASW_celltype"] = scib.me.silhouette(ckpt, label_key=label_key, embed=embed)
        r["iso_label_F1"] = scib.me.isolated_labels_f1(
            ckpt, label_key=label_key, batch_key=batch_key, embed=embed, verbose=False
        )
        r["iso_label_ASW"] = scib.me.isolated_labels_asw(
            ckpt, label_key=label_key, batch_key=batch_key, embed=embed, verbose=False
        )
        r["cLISI"] = scib.me.clisi_graph(
            ckpt, label_key=label_key, type_="embed", use_rep=embed,
            k0=90, subsample=100, scale=True, verbose=False,
        )
        r["ASW_batch"] = scib.me.silhouette_batch(
            ckpt, batch_key=batch_key, label_key=label_key, embed=embed, verbose=False
        )
        r["iLISI"] = scib.me.ilisi_graph(
            ckpt, batch_key=batch_key, type_="embed", use_rep=embed,
            k0=90, subsample=100, scale=True, verbose=False,
        )
    else:
        for k in ["ASW_celltype", "iso_label_F1", "iso_label_ASW", "cLISI", "ASW_batch", "iLISI"]:
            r[k] = np.nan

    r["graph_connectivity"] = scib.me.graph_connectivity(ckpt, label_key=label_key)
    try:
        r["kBET"] = scib.me.kBET(
            ckpt, batch_key=batch_key, label_key=label_key, type_="embed", embed=embed
        )
    except Exception:
        r["kBET"] = np.nan
    try:
        r["PCR"] = scib.me.pcr_comparison(
            adata_pre, ckpt, covariate=batch_key, embed=embed, n_comps=8, scale=True, verbose=False
        )
    except Exception:
        r["PCR"] = np.nan

    r["overall"] = overall_score(r)
    del ckpt
    gc.collect()
    return r


def plot_results(df: pd.DataFrame, out_dir: Path):
    fig_dir = out_dir / "figures"
    plot_df = df.copy()
    plot_df["Bio"] = plot_df[BIO].mean(axis=1)
    plot_df["Batch"] = plot_df[BATCH].mean(axis=1)
    plot_df = plot_df.sort_values("overall", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df.index, plot_df["overall"], color="steelblue", edgecolor="white")
    ax.set_ylabel("Overall scIB score (0.6 bio + 0.4 batch)")
    ax.set_title("Integration ranking")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "overall_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()

    cols = BIO + BATCH
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(plot_df))))
    sns.heatmap(plot_df[cols], annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("scIB metrics")
    plt.tight_layout()
    fig.savefig(fig_dir / "metrics_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    show = plot_df.rename(columns={"overall": "Overall", "Bio": "Bio (60%)", "Batch": "Batch (40%)"})
    blocks = [BIO, BATCH, ["Bio (60%)", "Batch (40%)"], ["Overall"]]
    titles = ["Biological conservation", "Batch correction", "Category", "Overall"]
    cmaps = ["Blues", "Oranges", "Greens", "Reds"]
    fig, axes = plt.subplots(
        1, 4, figsize=(22, max(4, 0.45 * len(show))),
        gridspec_kw={"width_ratios": [6, 5, 2.2, 1.2]},
    )
    plt.subplots_adjust(wspace=0.08)
    for ax, cols_, title, cmap in zip(axes, blocks, titles, cmaps):
        data = show[cols_]
        norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
        sns.heatmap(norm, annot=data, fmt=".3f", cmap=cmap, cbar=False, ax=ax, linewidths=0.4)
        ax.set_title(title, fontweight="bold")
        if ax is not axes[0]:
            ax.set_yticks([])
            ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Integration benchmark scorecard", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(fig_dir / "master_scorecard.png", dpi=200, bbox_inches="tight")
    plt.close()
    log.info("Figures -> %s", fig_dir)


def main():
    parser = argparse.ArgumentParser(description="snRNA-seq integration scIB benchmark")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = ensure_dirs(out_dir)
    enabled = cfg["enabled"]
    batch_key = cfg["batch_key"]

    log.info("Enabled methods: %s", enabled)
    adata = prepare_adata(cfg)

    # PCA is always built (baseline + PCR / R reference)
    if not (ckpt_dir / "checkpoint_pca.h5ad").exists():
        log.info("-- pca --")
        run_pca(adata, cfg, ckpt_dir, batch_key)

    for method in enabled:
        if method in R_METHODS or method == "pca":
            continue
        if method == "scanvi" and "scvi" not in enabled and not (ckpt_dir / "scvi_model").exists():
            log.info("scANVI needs a scVI model — running scVI first")
            run_scvi(adata, cfg, ckpt_dir, batch_key)
        out = ckpt_dir / f"checkpoint_{method}.h5ad"
        if out.exists():
            log.info("SKIP %s (checkpoint exists)", method)
            continue
        if method not in PY_RUNNERS:
            log.warning("Unknown method %s — skip", method)
            continue
        log.info("-- %s --", method)
        try:
            PY_RUNNERS[method](adata, cfg, ckpt_dir, batch_key)
        except Exception:
            log.exception("%s failed", method)
        gc.collect()

    if any(m in enabled for m in R_METHODS):
        export_for_r(adata, cfg, ckpt_dir)
        try:
            run_r_methods(cfg, ckpt_dir, enabled)
        except Exception:
            log.exception("R methods failed")

    adata_pre = ad.read_h5ad(ckpt_dir / "checkpoint_pca.h5ad")
    rows = []
    for method in enabled:
        path = ckpt_dir / f"checkpoint_{method}.h5ad"
        if not path.exists():
            log.warning("No checkpoint for %s — skip metrics", method)
            continue
        log.info("Scoring %s", method)
        try:
            rows.append(score_method(method, path, adata_pre, cfg))
            log.info("  overall=%.3f", rows[-1]["overall"])
        except Exception:
            log.exception("Metrics failed for %s", method)

    if not rows:
        raise SystemExit("No metrics computed — check method failures above.")

    metrics = pd.DataFrame(rows).set_index("method").sort_values("overall", ascending=False)
    csv_path = out_dir / "scib_metrics.csv"
    metrics.to_csv(csv_path)
    log.info("Metrics -> %s", csv_path)
    print("\n" + metrics[["overall"] + BIO + BATCH].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nBest method: {metrics.index[0]} (overall={metrics.iloc[0]['overall']:.3f})")

    plot_results(metrics, out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
