#!/usr/bin/env python3


from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_GPUS = [1, 4, 5, 6, 7, 8, 9]

DATASETS = {
    "iemocap": {
        "config":  ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
        "script":  ROOT / "scripts/train/train_iemocap.py",
        "ckpt_dir": ROOT / "experiments/results/iemocap/chado_disent",
        "has_seed": True,
        "mode": "trimodal",
    },
    "mosei": {
        "config":  ROOT / "configs/mosei/chado_mosei_best.yaml",
        "script":  ROOT / "scripts/train/train_mosei.py",
        "ckpt_dir": ROOT / "experiments/results/mosei/chado_disent",
        "has_seed": False,
        "mode": "feature",
    },
    "meld": {
        "config":  ROOT / "configs/meld/chado_meld_final.yaml",
        "script":  ROOT / "scripts/train/train_meld.py",
        "ckpt_dir": ROOT / "experiments/results/meld/chado_disent",
        "has_seed": True,
        "mode": "trimodal",
    },
}

FACTOR_NAMES  = ["z_c\n(Context)", "z_u\n(Individual)", "z_t\n(Temporal)", "z_m\n(Modality)", "z_e\n(Residual)"]
FACTOR_KEYS   = ["z_c", "z_u", "z_t", "z_m", "z_e"]

FIG_DIR = ROOT / "experiments/figures/disentanglement"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs/analysis"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Train (if checkpoint missing)
# ─────────────────────────────────────────────────────────────────────────────

def free_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            idx, free = line.split(",")
            if int(idx.strip()) in ALLOWED_GPUS and int(free.strip()) >= 30_000:
                return int(idx.strip())
    except Exception:
        pass
    return None


def train_dataset(ds_name: str, info: dict, gpu: int):
    ckpt_dir = info["ckpt_dir"]
    best_pt  = ckpt_dir / "best.pt"

    if best_pt.exists():
        log(f"  [{ds_name}] checkpoint exists → skip training")
        return True

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"disent_train_{ds_name}.log"

    cmd = [
        "python3", "-u", str(info["script"]),
        "--config", str(info["config"]),
        "--out-dir", str(ckpt_dir),
    ]
    if info["has_seed"]:
        cmd += ["--seed", str(SEED)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]              = str(gpu)
    env["TOKENIZERS_PARALLELISM"]            = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"]           = "expandable_segments:True"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    log(f"  [{ds_name}] TRAIN  GPU={gpu}  log={log_path.name}")
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
    return proc


def phase1_train():
    """Launch training for all 3 datasets in parallel on separate GPUs."""
    log("\n=== Phase 1: Training CHADO for each dataset ===")

    # Pre-assign one GPU per dataset (avoid re-checking a GPU right after launch)
    used_gpus: set[int] = set()
    running: dict[str, tuple] = {}

    for ds_name, info in DATASETS.items():
        if (info["ckpt_dir"] / "best.pt").exists():
            log(f"  [{ds_name}] checkpoint exists → skip training")
            continue
        # pick first free GPU not already assigned
        gpu = None
        for _ in range(30):           # wait up to 30 min
            for g in ALLOWED_GPUS:
                if g in used_gpus:
                    continue
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", f"--id={g}",
                         "--query-gpu=memory.free",
                         "--format=csv,noheader,nounits"], text=True)
                    if int(out.strip()) >= 30_000:
                        gpu = g
                        break
                except Exception:
                    pass
            if gpu is not None:
                break
            log(f"  [{ds_name}] waiting for a free GPU ...")
            time.sleep(60)

        if gpu is None:
            log(f"  [{ds_name}] ERROR: no free GPU — skipping")
            continue

        used_gpus.add(gpu)
        result = train_dataset(ds_name, info, gpu)
        if result is not True:
            running[ds_name] = (result, gpu, time.time())
        time.sleep(5)   # brief stagger before launching next

    # Wait for all training jobs to complete
    while running:
        for ds_name in list(running):
            proc, g, t0 = running[ds_name]
            ret = proc.poll()
            if ret is None:
                continue
            ela = (time.time() - t0) / 60
            if ret == 0:
                log(f"  [{ds_name}] DONE  {ela:.1f}min")
            else:
                log(f"  [{ds_name}] FAILED  exit={ret}  {ela:.1f}min")
            del running[ds_name]
        if running:
            log(f"  Waiting: " + ", ".join(
                f"{n} GPU={g} {(time.time()-t0)/60:.1f}min"
                for n, (_, g, t0) in running.items()))
            time.sleep(60)

    log("  Phase 1 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def build_iemocap_model(cfg: dict, device: torch.device):
    from models.chado.model import CHADOTrimodal
    mc = cfg["model"]
    cc = cfg.get("chado", {})
    model = CHADOTrimodal(
        text_model_name  = mc["text_model_name"],
        audio_model_name = mc["audio_model_name"],
        video_model_name = mc["video_model_name"],
        num_classes      = cfg["data"]["num_classes"],
        proj_dim         = mc.get("proj_dim", 256),
        dropout          = mc.get("dropout", 0.2),
        use_text         = mc.get("use_text", True),
        use_audio        = mc.get("use_audio", True),
        use_video        = mc.get("use_video", True),
        backbone         = mc.get("backbone", "ctnet"),
        n_heads          = mc.get("n_heads", 4),
        n_speakers       = mc.get("n_speakers", 10),
        use_causal       = cc.get("use_causal", True),
        use_hyperbolic   = cc.get("use_hyperbolic", True),
        use_ot           = cc.get("use_ot", True),
        use_mad          = cc.get("use_mad", True),
    )
    return model.to(device)


def build_meld_model(cfg: dict, device: torch.device):
    from models.chado.model import CHADOTrimodal
    mc = cfg["model"]
    cc = cfg.get("chado", {})
    model = CHADOTrimodal(
        text_model_name  = mc["text_model_name"],
        audio_model_name = mc["audio_model_name"],
        video_model_name = mc["video_model_name"],
        num_classes      = cfg["data"]["num_classes"],
        proj_dim         = mc.get("proj_dim", 256),
        dropout          = mc.get("dropout", 0.2),
        use_text         = mc.get("use_text", True),
        use_audio        = mc.get("use_audio", True),
        use_video        = mc.get("use_video", True),
        backbone         = mc.get("backbone", "ctnet"),
        n_heads          = mc.get("n_heads", 4),
        n_speakers       = mc.get("n_speakers", 7),
        use_causal       = cc.get("use_causal", True),
        use_hyperbolic   = cc.get("use_hyperbolic", True),
        use_ot           = cc.get("use_ot", True),
        use_mad          = cc.get("use_mad", True),
    )
    return model.to(device)


def build_mosei_model(cfg: dict, device: torch.device):
    from models.chado.model import CHADOFeature
    mc = cfg["model"]
    cc = cfg.get("chado", {})
    model = CHADOFeature(
        num_classes  = cfg["data"]["num_classes"],
        d_model      = mc.get("d_model", 256),
        use_audio    = mc.get("use_audio", True),
        use_video    = mc.get("use_video", True),
        text_model   = mc.get("text_model", "roberta-base"),
        use_causal   = cc.get("use_causal", True),
        use_hyperbolic = cc.get("use_hyperbolic", True),
        use_ot       = cc.get("use_ot", True),
        use_mad      = cc.get("use_mad", True),
    )
    return model.to(device)


def load_model(ds_name: str, info: dict, device: torch.device):
    cfg = yaml.safe_load(open(info["config"]))
    if ds_name == "iemocap":
        model = build_iemocap_model(cfg, device)
    elif ds_name == "meld":
        model = build_meld_model(cfg, device)
    else:
        model = build_mosei_model(cfg, device)

    from training.engine.trainer import load_checkpoint
    best_pt = info["ckpt_dir"] / "best.pt"
    load_checkpoint(model, str(best_pt))
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Extract factors from test set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_factors_iemocap(model, cfg: dict, device: torch.device):
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    from torch.utils.data import DataLoader
    dc = cfg["data"]
    mc = cfg["model"]
    test_ds = IEMOCAPDataset(
        csv_path         = dc["test_csv"],
        text_model_name  = mc["text_model_name"],
        image_model_name = mc["video_model_name"],
        max_text_len     = dc.get("max_text_len", 96),
        audio_sec        = dc.get("max_audio_seconds", 4.0),
        n_frames         = dc.get("num_frames", 8),
        use_audio        = mc.get("use_audio", True),
        use_video        = mc.get("use_video", True),
    )
    loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_iemocap)
    return _collect_factors_trimodal(model, loader, device, mc)


@torch.no_grad()
def extract_factors_meld(model, cfg: dict, device: torch.device):
    import functools
    from datasets.meld.meld_dataset import (
        MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    dc = cfg["data"]
    mc = cfg["model"]
    label_map = build_label_map_from_order(EMO_ORDER_7)
    tokenizer = AutoTokenizer.from_pretrained(mc["text_model_name"], use_fast=True)
    collate_fn = functools.partial(
        collate_meld, tokenizer=tokenizer,
        use_text=mc.get("use_text", True),
        use_audio=mc.get("use_audio", True),
        use_video=mc.get("use_video", True),
    )
    test_ds = MeldDataset(
        csv_path       = dc["test_csv"],
        label_map      = label_map,
        text_col       = dc["text_col"],
        audio_path_col = dc["audio_path_col"],
        video_path_col = dc["video_path_col"],
        label_col      = dc["label_col"],
        utt_id_col     = dc.get("utt_id_col", "utt_id"),
        text_model_name= mc["text_model_name"],
        num_frames     = dc.get("num_frames", 8),
        frame_size     = dc.get("frame_size", 224),
        sample_rate    = dc.get("sample_rate", 16000),
        max_audio_seconds = dc.get("max_audio_seconds", 6.0),
        use_text       = mc.get("use_text", True),
        use_audio      = mc.get("use_audio", True),
        use_video      = mc.get("use_video", True),
        context_turns  = dc.get("context_turns", 3),
        speaker_csv    = dc.get("speaker_csv", None),
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    return _collect_factors_trimodal(model, loader, device, mc)


@torch.no_grad()
def _collect_factors_trimodal(model, loader, device, mc):
    factors_all = {k: [] for k in FACTOR_KEYS}
    for batch in loader:
        iids = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        wav = batch["wav"].to(device) if mc.get("use_audio", True) and "wav" in batch else None
        pix = batch["pixel_values"].to(device) if mc.get("use_video", True) and "pixel_values" in batch else None
        _, _, aux = model(
            text_input={"input_ids": iids, "attention_mask": amask},
            audio_wave=wav, video_frames=pix,
        )
        if "factors" not in aux:
            continue
        z_c, z_u, z_t, z_m, z_e = aux["factors"]
        for k, v in zip(FACTOR_KEYS, [z_c, z_u, z_t, z_m, z_e]):
            factors_all[k].append(v.cpu().float())
    return {k: torch.cat(vs, dim=0) for k, vs in factors_all.items() if vs}


@torch.no_grad()
def extract_factors_mosei(model, cfg: dict, device: torch.device):
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    from torch.utils.data import DataLoader
    dc = cfg["data"]
    mc = cfg["model"]
    test_ds = MoseiUttDataset(
        manifest_path = dc["test_manifest"],
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                        num_workers=4, collate_fn=collate_mosei_utt)
    factors_all = {k: [] for k in FACTOR_KEYS}
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        _, _, _, aux = model(batch)
        if "factors" not in aux:
            continue
        z_c, z_u, z_t, z_m, z_e = aux["factors"]
        for k, v in zip(FACTOR_KEYS, [z_c, z_u, z_t, z_m, z_e]):
            factors_all[k].append(v.cpu().float())
    return {k: torch.cat(vs, dim=0) for k, vs in factors_all.items() if vs}


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity matrix
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine_matrix(factors: dict[str, torch.Tensor]) -> np.ndarray:
    """
    5×5 matrix of mean cosine similarities across test samples.
    For different-dim pairs, uses the min-dim prefix (same as independence_loss).
    Diagonal = 1.0 (self-similarity).
    """
    keys = FACTOR_KEYS
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                mat[i, j] = 1.0
                continue
            zi = factors[ki]   # [N, d_i]
            zj = factors[kj]   # [N, d_j]
            min_d = min(zi.size(1), zj.size(1))
            zi_n  = F.normalize(zi[:, :min_d], dim=1)
            zj_n  = F.normalize(zj[:, :min_d], dim=1)
            cos_per_sample = (zi_n * zj_n).sum(dim=1)   # [N]
            mat[i, j] = cos_per_sample.mean().item()
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Plotting
# ─────────────────────────────────────────────────────────────────────────────

DATASET_TITLES = {
    "iemocap": "IEMOCAP",
    "mosei":   "CMU-MOSEI",
    "meld":    "MELD",
}

# Professional palette
NAVY    = "#1B2A4A"
SLATE   = "#3A5683"
TEAL    = "#2C7873"
GOLD    = "#E8A838"
GRAY_BG = "#F4F5F7"


def plot_single_heatmap(mat: np.ndarray, ds_name: str, n_samples: int):
    """Publication-quality single-dataset heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    fig.patch.set_facecolor("white")

    # Custom diverging colormap: dark-blue (negative) → white (zero) → dark-red (positive)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    mask_diag = np.eye(5, dtype=bool)   # mask diagonal for cleaner annotation
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=-0.3, vmax=0.3,
        center=0,
        annot=True, fmt=".3f",
        annot_kws={"size": 10, "weight": "bold", "color": "black"},
        linewidths=0.8,
        linecolor="#CCCCCC",
        square=True,
        cbar_kws={"label": "Mean Cosine Similarity", "shrink": 0.82},
    )

    # Overlay diagonal with special color
    for i in range(5):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color=NAVY, alpha=0.85, zorder=3))
        ax.text(i + 0.5, i + 0.5, "1.000", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_xticklabels(FACTOR_NAMES, fontsize=9.5, rotation=0, ha="center")
    ax.set_yticklabels(FACTOR_NAMES, fontsize=9.5, rotation=0, va="center")
    ax.tick_params(length=0)

    ax.set_title(
        f"{DATASET_TITLES[ds_name]}\n"
        f"(n = {n_samples} test samples)",
        fontsize=12, fontweight="bold", color=NAVY, pad=10,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean Cosine Similarity", fontsize=8.5, color=NAVY)

    plt.tight_layout()
    out_png = FIG_DIR / f"cosine_heatmap_{ds_name}.png"
    out_pdf = FIG_DIR / f"cosine_heatmap_{ds_name}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  [{ds_name}] saved → {out_png.name}")


def plot_combined_heatmap(all_mats: dict[str, np.ndarray],
                          all_n: dict[str, int]):
    """3-panel combined figure for the paper."""
    n_ds = len(all_mats)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5.0))
    if n_ds == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    for ax, (ds_name, mat) in zip(axes, all_mats.items()):
        sns.heatmap(
            mat, ax=ax, cmap=cmap,
            vmin=-0.3, vmax=0.3, center=0,
            annot=True, fmt=".3f",
            annot_kws={"size": 9, "weight": "bold", "color": "black"},
            linewidths=0.7, linecolor="#CCCCCC",
            square=True,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Mean Cosine Similarity", "shrink": 0.82},
        )
        for i in range(5):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                       color=NAVY, alpha=0.85, zorder=3))
            ax.text(i + 0.5, i + 0.5, "1.000", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=4)
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(FACTOR_NAMES, fontsize=8.5, rotation=0, ha="center")
        ax.set_yticklabels(FACTOR_NAMES if ax is axes[0] else [""] * 5,
                           fontsize=8.5, rotation=0, va="center")
        ax.tick_params(length=0)
        ax.set_title(
            f"{DATASET_TITLES[ds_name]}\n(n={all_n[ds_name]})",
            fontsize=11, fontweight="bold", color=NAVY, pad=8,
        )

    # Global title
    fig.suptitle(
        "Pairwise Cosine Similarity of CHADO Latent Factors (Test Set)",
        fontsize=13, fontweight="bold", color=NAVY, y=1.01,
    )

    plt.tight_layout()
    out_png = FIG_DIR / "cosine_heatmap_combined.png"
    out_pdf = FIG_DIR / "cosine_heatmap_combined.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  Combined figure → {out_png}")
    log(f"  Combined figure → {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 72)
    log("CHADO Latent Factor Disentanglement Heatmap  [Reviewer Response #3]")
    log("=" * 72)

    # Phase 1: train if needed
    phase1_train()

    # Phase 2+3+4: extract factors + compute + plot
    log("\n=== Phase 2-4: Extract factors, compute cosine similarity, plot ===")

    # Pick first free GPU for inference
    infer_gpu = free_gpu()
    if infer_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(infer_gpu)
        log(f"  Using GPU {infer_gpu} for inference")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_mats = {}
    all_n    = {}
    all_data = {}

    extract_fns = {
        "iemocap": extract_factors_iemocap,
        "mosei":   extract_factors_mosei,
        "meld":    extract_factors_meld,
    }

    for ds_name, info in DATASETS.items():
        best_pt = info["ckpt_dir"] / "best.pt"
        if not best_pt.exists():
            log(f"  [{ds_name}] No checkpoint found — skipping (training failed?)")
            continue

        log(f"  [{ds_name}] Loading model from {best_pt} ...")
        model, cfg = load_model(ds_name, info, device)

        log(f"  [{ds_name}] Extracting factors from test set ...")
        factors = extract_fns[ds_name](model, cfg, device)

        n_samples = next(iter(factors.values())).size(0)
        log(f"  [{ds_name}] {n_samples} test samples | "
            f"factor dims: " + " ".join(f"{k}={v.shape[1]}" for k, v in factors.items()))

        mat = pairwise_cosine_matrix(factors)
        all_mats[ds_name] = mat
        all_n[ds_name]    = n_samples

        # Print matrix
        log(f"\n  [{ds_name}] Cosine similarity matrix:")
        header = "        " + "  ".join(f"{n:>18}" for n in FACTOR_NAMES)
        log(header.replace("\n", " "))
        for i, ki in enumerate(FACTOR_KEYS):
            row = f"  {ki:>5}   " + "  ".join(f"{mat[i,j]:>18.4f}" for j in range(5))
            log(row)

        # off-diagonal stats
        mask = ~np.eye(5, dtype=bool)
        off = mat[mask]
        log(f"\n  [{ds_name}] Off-diagonal: mean={off.mean():.4f}  "
            f"abs_mean={np.abs(off).mean():.4f}  max_abs={np.abs(off).max():.4f}")

        # Save individual heatmap
        plot_single_heatmap(mat, ds_name, n_samples)

        # Store for JSON
        all_data[ds_name] = {
            "n_samples": int(n_samples),
            "factor_dims": {k: int(v.shape[1]) for k, v in factors.items()},
            "cosine_matrix": mat.tolist(),
            "off_diagonal_mean":     float(off.mean()),
            "off_diagonal_abs_mean": float(np.abs(off).mean()),
            "off_diagonal_max_abs":  float(np.abs(off).max()),
        }

        del model, factors
        torch.cuda.empty_cache()

    # Combined figure
    if all_mats:
        plot_combined_heatmap(all_mats, all_n)

    # Save JSON summary
    out_json = FIG_DIR / "factor_cosine_similarity.json"
    json.dump(all_data, open(out_json, "w"), indent=2)
    log(f"\n  JSON summary → {out_json}")

    # Print paper-ready summary
    log("\n" + "=" * 72)
    log("SUMMARY — Off-diagonal cosine similarity (lower = more disentangled)")
    log("=" * 72)
    for ds_name, d in all_data.items():
        log(f"  {DATASET_TITLES[ds_name]:<12}  "
            f"mean={d['off_diagonal_mean']:+.4f}  "
            f"|mean|={d['off_diagonal_abs_mean']:.4f}  "
            f"max|cos|={d['off_diagonal_max_abs']:.4f}")

    log("\nDone. Figures in experiments/figures/disentanglement/")


if __name__ == "__main__":
    # Inference GPU is chosen dynamically in phase 2 (after training frees GPUs)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    main()
