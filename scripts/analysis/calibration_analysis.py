
from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

FIG_DIR  = ROOT / "experiments/figures/calibration"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ROOT / "experiments/results/calibration_summary.json"

NAVY  = "#1B2A4A"
TEAL  = "#2C7873"
GOLD  = "#E8A838"
RED   = "#C0392B"
GRAY  = "#7F8C8D"

N_BINS = 15

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error (confidence-binned)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(acc - conf)
    return float(ece * 100)   # as percentage


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score."""
    n_cls = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Returns bin_centers, mean_conf, mean_acc, bin_sizes for reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, sizes = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        confs.append(confidences[mask].mean())
        accs.append(correct[mask].mean())
        sizes.append(mask.sum())
    return np.array(centers), np.array(confs), np.array(accs), np.array(sizes)


# ─────────────────────────────────────────────────────────────────────────────
# Load logits from checkpoint via inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_iemocap(ckpt_path: Path, config_path: Path, gpu: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs [N,C], labels [N]) from IEMOCAP test set."""
    import yaml
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    from models.chado.model import CHADOTrimodal
    from training.engine.trainer import load_checkpoint
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(open(config_path))
    mc, dc, cc = cfg["model"], cfg["data"], cfg.get("chado", {})
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = CHADOTrimodal(
        text_model_name=mc["text_model_name"],
        audio_model_name=mc["audio_model_name"],
        video_model_name=mc["video_model_name"],
        num_classes=dc["num_classes"],
        proj_dim=mc.get("proj_dim", 256),
        dropout=mc.get("dropout", 0.2),
        use_text=mc.get("use_text", True),
        use_audio=mc.get("use_audio", True),
        use_video=mc.get("use_video", True),
        use_gated_fusion=mc.get("use_gated_fusion", True),
        backbone=cc.get("backbone", "ctnet"),
        n_heads=mc.get("n_heads", 4),
        n_speakers=cc.get("n_speakers", 10),
        use_causal=cc.get("use_causal", True),
        use_hyperbolic=cc.get("use_hyperbolic", True),
        use_ot=cc.get("use_ot", True),
        use_mad=cc.get("use_mad", True),
    ).to(device)
    load_checkpoint(model, str(ckpt_path))
    model.eval()

    test_ds = IEMOCAPDataset(
        csv_path=dc["test_csv"], text_model_name=mc["text_model_name"],
        image_model_name=mc["video_model_name"],
        max_text_len=dc.get("max_text_len", 96), audio_sec=dc.get("max_audio_seconds", 4.0),
        n_frames=dc.get("num_frames", 8), use_audio=mc.get("use_audio", True),
        use_video=mc.get("use_video", True),
    )
    loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_iemocap)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            wav   = batch.get("wav")
            pix   = batch.get("pixel_values")
            if wav is not None:  wav = wav.to(device)
            if pix is not None:  pix = pix.to(device)
            logits, _, _ = model(
                text_input={"input_ids": ids, "attention_mask": mask},
                audio_wave=wav, video_frames=pix,
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Load calibration data from saved logits (if available)
# ─────────────────────────────────────────────────────────────────────────────

def load_saved_logits(result_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load pre-saved logits/labels from result directory."""
    for fname in ("test_logits.npz", "calibration_data.npz"):
        p = result_dir / fname
        if p.exists():
            data = np.load(p)
            probs = torch.softmax(torch.tensor(data["logits"]), dim=-1).numpy()
            return probs, data["labels"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Save logits during training evaluation (hook into train scripts)
# ─────────────────────────────────────────────────────────────────────────────

def save_logits_to_result_dir(logits: np.ndarray, labels: np.ndarray, result_dir: str | Path):
    """Called from training script to persist logits for calibration analysis."""
    p = Path(result_dir) / "test_logits.npz"
    np.savez_compressed(str(p), logits=logits, labels=labels)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    entries: list[dict],   # [{label, probs, labels, color}]
    ds_name: str,
    title: str,
):
    """Multi-model reliability diagram with ECE annotations."""
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")

    # Perfect calibration diagonal
    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.2, label="Perfect")

    for entry in entries:
        _, conf_vals, acc_vals, _ = calibration_bins(entry["probs"], entry["labels"])
        ece = compute_ece(entry["probs"], entry["labels"])
        ax_main.plot(conf_vals, acc_vals, "o-",
                     color=entry["color"], linewidth=2.0, markersize=5,
                     label=f"{entry['label']}  ECE={ece:.2f}%")

    ax_main.set_xlabel("Confidence", fontsize=11, color=NAVY)
    ax_main.set_ylabel("Accuracy", fontsize=11, color=NAVY)
    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax_main.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_main.grid(True, alpha=0.25)
    ax_main.set_title(f"{ds_name} — Reliability Diagram\n{title}",
                      fontsize=12, fontweight="bold", color=NAVY)
    ax_main.legend(fontsize=9, loc="upper left", framealpha=0.85)

    # Confidence histogram (first entry only, for reference)
    if entries:
        confidences = entries[0]["probs"].max(axis=1)
        ax_hist.hist(confidences, bins=N_BINS, range=(0, 1),
                     color=entries[0]["color"], alpha=0.65, edgecolor="white")
        ax_hist.set_xlabel("Confidence", fontsize=9, color=NAVY)
        ax_hist.set_ylabel("Count", fontsize=9, color=NAVY)
        ax_hist.set_xlim(0, 1)
        ax_hist.grid(True, alpha=0.2)

    plt.tight_layout()
    stem = f"reliability_{ds_name.lower().replace(' ', '_').replace('-', '_')}"
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  Saved → {FIG_DIR / stem}.png")


def plot_ece_bar_chart(summary: dict):
    """Bar chart: ECE comparison across models and datasets."""
    ds_names  = list(summary.keys())
    models    = list({m for ds in summary.values() for m in ds})
    colors    = [NAVY, TEAL, GOLD, RED, GRAY, "#8E44AD", "#27AE60"]
    c_map     = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(ds_names), figsize=(5.5 * len(ds_names), 5.0),
                             sharey=False)
    if len(ds_names) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, ds_names):
        ms  = [m for m in models if m in summary[ds]]
        ece = [summary[ds][m]["ece"] for m in ms]
        bar_colors = [c_map[m] for m in ms]
        bars = ax.bar(ms, ece, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.65)
        for bar, val in zip(bars, ece):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_title(ds, fontsize=12, fontweight="bold", color=NAVY)
        ax.set_ylabel("ECE (%)", fontsize=10, color=NAVY)
        ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle("Expected Calibration Error (lower = better)",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ece_comparison.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  Saved → {FIG_DIR}/ece_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def scan_result_dirs() -> list[dict]:
    """Scan experiments/results for saved test_logits.npz files."""
    results_root = ROOT / "experiments/results"
    found = []
    for logits_file in sorted(results_root.rglob("test_logits.npz")):
        rel = logits_file.parent.relative_to(results_root)
        parts = rel.parts
        ds = parts[0] if parts else "unknown"
        label = "/".join(parts)
        found.append({"path": logits_file.parent, "label": label, "ds": ds})
    return found


def main():
    log("=" * 72)
    log("Calibration Analysis — ECE as Headline Metric")
    log("=" * 72)

    entries_by_ds: dict[str, list] = {}
    summary: dict = {}

    model_colors = {
        "chado": NAVY,
        "entropy": TEAL,
        "margin":  GOLD,
        "mad":     NAVY,
    }

    found = scan_result_dirs()
    if not found:
        log("  No saved logits found (test_logits.npz).")
        log("  Run calibration_analysis.py after training completes.")
        log("  Training scripts will save logits automatically.")
        return

    for item in found:
        data = load_saved_logits(item["path"])
        if data is None:
            continue
        probs, labels = data
        ece   = compute_ece(probs, labels)
        brier = compute_brier(probs, labels)
        entropy_mean = float(-(probs * np.log(probs.clip(1e-8))).sum(axis=1).mean())

        ds    = item["ds"]
        label = item["label"].split("/")[-1]

        # color assignment
        color = model_colors.get(label, GRAY)
        for key, col in model_colors.items():
            if key in item["label"].lower():
                color = col
                break

        log(f"  {item['label']:<40}  ECE={ece:.2f}%  Brier={brier:.4f}  H={entropy_mean:.3f}")

        if ds not in entries_by_ds:
            entries_by_ds[ds] = []
        entries_by_ds[ds].append({"label": label, "probs": probs, "labels": labels, "color": color})

        if ds not in summary:
            summary[ds] = {}
        summary[ds][label] = {"ece": ece, "brier": brier, "entropy": entropy_mean,
                               "n": int(len(labels))}

    for ds, entries in entries_by_ds.items():
        # Sort: CHADO first, then others
        entries.sort(key=lambda e: (0 if "chado" in e["label"].lower() else 1, e["label"]))
        plot_reliability_diagram(entries, ds.upper(), "CHADO vs Baselines")

    if summary:
        plot_ece_bar_chart(summary)
        with open(OUT_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"\n  Summary → {OUT_JSON}")

    log("\n=== Done ===")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
