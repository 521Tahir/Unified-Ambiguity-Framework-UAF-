#!/usr/bin/env python3
"""
A.3 — Factor-level ablation evaluation.

Main table:
  Variant          IEMOCAP Acc/F1   CMU-MOSEI wAcc/F1   MELD Acc/F1
  Full AR-GOT
  w/o context z_c
  w/o speaker z_u
  w/o temporal z_t
  w/o modal z_m
  w/o residual z_e

Targeted slices (model-independent stratification):
  z_m slice : low cross-modal cosine agreement (bottom tertile cos_sim(T,A))
              → removing z_m should hurt most here
  z_c slice : later-turn samples (top tertile turn_norm within conversation)
              → removing z_c should hurt most here

Usage:
  python3 scripts/eval/run_A3_factor_ablation_eval.py --dataset iemocap
  python3 scripts/eval/run_A3_factor_ablation_eval.py --dataset all
"""
import argparse, json, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RES     = os.path.join(ROOT, "experiments", "results")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "A3_factor_ablation")
FIG_DIR = os.path.join(ROOT, "experiments", "figures", "A3_factor_ablation")
for d in (OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

SEEDS = [42, 3407, 456]

VARIANTS = [
    ("Full AR-GOT",   "argot"),      # from main 3-seed run
    ("w/o z_c ctx",   "wo_ctx"),
    ("w/o z_u spk",   "wo_spk"),
    ("w/o z_t turn",  "wo_turn"),
    ("w/o z_m modal", "wo_modal"),
    ("w/o z_e resid", "wo_residual"),
]

COLORS = {
    "Full AR-GOT":   "#2E4057",
    "w/o z_c ctx":   "#048A81",
    "w/o z_u spk":   "#C25B5B",
    "w/o z_t turn":  "#E07B3F",
    "w/o z_m modal": "#7B5EA7",
    "w/o z_e resid": "#5B8DB8",
}

matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False


# ── result loaders ────────────────────────────────────────────────────────────

def _res_path(dataset, variant_dir, seed):
    """
    Full AR-GOT lives in {dataset}/argot_ms/seed_{s}
    Ablations live in  {dataset}/ablations/{variant}/seed_{s}
    """
    if variant_dir == "argot":
        return os.path.join(RES, dataset, "argot_ms", f"seed_{seed}", "test_results.json")
    return os.path.join(RES, dataset, "ablations", variant_dir, f"seed_{seed}", "test_results.json")


def load_mean_std(dataset, variant_dir):
    is_mosei = (dataset == "mosei")
    acc_vals, f1_vals = [], []
    for s in SEEDS:
        p = _res_path(dataset, variant_dir, s)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        acc = d.get("wacc" if is_mosei else "accuracy",
                    d.get("test_accuracy", None))
        f1  = d.get("macro_f1", d.get("test_macro_f1", None))
        if acc is not None: acc_vals.append(float(acc) * 100)
        if f1  is not None: f1_vals.append(float(f1)  * 100)
    n = len(acc_vals)
    if n == 0:
        return None, None, None, None, 0
    acc_m = np.mean(acc_vals);  acc_s = np.std(acc_vals, ddof=1) if n > 1 else 0.0
    f1_m  = np.mean(f1_vals);   f1_s  = np.std(f1_vals,  ddof=1) if n > 1 else 0.0
    return acc_m, acc_s, f1_m, f1_s, n


# ── print main ablation table ─────────────────────────────────────────────────

def print_main_table(datasets):
    border = "=" * 95
    print(f"\n{border}")
    print("  A.3 Factor-Level Ablation  —  Mean ± Std (%)  across seeds 42/3407/456")
    print(border)

    # header
    h = f"  {'Variant':<22}"
    for ds in datasets:
        label = "wAcc" if ds == "mosei" else "Acc"
        h += f"  {ds.upper()+' '+label:>12}  {'Macro-F1':>12}"
    print(h)
    print("-" * 95)

    all_results = {}
    for display, vdir in VARIANTS:
        row = f"  {display:<22}"
        all_results[display] = {}
        for ds in datasets:
            acc_m, acc_s, f1_m, f1_s, n = load_mean_std(ds, vdir)
            all_results[display][ds] = (acc_m, acc_s, f1_m, f1_s, n)
            if acc_m is None:
                row += f"  {'—':>12}  {'—':>12}"
            else:
                row += f"  {acc_m:5.2f}±{acc_s:4.2f}  {f1_m:5.2f}±{f1_s:4.2f}"
        print(row)

    print(border)
    print("  Drop from Full AR-GOT (negative = worse):")
    full = all_results["Full AR-GOT"]
    for display, vdir in VARIANTS[1:]:
        row = f"  {display:<22}"
        for ds in datasets:
            fa, _, ff, _, _ = full.get(ds, (None,)*5)
            va, _, vf, _, _ = all_results[display].get(ds, (None,)*5)
            if fa is None or va is None:
                row += f"  {'—':>12}  {'—':>12}"
            else:
                row += f"  {va-fa:>+12.2f}  {vf-ff:>+12.2f}"
        print(row)
    print(border)
    return all_results


# ── targeted slice inference ──────────────────────────────────────────────────

def _get_test_modal_sim_iemocap(cfg_path, ckpt_path, device):
    """Compute cos_sim(text_enc, audio_enc) on IEMOCAP test set."""
    from models.chado.model import CHADOTrimodal
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]; cc = cfg.get("chado", {}); dc = cfg["data"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)

    model = CHADOTrimodal(
        text_model_name=mc["text_model_name"],
        audio_model_name=mc["audio_model_name"],
        video_model_name=mc.get("video_model_name", "google/vit-base-patch16-224-in21k"),
        num_classes=dc["num_classes"],
        proj_dim=mc.get("proj_dim", 256),
        dropout=mc.get("dropout", 0.2),
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=mc.get("use_gated_fusion", True),
        backbone=cc.get("backbone", "ctnet"),
        n_heads=mc.get("n_heads", 4),
        n_speakers=cc.get("n_speakers", 10),
        use_causal=True, use_hyperbolic=False, use_ot=False, use_mad=False,
        w_causal=0.0, w_hyperbolic=0.0, w_ot=0.0, w_mad=0.0,
        w_speaker=0.0, w_turn=0.0, w_modal=0.0, w_context=0.0,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = IEMOCAPDataset(
        csv_path=dc["test_csv"],
        text_model_name=mc["text_model_name"],
        image_model_name=mc.get("video_model_name", "google/vit-base-patch16-224-in21k"),
        max_text_len=dc.get("max_text_len", 96),
        audio_sr=dc.get("sample_rate", 16000),
        audio_sec=dc.get("max_audio_seconds", 4.0),
        n_frames=dc.get("num_frames", 8),
        use_audio=True, use_video=True,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate_iemocap)

    modal_sims, labels, utt_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            ti  = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
            wav = batch.get("wav", batch.get("audio_wave"))
            if wav is not None: wav = wav.to(device)
            pv  = batch.get("pixel_values", batch.get("video_frames"))
            if pv is not None: pv = pv.to(device)
            _, raw_z, _ = model.base(text_input=ti, audio_wave=wav, video_frames=pv)
            # raw_z: [B, n_mods * proj_dim] — split into modality chunks
            proj = mc.get("proj_dim", 256)
            text_e = raw_z[:, 0:proj]
            audio_e = raw_z[:, proj:2*proj]
            sim = F.cosine_similarity(text_e, audio_e, dim=1).cpu().numpy()
            modal_sims.append(sim)
            labels.extend(batch["label"].tolist())
            utt_ids.extend(batch.get("utt_id", [""]*len(batch["label"])))
    return np.concatenate(modal_sims), np.array(labels), utt_ids


def _get_turn_norms_iemocap(cfg_path):
    """Read turn_norm from IEMOCAP test CSV."""
    import pandas as pd
    cfg = yaml.safe_load(open(cfg_path))
    df = pd.read_csv(cfg["data"]["test_csv"])
    if "turn_norm" in df.columns:
        return df["turn_norm"].fillna(-1).values
    # fallback: estimate from utt_id ordering within session
    return np.full(len(df), -1.0)


def run_targeted_slices_iemocap(all_results, datasets, device="cpu"):
    """
    Compare Full AR-GOT vs w/o z_m on low-modal-agreement samples.
    Compare Full AR-GOT vs w/o z_c on later-turn samples.
    Uses seed_42 checkpoint of Full AR-GOT for feature extraction.
    """
    if "iemocap" not in datasets:
        return

    print("\n" + "="*80)
    print("  IEMOCAP — Targeted Slices")
    print("="*80)

    cfg_path   = os.path.join(ROOT, "configs/iemocap/chado_iemocap_final.yaml")
    full_ckpt  = os.path.join(RES, "iemocap/argot_ms/seed_42/best.pt")

    if not os.path.exists(full_ckpt):
        print("  [SKIP] Full AR-GOT seed_42 checkpoint not found")
        return

    # ── get modal_sim scores from full model ──────────────────────────────
    print("  Computing cross-modal agreement scores …")
    try:
        modal_sims, labels, utt_ids = _get_test_modal_sim_iemocap(
            cfg_path, full_ckpt, device)
    except Exception as e:
        print(f"  [WARN] modal_sim extraction failed: {e}")
        modal_sims = None

    if modal_sims is not None:
        thr_lo = np.percentile(modal_sims, 33)
        thr_hi = np.percentile(modal_sims, 67)
        low_mask  = modal_sims <= thr_lo
        mid_mask  = (modal_sims > thr_lo) & (modal_sims < thr_hi)
        high_mask = modal_sims >= thr_hi

        print(f"\n  Cross-modal agreement distribution:")
        print(f"    Low  (cos≤{thr_lo:.3f}): {low_mask.sum()} samples")
        print(f"    Mid  :                 {mid_mask.sum()} samples")
        print(f"    High (cos≥{thr_hi:.3f}): {high_mask.sum()} samples")

        # For each variant, get predictions from seed_42 checkpoint
        def _get_preds(variant_dir):
            from models.chado.model import CHADOTrimodal
            from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
            from torch.utils.data import DataLoader

            ckpt_p = _res_path("iemocap", variant_dir, 42).replace(
                "test_results.json", "best.pt")
            if not os.path.exists(ckpt_p):
                return None, None
            cfg = yaml.safe_load(open(cfg_path))
            mc = cfg["model"]; cc = cfg.get("chado", {}); dc = cfg["data"]
            ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt)

            # Load ablation config for this variant to get correct weights
            abl_cfg_path = os.path.join(ROOT, f"configs/iemocap/ablation_{variant_dir}.yaml") \
                if variant_dir != "argot" else cfg_path
            abl_cfg = yaml.safe_load(open(abl_cfg_path)) if os.path.exists(abl_cfg_path) else cfg
            acc = abl_cfg.get("chado", {})

            model = CHADOTrimodal(
                text_model_name=mc["text_model_name"],
                audio_model_name=mc["audio_model_name"],
                video_model_name=mc.get("video_model_name", "google/vit-base-patch16-224-in21k"),
                num_classes=dc["num_classes"],
                proj_dim=mc.get("proj_dim", 256), dropout=mc.get("dropout", 0.2),
                use_text=True, use_audio=True, use_video=True,
                use_gated_fusion=mc.get("use_gated_fusion", True),
                backbone=acc.get("backbone", "ctnet"), n_heads=mc.get("n_heads", 4),
                n_speakers=acc.get("n_speakers", 10),
                use_causal=True, use_hyperbolic=False, use_ot=False, use_mad=False,
                w_causal=0.0, w_hyperbolic=0.0, w_ot=0.0, w_mad=0.0,
                w_speaker=0.0, w_turn=0.0, w_modal=0.0, w_context=0.0,
            ).to(device)
            model.load_state_dict(sd, strict=False)
            model.eval()

            ds = IEMOCAPDataset(
                csv_path=dc["test_csv"],
                text_model_name=mc["text_model_name"],
                image_model_name=mc.get("video_model_name", "google/vit-base-patch16-224-in21k"),
                max_text_len=dc.get("max_text_len", 96),
                audio_sr=dc.get("sample_rate", 16000),
                audio_sec=dc.get("max_audio_seconds", 4.0),
                n_frames=dc.get("num_frames", 8),
                use_audio=True, use_video=True,
            )
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                                collate_fn=collate_iemocap)
            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    ti  = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
                    wav = batch.get("wav", batch.get("audio_wave"))
                    if wav is not None: wav = wav.to(device)
                    pv  = batch.get("pixel_values", batch.get("video_frames"))
                    if pv is not None: pv = pv.to(device)
                    out = model(text_input=ti, audio_wave=wav, video_frames=pv)
                    logits = out[0] if isinstance(out, tuple) else out
                    all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    all_labels.extend(batch["label"].tolist())
            return np.vstack(all_probs), np.array(all_labels)

        print("\n  z_m targeted slice (low cross-modal agreement):")
        print(f"  {'Variant':<22}  {'Full-set Acc':>12}  {'Low-modal Acc':>13}  {'Drop':>6}")
        for display, vdir in [("Full AR-GOT", "argot"), ("w/o z_m modal", "wo_modal")]:
            probs, lbs = _get_preds(vdir)
            if probs is None:
                print(f"  {display:<22}  (checkpoint missing)")
                continue
            preds = probs.argmax(axis=1)
            full_acc = accuracy_score(lbs, preds) * 100
            if low_mask.sum() > 5:
                low_acc = accuracy_score(lbs[low_mask], preds[low_mask]) * 100
                drop = low_acc - full_acc
                print(f"  {display:<22}  {full_acc:>12.2f}  {low_acc:>13.2f}  {drop:>+6.2f}")
            else:
                print(f"  {display:<22}  {full_acc:>12.2f}  (too few samples)")

    # ── z_c turn-position slice ───────────────────────────────────────────
    turn_norms = _get_turn_norms_iemocap(cfg_path)
    valid_mask = turn_norms >= 0
    if valid_mask.sum() > 20:
        thr_late = np.percentile(turn_norms[valid_mask], 67)
        later_mask = (turn_norms >= thr_late)
        print(f"\n  z_c targeted slice (later-turn samples, turn_norm≥{thr_late:.2f}):")
        print(f"  {'Variant':<22}  {'Full-set Acc':>12}  {'Later-turn Acc':>14}  {'Drop':>6}")
        for display, vdir in [("Full AR-GOT", "argot"), ("w/o z_c ctx", "wo_ctx")]:
            probs, lbs = _get_preds(vdir) if modal_sims is not None else (None, None)
            if probs is None:
                print(f"  {display:<22}  (checkpoint missing)")
                continue
            preds = probs.argmax(axis=1)
            later_lbs = lbs[later_mask[:len(lbs)]]
            later_preds = preds[later_mask[:len(preds)]]
            if len(later_lbs) > 5:
                full_acc  = accuracy_score(lbs, preds) * 100
                later_acc = accuracy_score(later_lbs, later_preds) * 100
                drop = later_acc - full_acc
                print(f"  {display:<22}  {full_acc:>12.2f}  {later_acc:>14.2f}  {drop:>+6.2f}")
    else:
        print("\n  [INFO] turn_norm unavailable in test CSV — skipping z_c targeted slice")


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_ablation(all_results, datasets):
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]
    fig.suptitle("A.3 — Factor-Level Ablation (Accuracy drop from Full AR-GOT)",
                 fontsize=13, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        acc_key = "wAcc" if ds == "mosei" else "Acc"
        full_acc, _, _, _, _ = all_results.get("Full AR-GOT", {}).get(ds, (None,)*5) or (None,)*5
        if full_acc is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        names, drops, errs = [], [], []
        for display, vdir in VARIANTS[1:]:
            va, vs, _, _, _ = all_results.get(display, {}).get(ds, (None,)*5) or (None,)*5
            if va is None:
                continue
            names.append(display.replace("w/o ", ""))
            drops.append(va - full_acc)
            errs.append(vs if vs else 0.0)

        x = np.arange(len(names))
        colors = [COLORS.get(f"w/o {n}", "#888888") for n in names]
        bars = ax.bar(x, drops, color=colors, edgecolor="white", linewidth=0.8,
                      yerr=errs, capsize=4, zorder=3)
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax.set_ylabel(f"Δ {acc_key} (pp) vs Full AR-GOT")
        ax.set_title(ds.upper())
        ax.set_facecolor("#F7F9FB")
        ax.grid(axis="y", color="#DDEAF7", linewidth=0.8, zorder=0)
        for bar, drop in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2,
                    drop + (0.1 if drop >= 0 else -0.3),
                    f"{drop:+.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"A3_factor_ablation.{ext}")
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {FIG_DIR}/A3_factor_ablation.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["iemocap", "mosei", "meld", "all"], default="all")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    datasets = ["iemocap", "mosei", "meld"] if args.dataset == "all" else [args.dataset]
    device   = torch.device(args.device)

    all_results = print_main_table(datasets)

    # Save JSON
    out = {}
    for display, _ in VARIANTS:
        out[display] = {}
        for ds in datasets:
            vals = all_results.get(display, {}).get(ds, (None,)*5) or (None,)*5
            acc_m, acc_s, f1_m, f1_s, n = vals
            out[display][ds] = {
                "acc_mean": round(acc_m, 4) if acc_m else None,
                "acc_std":  round(acc_s, 4) if acc_s else None,
                "f1_mean":  round(f1_m,  4) if f1_m  else None,
                "f1_std":   round(f1_s,  4) if f1_s  else None,
                "n_seeds":  n,
            }
    json_path = os.path.join(OUT_DIR, "A3_factor_ablation.json")
    json.dump(out, open(json_path, "w"), indent=2)
    print(f"\n  Results saved → {json_path}")

    plot_ablation(all_results, datasets)

    # Targeted slices (IEMOCAP only — requires model inference)
    if "iemocap" in datasets:
        try:
            run_targeted_slices_iemocap(all_results, datasets, device)
        except Exception as e:
            print(f"\n  [WARN] Targeted slice evaluation failed: {e}")
            import traceback; traceback.print_exc()

    print("\nA.3 factor ablation evaluation done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
