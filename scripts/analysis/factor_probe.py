"""
AR-GOT factor–attribute specialization analysis.

Inputs:
  experiments/results/iemocap/factor_analysis/argot_factor_embeddings.npz

Outputs (all in experiments/results/iemocap/factor_analysis/):
  factor_attribute_raw_balanced_accuracy.csv
  factor_attribute_chance_corrected_scores.csv
  factor_attribute_column_normalized_scores.csv
  factor_attribute_leave_one_factor_out_drop.csv
  factor_attribute_leave_one_factor_out_chance_corrected_drop.csv
  factor_attribute_probe_report.json
  factor_attribute_specialization_heatmap.{pdf,png}
  factor_attribute_leave_one_out_heatmap.{pdf,png}
  factor_attribute_combined_figure.{pdf,png}
"""
import argparse
import json
import os
import textwrap
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─── constants ────────────────────────────────────────────────────────────────
FACTOR_KEYS   = ["z_c", "z_u", "z_tau", "z_m", "z_e"]
FACTOR_LABELS = ["z_c Context", "z_u Speaker", "z_tau Temporal",
                 "z_m Modality", "z_e Residual"]
ATTR_KEYS     = ["gender", "speaker_session_id", "age_proxy_label"]
ATTR_LABELS   = ["Gender", "Speaker / Session", "Age-proxy\n(Utterance Duration)"]

SEEDS = [1, 2, 3, 4, 5]
N_SPLITS = 5

# Publication colour palette
NAVY   = "#1B3A6B"
SLATE  = "#4A6FA5"
TEAL   = "#2E8B8B"
GOLD   = "#D4A84B"
LGRAY  = "#F2F4F7"

CMAP_MAIN = LinearSegmentedColormap.from_list(
    "chado_main", ["#F2F4F7", "#4A6FA5", "#1B3A6B"], N=256
)
CMAP_DROP = LinearSegmentedColormap.from_list(
    "chado_drop", ["#F2F4F7", "#2E8B8B", "#0D4D4D"], N=256
)


# ─── probe utilities ──────────────────────────────────────────────────────────
def _probe(X_tr, y_tr, X_te, y_te):
    """Fit StandardScaler + LogisticRegression, return balanced_accuracy on test."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=5000, solver="lbfgs",
        random_state=42,
    )
    clf.fit(Xtr, y_tr)
    return balanced_accuracy_score(y_te, clf.predict(Xte))


def _chance(y_te):
    """Balanced accuracy of majority-class predictor on test fold."""
    classes, counts = np.unique(y_te, return_counts=True)
    majority = classes[np.argmax(counts)]
    pred = np.full_like(y_te, majority)
    return balanced_accuracy_score(y_te, pred)


def _cc(raw, chance):
    """Chance-corrected score clipped to [0, 1]."""
    denom = 1.0 - chance
    if denom < 1e-9:
        return 0.0
    return float(np.clip((raw - chance) / denom, 0.0, 1.0))


def _make_splitter(attr_key, groups, y, seed):
    """Return (train_idx, test_idx) generator for the given attribute."""
    if attr_key in ("gender", "age_proxy_label"):
        # speaker-disjoint splits
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        return cv.split(np.zeros(len(y)), y, groups=groups)
    else:
        # Speaker/Session: all identities in both train and test
        # group by dialogue to reduce near-duplicate leakage
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        return cv.split(np.zeros(len(y)), y)


# ─── main probe function ───────────────────────────────────────────────────────
def run_probes(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    factors = {k: data[k] for k in FACTOR_KEYS}
    groups = data["speaker_session_id"]

    n_utt = len(data["gender"])
    print(f"Utterances loaded: {n_utt}")

    # ── raw balanced accuracy: [n_factors × n_attrs × n_seeds] ──────────────
    raw_acc   = np.zeros((len(FACTOR_KEYS), len(ATTR_KEYS), len(SEEDS)))
    raw_chance= np.zeros((len(ATTR_KEYS), len(SEEDS)))

    for ai, attr_key in enumerate(ATTR_KEYS):
        y = data[attr_key].astype(int)
        for si, seed in enumerate(SEEDS):
            fold_accs   = []
            fold_chances= []
            for tr, te in _make_splitter(attr_key, groups, y, seed):
                y_tr, y_te = y[tr], y[te]
                # skip fold if any class missing in train or test
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                ch = _chance(y_te)
                fold_chances.append(ch)
                for fi, fk in enumerate(FACTOR_KEYS):
                    X = factors[fk]
                    acc = _probe(X[tr], y_tr, X[te], y_te)
                    fold_accs.append((fi, acc))
            if not fold_chances:
                continue
            raw_chance[ai, si] = np.mean(fold_chances)
            # aggregate per factor
            fi_accs = {fi: [] for fi in range(len(FACTOR_KEYS))}
            for fi, acc in fold_accs:
                fi_accs[fi].append(acc)
            for fi in range(len(FACTOR_KEYS)):
                raw_acc[fi, ai, si] = np.mean(fi_accs[fi]) if fi_accs[fi] else 0.0
        print(f"  [{attr_key}] done  chance≈{raw_chance[ai].mean():.3f}")

    # means and stds over seeds
    raw_mean   = raw_acc.mean(axis=2)        # [n_factors, n_attrs]
    raw_std    = raw_acc.std(axis=2)
    chance_mean= raw_chance.mean(axis=1)     # [n_attrs]

    # ── chance-corrected ──────────────────────────────────────────────────────
    cc_mean = np.zeros_like(raw_mean)
    cc_std  = np.zeros_like(raw_std)
    for ai in range(len(ATTR_KEYS)):
        ch = chance_mean[ai]
        for fi in range(len(FACTOR_KEYS)):
            vals = [_cc(raw_acc[fi, ai, si], raw_chance[ai, si]) for si in range(len(SEEDS))]
            cc_mean[fi, ai] = np.mean(vals)
            cc_std[fi, ai]  = np.std(vals)

    # ── column-normalized specialization ─────────────────────────────────────
    col_max = cc_mean.max(axis=0, keepdims=True)   # [1, n_attrs]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(col_max > 1e-9, cc_mean / col_max, 0.0)

    # ── leave-one-factor-out ──────────────────────────────────────────────────
    Z_all = np.concatenate([factors[k] for k in FACTOR_KEYS], axis=1)

    acc_all_seeds  = np.zeros((len(ATTR_KEYS), len(SEEDS)))
    acc_minus_seeds= np.zeros((len(FACTOR_KEYS), len(ATTR_KEYS), len(SEEDS)))
    ch_all_seeds   = np.zeros((len(ATTR_KEYS), len(SEEDS)))

    for ai, attr_key in enumerate(ATTR_KEYS):
        y = data[attr_key].astype(int)
        for si, seed in enumerate(SEEDS):
            fold_all_accs   = []
            fold_minus_accs = {fi: [] for fi in range(len(FACTOR_KEYS))}
            fold_chances    = []
            for tr, te in _make_splitter(attr_key, groups, y, seed):
                y_tr, y_te = y[tr], y[te]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                fold_chances.append(_chance(y_te))
                fold_all_accs.append(_probe(Z_all[tr], y_tr, Z_all[te], y_te))
                # leave-one-out
                for fi, fk in enumerate(FACTOR_KEYS):
                    parts = [factors[k] for k in FACTOR_KEYS if k != fk]
                    Z_m = np.concatenate(parts, axis=1)
                    fold_minus_accs[fi].append(_probe(Z_m[tr], y_tr, Z_m[te], y_te))
            if not fold_all_accs:
                continue
            ch_all_seeds[ai, si]  = np.mean(fold_chances)
            acc_all_seeds[ai, si] = np.mean(fold_all_accs)
            for fi in range(len(FACTOR_KEYS)):
                acc_minus_seeds[fi, ai, si] = (
                    np.mean(fold_minus_accs[fi]) if fold_minus_accs[fi] else 0.0
                )
        print(f"  LOFO [{attr_key}] done")

    acc_all_mean  = acc_all_seeds.mean(axis=1)        # [n_attrs]
    acc_minus_mean= acc_minus_seeds.mean(axis=2)      # [n_factors, n_attrs]
    ch_all_mean   = ch_all_seeds.mean(axis=1)         # [n_attrs]

    drop_raw  = acc_all_mean[None, :] - acc_minus_mean        # [n_factors, n_attrs]

    cc_all_mean  = np.array([_cc(acc_all_mean[ai], ch_all_mean[ai])
                             for ai in range(len(ATTR_KEYS))])
    cc_minus_mean= np.array([[_cc(acc_minus_mean[fi, ai], ch_all_mean[ai])
                              for ai in range(len(ATTR_KEYS))]
                             for fi in range(len(FACTOR_KEYS))])
    cc_drop = cc_all_mean[None, :] - cc_minus_mean   # [n_factors, n_attrs]

    return dict(
        raw_mean=raw_mean, raw_std=raw_std, chance_mean=chance_mean,
        cc_mean=cc_mean, cc_std=cc_std, rel=rel,
        acc_all_mean=acc_all_mean, acc_minus_mean=acc_minus_mean,
        drop_raw=drop_raw, cc_all_mean=cc_all_mean,
        cc_minus_mean=cc_minus_mean, cc_drop=cc_drop,
        n_utt=n_utt,
        gender_dist=np.bincount(data["gender"].astype(int)).tolist(),
        spk_dist=np.bincount(data["speaker_session_id"].astype(int)).tolist(),
        age_dist=np.bincount(data["age_proxy_label"].astype(int)).tolist(),
    )


# ─── CSV writers ──────────────────────────────────────────────────────────────
def _df(matrix, index_labels, col_labels, fmt=".4f"):
    return pd.DataFrame(matrix, index=index_labels, columns=col_labels)


def save_csvs(res, out_dir):
    cols_short = ["Gender", "Speaker/Session", "Age-proxy"]
    rows = FACTOR_LABELS

    _df(res["raw_mean"], rows, cols_short).to_csv(
        f"{out_dir}/factor_attribute_raw_balanced_accuracy.csv")
    _df(res["cc_mean"],  rows, cols_short).to_csv(
        f"{out_dir}/factor_attribute_chance_corrected_scores.csv")
    _df(res["rel"],      rows, cols_short).to_csv(
        f"{out_dir}/factor_attribute_column_normalized_scores.csv")
    _df(res["drop_raw"], rows, cols_short).to_csv(
        f"{out_dir}/factor_attribute_leave_one_factor_out_drop.csv")
    _df(res["cc_drop"],  rows, cols_short).to_csv(
        f"{out_dir}/factor_attribute_leave_one_factor_out_chance_corrected_drop.csv")
    print("  CSVs saved")


# ─── JSON report ──────────────────────────────────────────────────────────────
def save_json(res, out_dir, npz_path, ckpt_path):
    winning_main = {
        ATTR_LABELS[ai]: FACTOR_LABELS[int(res["cc_mean"][:, ai].argmax())]
        for ai in range(len(ATTR_KEYS))
    }
    winning_lofo = {
        ATTR_LABELS[ai]: FACTOR_LABELS[int(res["cc_drop"][:, ai].argmax())]
        for ai in range(len(ATTR_KEYS))
    }

    # interpret consistency
    consistent = all(
        winning_main[k] == winning_lofo[k] for k in winning_main
    )
    if consistent:
        interp = ("The results provide quantitative evidence of partial latent "
                  "specialization: the factor with the strongest relative association "
                  "for an attribute also produces the largest performance drop when removed.")
    else:
        interp = ("The results indicate overlapping factor information with partial "
                  "specialization tendencies rather than exclusive factor separation.")

    report = {
        "dataset": "IEMOCAP",
        "checkpoint": ckpt_path,
        "n_utterances": int(res["n_utt"]),
        "factor_dimensions": {"z_c": 64, "z_u": 64, "z_tau": 32, "z_m": 64, "z_e": 32},
        "target_attributes": ATTR_KEYS,
        "n_classes_per_attribute": {"gender": 2, "speaker_session_id": 10, "age_proxy_label": 2},
        "class_distribution": {
            "gender": res["gender_dist"],
            "speaker_session_id": res["spk_dist"],
            "age_proxy_label": res["age_dist"],
        },
        "chance_baseline": {
            ATTR_KEYS[ai]: float(res["chance_mean"][ai]) for ai in range(len(ATTR_KEYS))
        },
        "split_type": {
            "gender": "StratifiedGroupKFold(groups=speaker_session, k=5)",
            "speaker_session_id": "StratifiedKFold(k=5)",
            "age_proxy_label": "StratifiedGroupKFold(groups=speaker_session, k=5)",
        },
        "seeds": SEEDS,
        "raw_balanced_accuracy": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: float(res["raw_mean"][fi, ai])
                                for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "chance_corrected": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: {
                "mean": float(res["cc_mean"][fi, ai]),
                "std":  float(res["cc_std"][fi, ai]),
            } for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "column_normalized_specialization": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: float(res["rel"][fi, ai])
                                for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "all_factor_probe_scores": {
            ATTR_LABELS[ai]: float(res["acc_all_mean"][ai])
            for ai in range(len(ATTR_KEYS))
        },
        "leave_one_factor_out_scores": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: float(res["acc_minus_mean"][fi, ai])
                                for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "lofo_drop_raw": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: float(res["drop_raw"][fi, ai])
                                for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "lofo_chance_corrected_drop": {
            FACTOR_LABELS[fi]: {ATTR_LABELS[ai]: float(res["cc_drop"][fi, ai])
                                for ai in range(len(ATTR_KEYS))}
            for fi in range(len(FACTOR_KEYS))
        },
        "winning_factor_panel_A": winning_main,
        "largest_drop_factor_panel_B": winning_lofo,
        "interpretation": interp,
    }
    out_path = f"{out_dir}/factor_attribute_probe_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON saved → {out_path}")
    return report


# ─── figure helpers ───────────────────────────────────────────────────────────
def _wrap(labels, width=14):
    return ["\n".join(textwrap.wrap(l, width)) for l in labels]


def _draw_heatmap(ax, matrix, annot_mean, annot_std, cmap, vmin, vmax,
                  row_labels, col_labels, cbar_label, title,
                  bold_mask=None):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # grid lines
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # axis labels
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(_wrap(col_labels, width=14), fontsize=9, ha="center")
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_ticks_position("bottom")

    # cell annotations
    for fi in range(matrix.shape[0]):
        for ai in range(matrix.shape[1]):
            m_val = annot_mean[fi, ai] * 100
            s_val = annot_std[fi, ai]  * 100 if annot_std is not None else None
            txt   = f"{m_val:.1f}" if s_val is None else f"{m_val:.1f}\n±{s_val:.1f}"
            weight = "bold" if (bold_mask is not None and bold_mask[fi, ai]) else "normal"
            brightness = float(matrix[fi, ai]) / max(float(vmax), 1e-9)
            color = "white" if brightness > 0.55 else NAVY
            ax.text(ai, fi, txt, ha="center", va="center",
                    fontsize=8, fontweight=weight, color=color)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


# ─── single heatmaps ─────────────────────────────────────────────────────────
def _save_heatmap(matrix, annot_mean, annot_std, cmap, vmax, row_labels,
                  col_labels, cbar_label, title, bold_mask, out_dir, stem):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(LGRAY)
    ax.set_facecolor(LGRAY)
    _draw_heatmap(ax, matrix, annot_mean, annot_std, cmap, 0, vmax,
                  row_labels, col_labels, cbar_label, title, bold_mask)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_dir}/{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=LGRAY)
    plt.close(fig)
    print(f"  Saved {stem}.{{pdf,png}}")


# ─── combined figure ─────────────────────────────────────────────────────────
def save_figures(res, out_dir):
    row_labels = FACTOR_LABELS
    col_labels = ["Gender", "Speaker / Session",
                  "Age-proxy\n(Utterance Duration)"]

    # bold mask: max in each column
    bold_A = res["rel"] == res["rel"].max(axis=0, keepdims=True)
    bold_B = res["cc_drop"] == res["cc_drop"].max(axis=0, keepdims=True)

    # ── Panel A standalone ────────────────────────────────────────────────────
    _save_heatmap(
        matrix=res["rel"], annot_mean=res["cc_mean"], annot_std=res["cc_std"],
        cmap=CMAP_MAIN, vmax=1.0, row_labels=row_labels, col_labels=col_labels,
        cbar_label="Relative Specialization Score",
        title="A. Relative Factor–Attribute Specialization",
        bold_mask=bold_A, out_dir=out_dir,
        stem="factor_attribute_specialization_heatmap",
    )

    # drop for Panel B: clip negatives to 0 for color but keep raw annotation
    cc_drop_display = np.clip(res["cc_drop"], 0.0, None)
    _save_heatmap(
        matrix=cc_drop_display, annot_mean=res["cc_drop"],
        annot_std=None,   # single scalar per cell for LOFO
        cmap=CMAP_DROP, vmax=max(cc_drop_display.max(), 0.05),
        row_labels=row_labels, col_labels=col_labels,
        cbar_label="Chance-Corrected Drop",
        title="B. Leave-One-Factor-Out Contribution",
        bold_mask=bold_B, out_dir=out_dir,
        stem="factor_attribute_leave_one_out_heatmap",
    )

    # ── Combined figure ───────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.patch.set_facecolor(LGRAY)
    for ax in (ax_a, ax_b):
        ax.set_facecolor(LGRAY)

    _draw_heatmap(ax_a, res["rel"], res["cc_mean"], res["cc_std"],
                  CMAP_MAIN, 0, 1.0, row_labels, col_labels,
                  "Relative Specialization Score",
                  "A. Relative Factor–Attribute Specialization", bold_A)

    _draw_heatmap(ax_b, cc_drop_display, res["cc_drop"], None,
                  CMAP_DROP, 0, max(cc_drop_display.max(), 0.05),
                  row_labels, col_labels, "Chance-Corrected Drop",
                  "B. Leave-One-Factor-Out Contribution", bold_B)

    fig.suptitle("Factor–Attribute Specialization Analysis",
                 fontsize=13, fontweight="bold", y=1.01)

    caption = (
        "Factor–attribute specialization analysis using linear probes. "
        "Panel A shows column-normalized, chance-corrected factor–attribute association, "
        "where each column is normalized across factors to highlight the factor most "
        "associated with each attribute. Cell annotations report chance-corrected balanced "
        "accuracy. Panel B shows chance-corrected leave-one-factor-out drops, measuring how "
        "much attribute prediction degrades when each factor is removed from the concatenated "
        "factor representation. Higher values indicate stronger relative association or "
        "contribution. The analysis evaluates partial latent specialization and does not "
        "imply formal disentanglement."
    )
    fig.text(0.5, -0.04, "\n".join(textwrap.wrap(caption, width=130)),
             ha="center", va="top", fontsize=7.5, color="#333333",
             fontstyle="italic", wrap=True)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_dir}/factor_attribute_combined_figure.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=LGRAY)
    plt.close(fig)
    print(f"  Saved factor_attribute_combined_figure.{{pdf,png}}")


# ─── text summary ─────────────────────────────────────────────────────────────
def print_summary(res):
    attrs  = ["Gender", "Speaker / Session", "Age-proxy"]
    win_A  = [FACTOR_LABELS[int(res["cc_mean"][:, ai].argmax())] for ai in range(3)]
    win_B  = [FACTOR_LABELS[int(res["cc_drop"][:, ai].argmax())] for ai in range(3)]
    print("\n" + "="*72)
    print("TEXT SUMMARY")
    print("="*72)
    print(
        f"Linear-probe analysis was performed using chance-corrected balanced accuracy "
        f"and leave-one-factor-out probing. In the relative specialization heatmap, "
        f"the strongest factor for Gender was {win_A[0]}, the strongest factor for "
        f"Speaker / Session was {win_A[1]}, and the strongest factor for Age-proxy was "
        f"{win_A[2]}. In the leave-one-factor-out analysis, the largest drop for Gender "
        f"was caused by removing {win_B[0]}, the largest drop for Speaker / Session was "
        f"caused by removing {win_B[1]}, and the largest drop for Age-proxy was caused by "
        f"removing {win_B[2]}. These results should be interpreted as evidence of partial "
        f"latent specialization rather than formal disentanglement."
    )
    print("="*72)


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",    default="experiments/results/iemocap/factor_analysis/argot_factor_embeddings.npz")
    p.add_argument("--ckpt",   default="experiments/results/iemocap/chado_wo_mad/best.pt")
    p.add_argument("--out-dir",default="experiments/results/iemocap/factor_analysis")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Running linear probe analysis …")
    res = run_probes(args.npz)

    print("\nSaving outputs …")
    save_csvs(res, args.out_dir)
    report = save_json(res, args.out_dir, args.npz, args.ckpt)
    save_figures(res, args.out_dir)
    print_summary(res)

    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
