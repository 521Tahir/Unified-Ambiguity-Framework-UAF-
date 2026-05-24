"""
Unified classification metrics for single-label (IEMOCAP, MELD) and
multi-label (CMU-MOSEI) tasks.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from typing import Dict, List, Optional


def single_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    For IEMOCAP (4-class) and MELD (7-class).
    Returns: acc, macro_f1, weighted_f1, macro_precision, macro_recall, per_class, confusion_matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]

    per_class = {}
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-12, p + r)
        name = class_names[i] if class_names else str(i)
        per_class[name] = {"precision": p, "recall": r, "f1": f, "support": int(tp + fn)}

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    For CMU-MOSEI (6-class multi-label).
    y_true, y_pred: [N, C] binary arrays.
    Returns: subset_acc, macro_f1, weighted_f1, per_class.
    """
    # Subset accuracy: all labels must match
    subset_acc = float((y_true == y_pred).all(axis=1).mean())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    n_classes = y_true.shape[1]
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i in range(n_classes):
        support = int(y_true[:, i].sum())
        name = class_names[i] if class_names else str(i)
        per_class[name] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1": float(per_class_f1[i]),
            "support": support,
        }

    # WAcc: average binary per-class accuracy (TP+TN)/N — same metric as MESCA paper Table 3
    binary_acc_per_class = []
    for i in range(n_classes):
        correct = ((y_true[:, i] == y_pred[:, i])).sum()
        binary_acc_per_class.append(correct / len(y_true))
    wacc = float(np.mean(binary_acc_per_class))

    return {
        "subset_accuracy": subset_acc,
        "wacc": wacc,                 # paper's WAcc metric
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class": per_class,
        "binary_acc_per_class": {
            (class_names[i] if class_names else str(i)): float(binary_acc_per_class[i])
            for i in range(n_classes)
        },
    }


def print_metrics(metrics: Dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    if "accuracy" in metrics:
        print(f"{tag}Accuracy:          {metrics['accuracy']:.4f}")
    if "subset_accuracy" in metrics:
        print(f"{tag}Subset Accuracy:   {metrics['subset_accuracy']:.4f}")
    if "wacc" in metrics:
        print(f"{tag}WAcc (paper):      {metrics['wacc']:.4f}")
    print(f"{tag}Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"{tag}Weighted-F1:       {metrics['weighted_f1']:.4f}")
    print(f"{tag}Macro-Precision:   {metrics['macro_precision']:.4f}")
    print(f"{tag}Macro-Recall:      {metrics['macro_recall']:.4f}")
    if "per_class" in metrics:
        print(f"{tag}Per-class F1:")
        for name, v in metrics["per_class"].items():
            print(f"  {name:12s}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}  support={v['support']}")
