import torch
from typing import Tuple

def multilabel_confusion(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    y_true,y_pred: [N,C] {0,1}
    returns conf: [C,4] TP,FP,FN,TN
    """
    y_true = y_true.int()
    y_pred = y_pred.int()
    N, C = y_true.shape
    conf = torch.zeros((C, 4), dtype=torch.long)
    for c in range(C):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        conf[c] = torch.stack([tp, fp, fn, tn]).long()
    return conf

def f1_from_conf(conf: torch.Tensor) -> torch.Tensor:
    """
    conf: [C,4] TP,FP,FN,TN
    returns f1: [C]
    """
    tp = conf[:, 0].float()
    fp = conf[:, 1].float()
    fn = conf[:, 2].float()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    return f1

def multilabel_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true.int() == y_pred.int()).float().mean().item()

def weighted_f1(conf: torch.Tensor) -> float:
    """
    Weight per-label F1 by positive support (TP+FN).
    """
    support = (conf[:, 0] + conf[:, 2]).float()
    f1 = f1_from_conf(conf)
    w = support / (support.sum() + 1e-9)
    return float((w * f1).sum().item())

def compute_acc_wf1(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    conf = multilabel_confusion(y_true, y_pred)
    acc = multilabel_acc(y_true, y_pred)
    wf1 = weighted_f1(conf)
    return acc, wf1, conf
