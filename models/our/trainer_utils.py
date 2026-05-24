import torch
from models.chado.metrics import compute_acc_wf1, f1_from_conf


def apply_thresholds(probs: torch.Tensor, thr_vec: torch.Tensor) -> torch.Tensor:
    """
    probs: [N,C] in [0,1]
    thr_vec: [C]
    returns pred: [N,C] {0,1}
    """
    if not torch.is_tensor(thr_vec):
        thr_vec = torch.tensor(thr_vec, dtype=probs.dtype)
    thr = thr_vec.view(1, -1)
    return (probs >= thr).int()


@torch.no_grad()
def tune_thresholds(
    logits: torch.Tensor,
    y: torch.Tensor,
    grid=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
    objective: str = "acc",
) -> torch.Tensor:
    """
    FP-aware per-class threshold tuning.

    objective:
      - "acc": maximize multilabel subset accuracy, tie-break with WF1.
      - "wf1": maximize WF1, tie-break with accuracy.

    Implementation: coordinate ascent on chosen objective.
    This is robust on MOSEI where naive per-class F1 often collapses into low thresholds and high FP.
    """
    probs = torch.sigmoid(logits)
    y_true = (y > 0.5).int()

    N, C = probs.shape
    thr = torch.full((C,), 0.5, dtype=torch.float32)

    def score(thr_vec):
        pred = apply_thresholds(probs, thr_vec)
        acc, wf1, conf = compute_acc_wf1(y_true, pred)
        if objective == "wf1":
            return (wf1, acc)
        if objective == "macro_f1":
            mf1 = float(f1_from_conf(conf).mean())
            return (mf1, wf1)
        return (acc, wf1)

    best = score(thr)

    # 3 passes is still cheap and converges well
    for _ in range(3):
        improved = False
        for c in range(C):
            best_t = float(thr[c].item())
            best_sc = best

            # search grid for this class while holding others fixed
            for t in grid:
                thr_try = thr.clone()
                thr_try[c] = float(t)
                sc = score(thr_try)
                if sc > best_sc:
                    best_sc = sc
                    best_t = float(t)

            if best_t != float(thr[c].item()):
                thr[c] = best_t
                best = best_sc
                improved = True

        if not improved:
            break

    return thr
