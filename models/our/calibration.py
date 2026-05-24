import torch

@torch.no_grad()
def temperature_scale_logits(logits: torch.Tensor, y: torch.Tensor, temps=(0.7,0.8,0.9,1.0,1.1,1.2,1.3)):
    """
    Choose a single scalar temperature T on val to minimize BCE.
    logits: [N,C]  y: [N,C] float {0,1}
    """
    y = (y > 0.5).float()
    best_T = 1.0
    best_loss = 1e18
    for T in temps:
        l = logits / float(T)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(l, y).item()
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T
