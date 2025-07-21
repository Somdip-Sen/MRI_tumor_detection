import torch


def to_single_channel(t: torch.Tensor) -> torch.Tensor:
    """
    Average across RGB channels, keep a singleton channel dimension.
    Input : Tensor [C,H,W]  (C can be 1 or 3)
    Output: Tensor [1,H,W]
    the purpose of this file/function is to allow the worker thread to access this file parallely
    """
    return t.mean(dim=0, keepdim=True)


