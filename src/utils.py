from typing import Tuple

import numpy as np

import torch

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def to_tensor(
    s: np.ndarray | torch.Tensor,
    obs_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(s, np.ndarray):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    elif isinstance(s, torch.Tensor):
        s_t = s.to(device=device, dtype=torch.float32)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(s)}")

    if s_t.dim() == len(obs_shape):
        s_t = s_t.unsqueeze(0)

    return s_t