import torch

import torch.nn.functional as F
from torch import nn


class L2Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print("L2 Loss input:", input)
        print("L2 Loss input shape:", input.shape)

        print("L2 Loss target:", target)
        print("L2 Loss target shape:", target.shape)
        unreduced_loss = F.mse_loss(input=input, target=target, reduction="none")
        return unreduced_loss.mean(list(range(1, len(input.shape))))
