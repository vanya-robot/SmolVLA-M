from __future__ import annotations

import torch
from torch import nn


class CompressedMemory(nn.Module):
    """Сжимает последние K векторов наблюдений в один memory-вектор.

    На первом этапе это самостоятельный модуль: его можно тестировать без
    SmolVLA, а затем подключить к conditioning action expert.
    """

    def __init__(self, input_dim: int, memory_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: [batch, time, input_dim] -> memory: [batch, memory_dim]."""
        if history.ndim != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(history.shape)}")
        encoded = self.encoder(history)
        return encoded.mean(dim=1)
