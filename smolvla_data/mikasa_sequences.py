"""Contiguous episode sequences for recurrent-memory training/evaluation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset

from .mikasa_dataset import MikasaWindowDataset


class MikasaSequenceDataset(Dataset):
    def __init__(self, manifest_path, split: str, sequence_length: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.base = MikasaWindowDataset(manifest_path, split)
        self.sequence_length = sequence_length
        by_episode: dict[int, dict[int, int]] = defaultdict(dict)
        for index, window in enumerate(self.base.windows):
            by_episode[int(window["episode_index"])][int(window["start_in_episode"])] = index
        self.starts: list[tuple[int, int]] = []
        for episode, starts in sorted(by_episode.items()):
            max_start = max(starts) if starts else -1
            for start in range(max_start + 1):
                if all(start + offset in starts for offset in range(sequence_length)):
                    self.starts.append((episode, start))
        self.index = by_episode

    @property
    def dataset(self):
        return self.base.dataset

    @property
    def payload(self):
        return self.base.payload

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        episode, start = self.starts[index]
        return [self.base[self.index[episode][start + offset]] for offset in range(self.sequence_length)]

