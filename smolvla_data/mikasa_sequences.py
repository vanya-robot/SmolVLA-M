"""Contiguous episode sequences for recurrent-memory training/evaluation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset

from .mikasa_dataset import MikasaWindowDataset


class MikasaSequenceDataset(Dataset):
    def __init__(self, manifest_path, split: str, sequence_length: int):
        self.base = MikasaWindowDataset(manifest_path, split)
        self.sequence_length = sequence_length
        by_episode: dict[int, dict[int, int]] = defaultdict(dict)
        for index, window in enumerate(self.base.windows):
            by_episode[int(window["episode_index"])][int(window["start_in_episode"])] = index
        self.starts: list[tuple[int, int]] = []
        self.ranges: list[list[int]] = []
        for episode, starts in sorted(by_episode.items()):
            max_start = max(starts) if starts else -1
            if sequence_length <= 0:
                offsets = sorted(starts)
                if offsets:
                    self.starts.append((episode, offsets[0]))
                    self.ranges.append(offsets)
            else:
                for start in range(max_start + 1):
                    offsets = [start + offset for offset in range(sequence_length)]
                    if all(offset in starts for offset in offsets):
                        self.starts.append((episode, start))
                        self.ranges.append(offsets)
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
        episode, _start = self.starts[index]
        return [self.base[self.index[episode][offset]] for offset in self.ranges[index]]

    def representative_indices(self, max_sequences: int) -> list[int]:
        """Select sequence starts across episodes, avoiding a prefix-only eval."""
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, (episode, _start) in enumerate(self.starts):
            grouped[episode].append(index)
        selected: list[int] = []
        episodes = sorted(grouped)
        per_episode = max(1, max_sequences // max(1, len(episodes)))
        for episode in episodes:
            candidates = grouped[episode]
            count = min(per_episode, len(candidates))
            if count == 1:
                positions = [len(candidates) // 2]
            else:
                positions = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
            selected.extend(candidates[position] for position in positions)
        return selected[:max_sequences]

