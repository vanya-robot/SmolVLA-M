"""Small LeRobot-backed dataset adapter driven by a saved window manifest."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class MikasaWindowDataset(Dataset):
    """Return one observation and a future action chunk for each manifest window."""

    def __init__(self, manifest_path: str | Path, split: str, repo_id: str | None = None):
        from lerobot.datasets import LeRobotDataset

        self.manifest_path = Path(manifest_path).expanduser()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.payload = payload
        self.windows = payload["windows"][split]
        root = Path(payload["root"]).expanduser()
        self.dataset = LeRobotDataset(repo_id or payload["repo_id"], root=root)
        self.horizon = int(payload["horizon"])
        self.episode_lengths = [int(length) for length in payload["episode_lengths"]]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        global_start = int(window["global_start"])
        valid_length = int(window["valid_length"])
        observation = self.dataset[global_start]

        actions = []
        for offset in range(valid_length):
            actions.append(self.dataset[global_start + offset]["action"])
        action = torch.stack(actions, dim=0)
        action_dim = action.shape[-1]
        if valid_length < self.horizon:
            action = torch.cat((action, torch.zeros(self.horizon - valid_length, action_dim, dtype=action.dtype)), dim=0)
        action_is_pad = torch.arange(self.horizon) >= valid_length

        return {
            "observation.images.top": observation["observation.images.top"],
            "observation.images.wrist": observation["observation.images.wrist"],
            "observation.state": observation["observation.state"],
            "action": action,
            "action_is_pad": action_is_pad,
            "task": observation["task"],
            "episode_index": torch.tensor(int(window["episode_index"])),
            "start_in_episode": torch.tensor(int(window["start_in_episode"])),
        }
