"""Reproducible episode splits and action-window manifests for MIKASA."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


@dataclass(frozen=True)
class Window:
    episode_index: int
    start_in_episode: int
    global_start: int
    horizon: int
    valid_length: int


def make_episode_split(episode_count: int, seed: int, train_fraction: float, validation_fraction: float) -> EpisodeSplit:
    if episode_count < 3:
        raise ValueError("At least three episodes are required.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid train/validation fractions.")
    ids = list(range(episode_count))
    random.Random(seed).shuffle(ids)
    n_train = max(1, int(episode_count * train_fraction))
    n_val = max(1, int(episode_count * validation_fraction))
    return EpisodeSplit(
        train=sorted(ids[:n_train]),
        validation=sorted(ids[n_train:n_train + n_val]),
        test=sorted(ids[n_train + n_val:]),
    )


def make_windows(episode_lengths: list[int], episode_ids: list[int], horizon: int, pad_to_horizon: bool) -> list[Window]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cumulative = [0]
    for length in episode_lengths:
        cumulative.append(cumulative[-1] + length)
    windows: list[Window] = []
    for episode_index in episode_ids:
        length = episode_lengths[episode_index]
        count = length if pad_to_horizon else max(0, length - horizon + 1)
        for start in range(count):
            windows.append(Window(
                episode_index=episode_index,
                start_in_episode=start,
                global_start=cumulative[episode_index] + start,
                horizon=horizon,
                valid_length=min(horizon, length - start),
            ))
    return windows


def write_manifest(path: Path, *, repo_id: str, root: str, seed: int, horizon: int,
                   pad_to_horizon: bool, episode_lengths: list[int], split: EpisodeSplit,
                   windows: dict[str, list[Window]], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "repo_id": repo_id,
        "root": root,
        "seed": seed,
        "horizon": horizon,
        "pad_to_horizon": pad_to_horizon,
        "episode_lengths": episode_lengths,
        "split": asdict(split),
        "window_counts": {name: len(items) for name, items in windows.items()},
        "windows": {name: [asdict(item) for item in items] for name, items in windows.items()},
    }
    if extra:
        payload["dataset"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
