"""Deterministic, episode-stratified evaluation window selection."""
from __future__ import annotations

from collections import defaultdict


def select_stratified_window_indices(
    payload: dict,
    split: str,
    max_windows: int,
    *,
    full_only: bool = True,
) -> list[int]:
    """Select windows across all episodes instead of taking the manifest prefix."""
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, window in enumerate(payload["windows"][split]):
        if full_only and int(window["valid_length"]) < int(window["horizon"]):
            continue
        grouped[int(window["episode_index"])].append(index)
    if not grouped:
        raise ValueError(f"No eligible windows for split={split!r}.")

    episode_ids = sorted(grouped)
    per_episode = max(1, max_windows // len(episode_ids))
    selected: list[int] = []
    for episode_id in episode_ids:
        candidates = grouped[episode_id]
        count = min(per_episode, len(candidates))
        if count == 1:
            positions = [len(candidates) // 2]
        else:
            positions = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
        selected.extend(candidates[position] for position in positions)

    if len(selected) < max_windows:
        remaining = [
            index
            for episode_id in episode_ids
            for index in grouped[episode_id]
            if index not in set(selected)
        ]
        selected.extend(remaining[: max_windows - len(selected)])
    return selected[:max_windows]
