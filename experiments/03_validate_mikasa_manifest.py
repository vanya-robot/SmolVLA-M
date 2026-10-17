"""Validate a generated MIKASA manifest before any model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
payload = json.loads(args.manifest.read_text(encoding="utf-8"))
lengths = payload["episode_lengths"]
split = payload["split"]
groups = [set(split["train"]), set(split["validation"]), set(split["test"])]
assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
assert sum(map(len, groups)) == len(lengths)

for name, windows in payload["windows"].items():
    for window in windows:
        episode = window["episode_index"]
        start = window["start_in_episode"]
        valid = window["valid_length"]
        horizon = window["horizon"]
        assert 0 <= episode < len(lengths)
        assert 0 <= start < lengths[episode]
        assert 1 <= valid <= horizon
        assert valid == min(horizon, lengths[episode] - start)

print("manifest: valid")
print("episodes:", {name: len(ids) for name, ids in split.items()})
print("windows:", payload["window_counts"])
