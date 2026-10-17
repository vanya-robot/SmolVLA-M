#!/usr/bin/env python3
"""Download selected MIKASA-Robo-VLA LeRobotDataset v3 tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RememberColor3-VLA-v0")
parser.add_argument(
    "--repo-id",
    default="mikasa-robo/mikasa-robo-vla-lerobot",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path.home() / "datasets" / "mikasa-robo-vla",
)
args = parser.parse_args()

allow_pattern = f"{args.task}/**"
args.output.mkdir(parents=True, exist_ok=True)

print("Downloading:")
print("  repo:", args.repo_id)
print("  task:", args.task)
print("  output:", args.output)
print("  pattern:", allow_pattern)

path = snapshot_download(
    repo_id=args.repo_id,
    repo_type="dataset",
    allow_patterns=allow_pattern,
    local_dir=str(args.output),
)

print("Downloaded to:", path)
