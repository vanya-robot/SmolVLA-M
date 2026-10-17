#!/usr/bin/env python3
"""Download a selected MIKASA-Robo-VLA LeRobotDataset v3 task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download


def normalize_task_name(task: str) -> str:
    """Convert public benchmark names to Hub directory names."""
    task = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", task)
    task = re.sub(r"(?<=[A-Za-z])(?=\\d)", "_", task)
    task = re.sub(r"(?<=\\d)(?=[A-Za-z])", "_", task)
    task = task.replace("-", "_").lower()
    return task.replace("v_l_a", "vla")


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

task_path = normalize_task_name(args.task)
allow_pattern = f"{task_path}/**"
args.output.mkdir(parents=True, exist_ok=True)

print("Downloading:")
print("  repo:", args.repo_id)
print("  requested task:", args.task)
print("  Hub task path:", task_path)
print("  output:", args.output)
print("  pattern:", allow_pattern)

path = snapshot_download(
    repo_id=args.repo_id,
    repo_type="dataset",
    allow_patterns=allow_pattern,
    local_dir=str(args.output),
)

print("Downloaded to:", path)
