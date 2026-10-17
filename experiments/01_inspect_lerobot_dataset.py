#!/usr/bin/env python3
"""Inspect a local LeRobotDataset v3 task.

This script intentionally reads metadata first and only then opens a few frames.
Run it on the remote machine after downloading one MIKASA task.
"""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--root", type=Path, default=None)
parser.add_argument("--samples", type=int, default=2)
args = parser.parse_args()

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

print(f"repo_id: {args.repo_id}")
print(f"root: {args.root}")

try:
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    print("\n=== metadata ===")
    for name in [
        "info",
        "features",
        "total_episodes",
        "total_frames",
        "fps",
        "robot_type",
    ]:
        value = getattr(metadata, name, "<not exposed>")
        print(f"{name}: {value}")
except Exception as exc:
    print(f"metadata load failed: {type(exc).__name__}: {exc}")
    raise

print("\n=== dataset samples ===")
dataset = LeRobotDataset(args.repo_id, root=args.root)
print("len:", len(dataset))

for index in range(min(args.samples, len(dataset))):
    sample = dataset[index]
    print(f"\n--- sample {index} ---")
    for key, value in sample.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"{key}: type={type(value).__name__}, shape={shape}, dtype={dtype}")
        if key in {"task", "language_instruction"}:
            print("  value:", value)

print("\ninspection complete")
