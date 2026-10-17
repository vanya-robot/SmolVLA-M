"""Decode a tiny batch from a saved MIKASA manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from smolvla_data.mikasa_dataset import MikasaWindowDataset


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
parser.add_argument("--batch-size", type=int, default=2)
args = parser.parse_args()

dataset = MikasaWindowDataset(args.manifest, args.split)
batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))

print("split:", args.split)
for key, value in batch.items():
    if hasattr(value, "shape"):
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    elif isinstance(value, list):
        print(f"{key}: list[{len(value)}] first={value[0]!r}")
    else:
        print(f"{key}: {value!r}")

assert batch["observation.images.top"].ndim == 4
assert batch["observation.images.wrist"].ndim == 4
assert batch["observation.state"].shape[-1] == 7
assert batch["action"].shape[1:] == (50, 7)
assert batch["action_is_pad"].shape[1:] == (50,)
print("batch contract: valid")
