"""Run one real SmolVLA flow-matching forward pass on a MIKASA sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from smolvla_data.mikasa_dataset import MikasaWindowDataset


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
print("device:", device)

dataset = MikasaWindowDataset(args.manifest, args.split)
raw_sample = dataset[0]
stats = dataset.dataset.meta.stats

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", device=str(device))
preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
batch = preprocessor(raw_sample)

print("processed batch:")
for key, value in batch.items():
    if hasattr(value, "shape"):
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
    elif isinstance(value, str):
        print(f"  {key}: {value!r}")

policy.eval()
with torch.no_grad():
    loss, loss_dict = policy(batch)

print("loss:", float(loss))
print("loss_dict:", loss_dict)
assert torch.isfinite(loss).item()
print("SmolVLA forward: valid")
