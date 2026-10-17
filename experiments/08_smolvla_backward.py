"""Check masked-tail loss and gradients for one SmolVLA training step."""
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
parser.add_argument("--index", type=int, default=-1, help="Window index; -1 selects the final window.")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=20261018)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
dataset = MikasaWindowDataset(args.manifest, args.split)
raw_sample = dataset[args.index]
stats = dict(dataset.dataset.meta.stats)

camera_map = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
for source_key, target_key in camera_map.items():
    raw_sample[target_key] = raw_sample.pop(source_key)
    if source_key in stats:
        stats[target_key] = stats[source_key]

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", device=str(device))
preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
batch = preprocessor(raw_sample)
if batch["action"].ndim == 2:
    batch["action"] = batch["action"].unsqueeze(0)
if batch["action_is_pad"].ndim == 1:
    batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)

valid_actions = int((~batch["action_is_pad"]).sum().item())
print("device:", device)
print("split:", args.split, "index:", args.index)
print("valid actions:", valid_actions, "/", batch["action_is_pad"].shape[-1])

policy.train()
loss, loss_dict = policy(batch)
loss.backward()

trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
with_grad = [name for name, parameter in trainable if parameter.grad is not None]
print("loss:", float(loss.detach()))
print("loss_dict:", loss_dict)
print("trainable parameters:", len(trainable))
print("parameters with gradients:", len(with_grad))
print("gradient examples:", with_grad[:12])
assert torch.isfinite(loss).item()
assert with_grad, "No trainable parameter received a gradient."
print("SmolVLA backward: valid")
