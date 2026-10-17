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
stats = dict(dataset.dataset.meta.stats)

# The pretrained SmolVLA checkpoint uses generic camera slots.  MIKASA has two
# physical cameras with different names, so make this mapping explicit rather
# than relying on an implicit processor rename.
camera_map = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
for source_key, target_key in camera_map.items():
    raw_sample[target_key] = raw_sample.pop(source_key)
    if source_key in stats:
        stats[target_key] = stats[source_key]

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", device=str(device))
print("checkpoint image features:", policy.config.image_features)
print("empty camera slots:", policy.config.empty_cameras)
preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
batch = preprocessor(raw_sample)

# The policy processor adds a batch dimension to observations, while an action
# chunk is already a sequence. Add its batch dimension explicitly for the
# training forward contract: [B, H, A] and [B, H].
if batch["action"].ndim == 2:
    batch["action"] = batch["action"].unsqueeze(0)
if batch["action_is_pad"].ndim == 1:
    batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)

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
