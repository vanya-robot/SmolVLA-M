#!/usr/bin/env bash
set -euo pipefail

# Train the two memory placements sequentially on full short episodes.
# The short baseline checkpoint must exist before this script starts.

SHORT_MANIFEST="${SHORT_MANIFEST:-outputs/mikasa/remember_color_3_vla_v0/manifest.json}"

python -m scripts.train_smolvla_memory \
  experiment_name=memory_prefix_remember_color_3_short_full \
  data.manifest="$SHORT_MANIFEST" \
  pretrained_policy=lerobot/smolvla_base \
  memory.variant=prefix \
  memory.num_tokens=4 \
  memory.sequence_length=-1 \
  memory.unfreeze_vlm_layers=4 \
  memory.probe_episode_selection=random \
  trainer.initialization_checkpoint="" \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=8 \
  trainer.steps=200 \
  trainer.validation_every=25 \
  trainer.validation_sequences=256 \
  trainer.checkpoint_every=100 \
  trainer.save_checkpoints=true \
  trainer.output_dir=outputs/checkpoints/memory_prefix_remember_color_3_short_full

python -m scripts.train_smolvla_memory \
  experiment_name=memory_expert_remember_color_3_short_full \
  data.manifest="$SHORT_MANIFEST" \
  pretrained_policy=lerobot/smolvla_base \
  memory.variant=expert \
  memory.num_tokens=4 \
  memory.sequence_length=-1 \
  memory.unfreeze_vlm_layers=4 \
  memory.probe_episode_selection=random \
  trainer.initialization_checkpoint="" \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=8 \
  trainer.steps=200 \
  trainer.validation_every=25 \
  trainer.validation_sequences=256 \
  trainer.checkpoint_every=100 \
  trainer.save_checkpoints=true \
  trainer.output_dir=outputs/checkpoints/memory_expert_remember_color_3_short_full

