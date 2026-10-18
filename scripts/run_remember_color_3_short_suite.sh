#!/usr/bin/env bash
set -euo pipefail

# Capacity-matched short-task suite: baseline, then prefix and expert memory.
# The memory pair starts from the pretrained policy; the baseline checkpoint is
# kept separately for evaluation and optional warm-start ablations.

python -m scripts.train_smolvla_baseline \
  experiment_name=baseline_remember_color_3_short_unfreeze4 \
  data.manifest=outputs/mikasa/remember_color_3_vla_v0/manifest.json \
  trainer.unfreeze_vlm_layers=4 \
  trainer.batch_size=24 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=1000 \
  trainer.validation_every=100 \
  trainer.validation_windows=256 \
  trainer.validation_full_only=false \
  trainer.checkpoint_every=250 \
  trainer.save_checkpoints=true \
  trainer.output_dir=outputs/checkpoints/baseline_remember_color_3_short_unfreeze4

bash scripts/run_short_memory_pair.sh

