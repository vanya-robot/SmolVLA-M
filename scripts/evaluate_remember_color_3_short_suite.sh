#!/usr/bin/env bash
set -euo pipefail

# Evaluate the three trained policies on the same held-out test split.
# This measures flow-matching loss. Task success still requires simulator rollout.

MANIFEST="${MANIFEST:-outputs/mikasa/remember_color_3_vla_v0/manifest.json}"
WINDOWS="${WINDOWS:-256}"
SEQUENCES="${SEQUENCES:-256}"
BATCH_SIZE="${BATCH_SIZE:-24}"

BASELINE_CKPT="${BASELINE_CKPT:-outputs/checkpoints/baseline_remember_color_3_short_unfreeze4/best.pt}"
PREFIX_CKPT="${PREFIX_CKPT:-outputs/checkpoints/memory_prefix_remember_color_3_short_full/best.pt}"
EXPERT_CKPT="${EXPERT_CKPT:-outputs/checkpoints/memory_expert_remember_color_3_short_full/best.pt}"

for checkpoint in "$BASELINE_CKPT" "$PREFIX_CKPT" "$EXPERT_CKPT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

python -m scripts.evaluate_smolvla \
  experiment_name=eval_baseline_remember_color_3_short_test \
  data.manifest="$MANIFEST" \
  model.source=checkpoint \
  model.checkpoint_path="$BASELINE_CKPT" \
  evaluation.split=test \
  evaluation.batch_size="$BATCH_SIZE" \
  evaluation.max_windows="$WINDOWS" \
  evaluation.full_only=false \
  evaluation.output=outputs/evaluations/baseline_remember_color_3_short_test.json

python -m scripts.evaluate_smolvla_memory \
  experiment_name=eval_memory_prefix_remember_color_3_short_test \
  data.manifest="$MANIFEST" \
  memory.variant=auto \
  memory.sequence_length=-1 \
  evaluation.split=test \
  evaluation.max_sequences="$SEQUENCES" \
  evaluation.checkpoint_path="$PREFIX_CKPT" \
  evaluation.output=outputs/evaluations/memory_prefix_remember_color_3_short_test.json

python -m scripts.evaluate_smolvla_memory \
  experiment_name=eval_memory_expert_remember_color_3_short_test \
  data.manifest="$MANIFEST" \
  memory.variant=auto \
  memory.sequence_length=-1 \
  evaluation.split=test \
  evaluation.max_sequences="$SEQUENCES" \
  evaluation.checkpoint_path="$EXPERT_CKPT" \
  evaluation.output=outputs/evaluations/memory_expert_remember_color_3_short_test.json
