# Experiment runbook

This file lists reproducible commands for the current SmolVLA/MIKASA experiments. It omits machine-specific setup and debugging history.

## Common rules

- `trainer.steps` is the number of optimizer updates, not the number of individual windows.
- `trainer.batch_size` is the number of independent samples processed together. Memory training currently requires `1` because one sample is a temporal sequence.
- `trainer.gradient_accumulation_steps` processes that many sequences before one optimizer update. The effective number of observation windows per update is approximately:

  `batch_size × gradient_accumulation_steps × memory.sequence_length`.

- `memory.sequence_length` is the number of consecutive observation steps passed through the recurrent memory. It is unrelated to the 50-action prediction chunk.
- `memory.sequence_length=-1` means one complete episode, whose length is taken from the dataset.
- `memory.probe_episode_selection=longest` selects the longest episode in the training split. `closest` selects the episode closest to `memory.probe_episode_length`.
- Probe runs use `trainer.steps=1`, disable validation and checkpointing, and are intended to measure whether a sequence fits in VRAM.

## Baseline evaluation

The pretrained baseline was evaluated on 256 representative held-out windows:

```bash
python -m scripts.evaluate_smolvla \
  experiment_name=eval_pretrained_test_representative \
  model.source=pretrained \
  evaluation.split=test \
  evaluation.batch_size=24 \
  evaluation.max_windows=256 \
  evaluation.output=outputs/evaluations/pretrained_test_representative.json
```

The fine-tuned baseline checkpoint is evaluated with the same sampler:

```bash
python -m scripts.evaluate_smolvla \
  experiment_name=eval_finetuned_best_test \
  model.source=checkpoint \
  model.checkpoint_path=outputs/checkpoints/baseline_smolvla_mikasa_long/best.pt \
  evaluation.split=test \
  evaluation.batch_size=24 \
  evaluation.max_windows=256 \
  evaluation.output=outputs/evaluations/finetuned_best_test.json
```

## Memory architecture smoke tests

These runs check tensor shapes, forward/backward, recurrent state passing, and checkpoint construction. They are not research results.

Prefix memory puts memory tokens into the multimodal VLM prefix. The current configuration unfreezes the last four text-transformer VLM layers; the vision encoder remains frozen.

```bash
python -m scripts.train_smolvla_memory \
  experiment_name=memory_prefix_smoke \
  memory.variant=prefix \
  memory.num_tokens=4 \
  memory.sequence_length=2 \
  memory.unfreeze_vlm_layers=4 \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=2 \
  trainer.validation_every=2 \
  trainer.validation_sequences=256 \
  trainer.save_checkpoints=false \
  trainer.output_dir=outputs/checkpoints/memory_prefix_smoke
```

Expert memory puts memory tokens into the action-expert suffix. The VLM remains frozen in this variant.

```bash
python -m scripts.train_smolvla_memory \
  experiment_name=memory_expert_smoke \
  memory.variant=expert \
  memory.num_tokens=4 \
  memory.sequence_length=2 \
  memory.unfreeze_vlm_layers=0 \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=2 \
  trainer.validation_every=2 \
  trainer.validation_sequences=256 \
  trainer.save_checkpoints=false \
  trainer.output_dir=outputs/checkpoints/memory_expert_smoke
```

## Sequence-length VRAM probes

Run one optimizer step on a selected episode. Start with fixed lengths and then try the full episode.

```bash
python -m scripts.train_smolvla_memory \
  experiment_name=memory_prefix_len32_probe \
  memory.variant=prefix \
  memory.num_tokens=4 \
  memory.sequence_length=32 \
  memory.unfreeze_vlm_layers=4 \
  memory.probe_episode_selection=longest \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=1 \
  trainer.validation_every=1000 \
  trainer.save_checkpoints=false \
  trainer.output_dir=outputs/checkpoints/memory_prefix_len32_probe
```

Replace `32` with `8`, `16`, `64`, or `128`. For a full selected episode:

```bash
python -m scripts.train_smolvla_memory \
  experiment_name=memory_prefix_full_longest_probe \
  memory.variant=prefix \
  memory.num_tokens=4 \
  memory.sequence_length=-1 \
  memory.unfreeze_vlm_layers=4 \
  memory.probe_episode_selection=longest \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=1 \
  trainer.validation_every=1000 \
  trainer.save_checkpoints=false \
  trainer.output_dir=outputs/checkpoints/memory_prefix_full_longest_probe
```

To select an episode nearest to a target length:

```bash
memory.probe_episode_selection=closest \
memory.probe_episode_length=500
```

The trainer prints the selected episode index and actual length. Keep that line with the VRAM result.

## First comparable memory training run

After the sequence-length probe succeeds, use the same observation-window budget as the 1000-step baseline. With sequence length 2 and accumulation 8, one optimizer update consumes approximately 16 observation windows, so 1500 optimizer steps is approximately 24,000 windows.

```bash
python -m scripts.train_smolvla_memory \
  experiment_name=memory_prefix_seq2_acc8 \
  memory.variant=prefix \
  memory.num_tokens=4 \
  memory.sequence_length=2 \
  memory.unfreeze_vlm_layers=4 \
  trainer.batch_size=1 \
  trainer.gradient_accumulation_steps=8 \
  trainer.steps=1500 \
  trainer.validation_every=100 \
  trainer.validation_sequences=256 \
  trainer.checkpoint_every=250 \
  trainer.save_checkpoints=true \
  trainer.output_dir=outputs/checkpoints/memory_prefix_seq2_acc8
```

## Short-task rollout suite

Download the short variant of the same task and create its episode-level manifest:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mikasa-robo/mikasa-robo-vla-lerobot",
    repo_type="dataset",
    allow_patterns="remember_color_3_vla_v0/**",
    local_dir="~/datasets/mikasa-robo-vla",
)
PY

python -m scripts.prepare_mikasa \
  data.root="$HOME/datasets/mikasa-robo-vla/remember_color_3_vla_v0" \
  output.manifest=outputs/mikasa/remember_color_3_vla_v0/manifest.json \
  experiment_name=manifest_remember_color_3_short
```

Train its matching baseline with the same 1000-step, batch-24 budget:

```bash
python -m scripts.train_smolvla_baseline \
  experiment_name=baseline_remember_color_3_short \
  data.manifest=outputs/mikasa/remember_color_3_vla_v0/manifest.json \
  trainer.batch_size=24 \
  trainer.gradient_accumulation_steps=1 \
  trainer.steps=1000 \
  trainer.output_dir=outputs/checkpoints/baseline_remember_color_3_short
```

After the baseline checkpoint exists, run the prefix and expert variants sequentially:

```bash
bash scripts/run_short_memory_pair.sh
```

The pair uses full short episodes, four memory tokens, accumulation 8, and 200 optimizer steps. With roughly 15 observation steps per episode, this is approximately the same 24,000-observation budget as the baseline.

