# SmolVLA-M research notebook

## Current agenda

**Status: baseline complete; memory placement and sequence-length probes in progress.**

1. Reproduce the memory-free SmolVLA baseline on MIKASA.
2. Add recurrent memory tokens to the VLM prefix and action-expert suffix.
3. Compare the variants in simulation by task success rate.
4. Revisit the long-horizon experiment with truncated backpropagation or a lightweight memory writer if full unrolling is too expensive.

## System under study

SmolVLA combines a pretrained visual-language backbone with a continuous flow-matching action expert. The policy receives camera images, language, and robot state, then predicts a 50-step action chunk. Memory is carried between physical observation steps and remains fixed during the internal flow-denoising steps for one action chunk.

## Dataset

Main task: remember_color_3_long_vla_v0 from mikasa-robo/mikasa-robo-vla-lerobot.

| Property | Value |
|---|---:|
| Train / validation / test episodes | 175 / 37 / 38 |
| Observation rate | 10 Hz |
| Episode length | 92–599 frames |
| Observation cameras | top, wrist |
| State / action dimension | 7 / 7 |
| Action chunk | 50 |

The split is episode-level and evaluation windows come from held-out episodes.

## Baseline

The pretrained vision encoder and VLM backbone are frozen. The action-expert path and state/action projection layers are trainable.

| Model | Validation loss | Test loss |
|---|---:|---:|
| Pretrained SmolVLA | 2.323881 | 2.294195 |
| Fine-tuned baseline | 0.386753 | 0.383033 |

The loss is an optimization diagnostic. Success rate in rollout is the decision metric.

## Memory implementations

### Prefix

At observation step t, the input sequence is image, language, state, and memory tokens. The output at the memory positions becomes the memory state for step t+1. The current implementation can unfreeze the last N VLM text layers while keeping the vision encoder frozen.

### Expert suffix

The VLM receives only the current observation. The action-expert input contains memory tokens followed by noisy action tokens. The output at the memory positions becomes the next memory state. The VLM remains frozen.

## Compute finding

Full backpropagation through a 92–599-step episode retained activations for every temporal step and exceeded available training memory with four trainable VLM text layers at batch size one. This is a training limitation. Inference does not retain the training graph or gradients; it carries the current memory state between policy calls and computes only the current observation/action call.

The long task remains the target for the memory question. Short tasks are useful for validating rollout and success-rate measurement. Long-task training will use truncated backpropagation or a lower-cost trainable memory writer.

## Results policy

Every comparison uses the same episode split, seed, action horizon, rollout protocol, and success-rate definition. Flow loss may diagnose training, but cannot establish that memory improved the task.
