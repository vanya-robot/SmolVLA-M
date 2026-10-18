# SmolVLA memory experiments

## Current agenda

1. Establish a reproducible SmolVLA baseline on the MIKASA `remember_color_3_long_vla_v0` task.
2. Implement one minimal recurrent memory variant with the same data split, optimizer budget, and evaluator.
3. Compare baseline and memory by rollout success rate in simulation; use action loss only as a training diagnostic.
4. If the first memory variant is useful, evaluate it on additional MIKASA memory tasks and prepare a broader ablation.

## Dataset protocol

- Dataset: `mikasa-robo/mikasa-robo-vla-lerobot`
- Task: `remember_color_3_long_vla_v0`
- Episode split: 175 train / 37 validation / 38 test episodes
- Window horizon: 50 actions
- Evaluation: 256 full, stratified windows from the held-out split, fixed seed `20261018`
- The train/validation/test split is episode-level; windows from one episode are not distributed across splits.

## Baseline

The baseline starts from `lerobot/smolvla_base`. The vision encoder and VLM are frozen; only the action-expert parameters are optimized. The policy receives one observation step (`n_obs_steps=1`) and predicts an action chunk of length 50.

### Offline diagnostics

These numbers are flow-matching losses, not task success rates.

| Model | Validation loss | Test loss | Windows |
|---|---:|---:|---:|
| Pretrained SmolVLA | 2.323881 | 2.294195 | 256 |
| Fine-tuned baseline (`best.pt`) | 0.386753 | 0.383033 | 256 |

The fine-tuned checkpoint was selected by held-out validation loss. The validation and test losses are close, which is a useful sanity check, but the decisive metric for this project is rollout success rate.

## First memory experiment

The first modification will be a minimal recurrent memory module. It will receive the current SmolVLA context at environment step `t`, update a fixed-size memory state, and expose that state to the action expert. The base vision encoder and VLM remain frozen. We will train only the memory module and action expert, using the same MIKASA split and training budget as the baseline.

The comparison will answer one narrow question: does a compact learned temporal state improve successful execution when the relevant cue is no longer available in the current observation?

The implementation provides two placements: prefix memory and action-expert suffix memory. It supports contiguous sequence training, deterministic episode probes, and configurable unfreezing of the final VLM text layers for the prefix variant.

No claim about either memory variant is made until it is evaluated in rollout.

## Compute observation

Full backpropagation through a 92–599-step long episode exceeded available training memory with four unfrozen VLM text layers, even at batch size one. This is a training-graph limitation: inference does not retain gradients or the temporal graph, and carries only the current memory state between policy calls. The long task will be revisited with truncated backpropagation or a lower-cost memory writer.

## Interpretation rule

Action loss is used to debug optimization and compare training dynamics. Final conclusions about memory use success rate in simulation, with identical seeds, task episodes, and rollout protocol for all models.

