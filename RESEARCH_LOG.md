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



## Sequence-level training hypotheses

The memory-free baseline treats one observation window as a training sample. Windows are shuffled, and no memory state is carried between them.

A memory sample is an ordered sequence of windows from one episode. At each observation step, the model receives one observation and predicts an action chunk of shape 50 x 7; the resulting memory state is carried to the next observation. For full-episode training, the sequence starts at the episode reset and contains all observation steps until termination. The loss is computed at each step, while the episode-level memory is reset only at the beginning.

The first short-task experiment uses full episodes because the relevant clue appears near the beginning and the episodes fit the available training memory. Starting a sequence at an arbitrary middle observation resets memory without the earlier clue and therefore tests a different problem.

A later ablation will compare:

1. full-episode sequences;
2. local subsequences starting at random episode positions;
3. burn-in sequences that start earlier but apply action loss only after a chosen delay;
4. a mixture of full and local sequences.

The local-subsequence condition may improve robustness and encourage memory slots to represent multiple kinds of information, but it also changes the task distribution and can make the initial memory state unrealistic. It is therefore a follow-up hypothesis, not part of the first comparison.

Full-episode training can overfit trajectory-specific patterns. Episode-level held-out validation and test, multiple seeds, and rollout success rate are required to distinguish useful memory from scene memorization.
