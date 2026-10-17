# Project Context

## Goal

Research prototype for adding compact temporal memory to SmolVLA. The first milestone is an engineering baseline on offline robot datasets, followed by controlled memory ablations and benchmark evaluation.

## Decisions

- Base model: Hugging Face LeRobot / SmolVLA.
- First stage: offline datasets and reproducible baseline experiments.
- Start with an external compressed-memory module.
- Ablations: no memory, recent-history baseline, compressed latent memory.
- Later investigate injection into SmolVLA conditioning and the action expert.
- Do not start with RL; establish a supervised/offline baseline first.

## Repository workflow

- `main`: stable baseline.
- `dev`: active development.
- Do not commit datasets, checkpoints or virtual environments.
- Keep dependencies in `requirements-dev.txt` or `pyproject.toml`.
- Record model, dataset, code revision, seed and evaluation protocol for every run.

## Milestones

1. Verify PyTorch, LeRobot and SmolVLA loading.
2. Inspect the LeRobotDataset structure.
3. Implement a dataset adapter and real-batch tests.
4. Implement the memory module with shape and gradient tests.
5. Run the memory-free offline baseline and log metrics.
6. Connect memory to SmolVLA with a small code change.
7. Add ablations and compare task quality, latency, parameter count and memory use.

## Research framing

Study compact, bounded-cost temporal state for small VLAs. The main question is how memory representation, update rule and insertion point affect long-horizon manipulation under partial observability.

## Rule

Keep changes small and runnable. Document tensor shapes, forward path, loss, and trainable/frozen parameters.
