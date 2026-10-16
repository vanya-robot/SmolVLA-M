# Project Context

## Goal

Research prototype for adding compact temporal memory to SmolVLA. First milestone: an engineering baseline tested on offline robot datasets and later trained on one RTX 4060 8 GB.

## Decisions

- Base model: Hugging Face LeRobot / SmolVLA.
- First stage: offline datasets and CPU smoke tests; no simulator or physical robot required.
- Start with an external compressed-memory module.
- Ablations: no memory, recent-history baseline, compressed latent memory.
- Later investigate injection into SmolVLA conditioning/action expert.
- Do not start with RL; establish a supervised/offline baseline first.

## Hardware

- Remote: Windows 11 + WSL2 Ubuntu 24.04.
- GPU: NVIDIA RTX 4060 8 GB.
- Access: SSH through Tailscale to Windows, then WSL.
- Cloud/local CPU is for code checks and small smoke tests.

## Repository workflow

- main: stable baseline.
- dev: active development.
- Do not commit datasets, checkpoints or virtual environments.
- Keep dependencies in requirements-dev.txt or pyproject.toml.

## Milestones

1. Verify PyTorch and CUDA remotely.
2. Inspect LeRobotDataset structure.
3. Implement dataset adapter and synthetic sequence tests.
4. Implement memory module with shape and gradient tests.
5. Run offline baseline and log metrics.
6. Connect memory to SmolVLA with a small code change.
7. Add ablations and compare quality, latency, parameters and GPU memory.

## Research framing

Study compact, bounded-cost temporal state for small VLAs, measuring task quality, latency, parameter count and GPU memory. Learned gating/retrieval comes later.

## Rule

Keep changes small and runnable. Document tensor shapes, forward path, loss, and trainable/frozen parameters.
