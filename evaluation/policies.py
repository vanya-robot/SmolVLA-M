"""Model loading boundary used by evaluators.

Future memory variants should expose the same ``forward(batch)`` contract and
can be registered here without changing dataset or metric code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from smolvla_data.policy_factory import load_smolvla


def load_policy(cfg, device: torch.device):
    if cfg.name != "smolvla":
        raise ValueError(f"Unknown evaluation policy: {cfg.name}")
    action_dim = int(cfg.get("action_dim", 7))
    policy = load_smolvla(cfg.pretrained_id, str(device), action_dim)
    if cfg.source == "checkpoint":
        if not cfg.checkpoint_path:
            raise ValueError("model.checkpoint_path is required when model.source=checkpoint")
        checkpoint = torch.load(Path(cfg.checkpoint_path).expanduser(), map_location="cpu")
        policy.load_state_dict(checkpoint["model"], strict=True)
    elif cfg.source != "pretrained":
        raise ValueError(f"Unknown model source: {cfg.source}")
    policy.eval()
    return policy


def model_metadata(cfg, policy) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "source": cfg.source,
        "pretrained_id": cfg.pretrained_id,
        "checkpoint_path": str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
        "chunk_size": int(policy.config.chunk_size),
        "n_obs_steps": int(policy.config.n_obs_steps),
    }
