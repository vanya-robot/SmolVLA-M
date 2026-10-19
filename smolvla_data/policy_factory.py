"""Construct SmolVLA with the action feature of the target dataset."""
from __future__ import annotations

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def load_smolvla(pretrained_id: str, device: str, action_dim: int) -> SmolVLAPolicy:
    """Load pretrained weights while overriding the output action feature.

    The public ``lerobot/smolvla_base`` config describes a 6D SO-100 action
    space. MIKASA uses a 7D end-effector action space, so this override must be
    applied before training or inference. The underlying projection is padded
    to ``max_action_dim`` and remains compatible with the pretrained weights.
    """
    config = SmolVLAConfig.from_pretrained(pretrained_id)
    config.output_features = dict(config.output_features)
    config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(int(action_dim),),
    )
    config.device = device
    return SmolVLAPolicy.from_pretrained(pretrained_id, config=config, device=device)
