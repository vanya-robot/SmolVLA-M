"""Recurrent memory-token adapters for the LeRobot SmolVLA policy.

The adapter keeps the pretrained SmolVLA implementation intact and adds a
small recurrent state.  ``prefix`` places memory tokens in the multimodal
prefix; ``expert`` places them in the action-expert suffix.  The state is
updated from the corresponding output positions after each environment step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.common.vla_utils import make_att_2d_masks


@dataclass(frozen=True)
class MemorySpec:
    variant: str
    num_tokens: int = 4

    def __post_init__(self) -> None:
        if self.variant not in {"prefix", "expert"}:
            raise ValueError("Memory variant must be 'prefix' or 'expert'.")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")


class MemorySmolVLAPolicy(nn.Module):
    """SmolVLA plus recurrent memory tokens.

    ``forward_step`` consumes one already-preprocessed observation and returns
    a flow-matching loss plus the memory state for the next observation.
    ``memory`` is a runtime tensor, not a model parameter.  It is reset at the
    beginning of every episode with zeros.
    """

    def __init__(self, base_policy: SmolVLAPolicy, spec: MemorySpec, unfreeze_vlm_layers: int = 0):
        super().__init__()
        self.base_policy = base_policy
        self.spec = spec
        model = self.base_policy.model
        if unfreeze_vlm_layers < 0:
            raise ValueError("unfreeze_vlm_layers must be non-negative")
        self.unfreeze_vlm_layers = int(unfreeze_vlm_layers)
        if self.unfreeze_vlm_layers:
            self._unfreeze_last_vlm_layers(self.unfreeze_vlm_layers)
        self.vlm_dim = int(model.vlm_with_expert.config.text_config.hidden_size)
        self.expert_dim = int(model.vlm_with_expert.expert_hidden_size)
        self.register_buffer("zero_memory_vlm", torch.zeros(spec.num_tokens, self.vlm_dim), persistent=False)
        self.register_buffer("zero_memory_expert", torch.zeros(spec.num_tokens, self.expert_dim), persistent=False)

    @property
    def model(self):
        return self.base_policy.model

    def _unfreeze_last_vlm_layers(self, count: int) -> None:
        """Unfreeze only the final text-transformer layers for prefix memory."""
        text_model = self.model.vlm_with_expert.get_vlm_model().text_model
        for parameter in self.model.vlm_with_expert.vlm.parameters():
            parameter.requires_grad = False
        count = min(count, len(text_model.layers))
        for layer in text_model.layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    @property
    def config(self):
        return self.base_policy.config

    def initial_memory(self, batch_size: int, device: torch.device) -> Tensor:
        template = self.zero_memory_vlm if self.spec.variant == "prefix" else self.zero_memory_expert
        return template.to(device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

    def reset(self) -> None:
        """Reset the wrapped policy state at the beginning of an episode."""
        self.base_policy.reset()

    def _prefix_with_memory(self, images, img_masks, lang_tokens, lang_masks, state, memory):
        prefix, pad_mask, att_mask = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        memory = memory.to(dtype=prefix.dtype)
        prefix = torch.cat([prefix, memory], dim=1)
        memory_pad = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        memory_att = torch.ones(memory.shape[:2], dtype=att_mask.dtype, device=memory.device)
        return (
            prefix,
            torch.cat([pad_mask, memory_pad], dim=1),
            torch.cat([att_mask, memory_att], dim=1),
        )

    def _suffix_with_memory(self, noisy_actions, timestep, memory):
        suffix, pad_mask, att_mask = self.model.embed_suffix(noisy_actions, timestep)
        memory = memory.to(dtype=suffix.dtype)
        suffix = torch.cat([memory, suffix], dim=1)
        memory_pad = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        memory_att = torch.ones(memory.shape[:2], dtype=att_mask.dtype, device=memory.device)
        return (
            suffix,
            torch.cat([memory_pad, pad_mask], dim=1),
            torch.cat([memory_att, att_mask], dim=1),
        )

    def forward_step(self, batch: dict[str, Tensor], memory: Tensor) -> tuple[Tensor, Tensor]:
        images, img_masks = self.base_policy.prepare_images(batch)
        state = self.base_policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.base_policy.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        x_t, target_velocity, _ = make_flow_matching_inputs(
            actions, noise, time, convention=FlowConvention.NOISE_AT_ONE
        )

        if self.spec.variant == "prefix":
            prefix, prefix_pad, prefix_att = self._prefix_with_memory(
                images, img_masks, lang_tokens, lang_masks, state, memory
            )
            suffix, suffix_pad, suffix_att = self.model.embed_suffix(x_t, time)
            memory_slice = slice(-self.spec.num_tokens, None)
        else:
            prefix, prefix_pad, prefix_att = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            suffix, suffix_pad, suffix_att = self._suffix_with_memory(x_t, time, memory)
            memory_slice = slice(0, self.spec.num_tokens)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        attention_mask = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        outputs, _ = self.model.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
        )
        prefix_out, suffix_out = outputs
        if self.spec.variant == "prefix":
            next_memory = prefix_out[:, memory_slice, :]
        else:
            next_memory = suffix_out[:, memory_slice, :]

        action_out = suffix_out[:, -self.config.chunk_size :, :]
        action_out = action_out.to(dtype=self.model.action_out_proj.weight.dtype)
        action_dim = int(self.config.action_feature.shape[0])
        predicted_velocity = self.model.action_out_proj(action_out).float()[:, :, :action_dim]
        losses = F.mse_loss(predicted_velocity, target_velocity[:, :, :action_dim], reduction="none")
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            losses = losses * (~action_is_pad).unsqueeze(-1)
            valid = ((~action_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / valid
        else:
            loss = losses.mean()
        return loss, next_memory

    def forward_sequence(self, sequence: list[dict[str, Tensor]]) -> Tensor:
        if not sequence:
            raise ValueError("Sequence must contain at least one observation.")
        memory = self.initial_memory(1, next(self.parameters()).device)
        losses = []
        for batch in sequence:
            loss, memory = self.forward_step(batch, memory)
            losses.append(loss)
        return torch.stack(losses).mean()

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], memory: Tensor) -> tuple[Tensor, Tensor]:
        """Generate one action chunk and the memory state for the next observation.

        This is the inference counterpart of ``forward_step``.  It follows the
        same flow-matching convention as SmolVLA, but starts from Gaussian noise
        and integrates the learned velocity field from t=1 to t=0.  We keep the
        implementation explicit and do not use the base policy KV cache because
        the inserted memory tokens must participate in every denoising step.
        """
        images, img_masks = self.base_policy.prepare_images(batch)
        state = self.base_policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        bsize = state.shape[0]
        device = state.device
        dtype = self.model.action_in_proj.weight.dtype
        shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        x = self.model.sample_noise(shape, device).to(dtype=dtype)
        steps = int(self.config.num_steps)
        if steps <= 0:
            raise ValueError(f"SmolVLA num_steps must be positive, got {steps}")

        next_memory = memory
        memory_slice = slice(-self.spec.num_tokens, None) if self.spec.variant == "prefix" else slice(0, self.spec.num_tokens)
        for index in range(steps):
            t_value = 1.0 - index / steps
            timestep = torch.full((bsize,), t_value, device=device, dtype=torch.float32)
            if self.spec.variant == "prefix":
                prefix, prefix_pad, prefix_att = self._prefix_with_memory(
                    images, img_masks, lang_tokens, lang_masks, state, memory
                )
                suffix, suffix_pad, suffix_att = self.model.embed_suffix(x, timestep)
            else:
                prefix, prefix_pad, prefix_att = self.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks, state=state
                )
                suffix, suffix_pad, suffix_att = self._suffix_with_memory(x, timestep, memory)
            pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att, suffix_att], dim=1)
            attention_mask = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            outputs, _ = self.model.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
            )
            prefix_out, suffix_out = outputs
            next_memory = (prefix_out if self.spec.variant == "prefix" else suffix_out)[:, memory_slice, :]
            action_out = suffix_out[:, -self.config.chunk_size :, :]
            action_out = action_out.to(dtype=self.model.action_out_proj.weight.dtype)
            velocity = self.model.action_out_proj(action_out).to(dtype=x.dtype)
            x = x - velocity / steps
        action_dim = int(self.config.action_feature.shape[0])
        return x[:, :, :action_dim].float(), next_memory.detach()

