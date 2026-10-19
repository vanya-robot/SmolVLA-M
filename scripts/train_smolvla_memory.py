"""Train recurrent memory-token variants on contiguous MIKASA sequences."""
from __future__ import annotations

import random
import time
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, default_collate

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from memory.smolvla_memory import MemorySmolVLAPolicy, MemorySpec
from scripts.train_smolvla_baseline import preprocess_batch, map_mikasa_cameras
from smolvla_data.mikasa_sequences import MikasaSequenceDataset
from smolvla_data.policy_factory import load_smolvla


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def collate_sequence(batch):
    if len(batch) != 1:
        raise ValueError("Memory training currently requires batch_size=1.")
    return batch[0]


def prepare_sequence(raw_sequence, preprocessor, stats, device):
    """Preprocess one observation at a time instead of materializing a full episode on GPU."""
    for raw_sample in raw_sequence:
        raw_batch = default_collate([raw_sample])
        processed = preprocess_batch(raw_batch, preprocessor, stats)
        yield move_batch(processed, device)


def save_checkpoint(path, policy, optimizer, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": policy.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)


@hydra.main(version_base=None, config_path="../configs", config_name="config_memory")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.trainer.device) if str(cfg.trainer.device) == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.data.manifest).expanduser()
    train_data = MikasaSequenceDataset(manifest, "train", int(cfg.memory.sequence_length))
    val_data = MikasaSequenceDataset(manifest, "validation", int(cfg.memory.sequence_length))
    probe_selection = str(cfg.memory.probe_episode_selection)
    probe_target = cfg.memory.probe_episode_length
    if probe_selection == "random":
        train_source = train_data
        shuffle = True
    else:
        indices = train_data.probe_indices(
            selection=probe_selection,
            target_length=None if probe_target is None else int(probe_target),
        )
        train_source = Subset(train_data, indices)
        shuffle = False
        chosen_index = indices[0]
        chosen_episode, _ = train_data.starts[chosen_index]
        print(
            f"probe episode: index={chosen_episode} "
            f"length={train_data.base.episode_lengths[chosen_episode]} "
            f"selection={probe_selection} target={probe_target}"
        )
    train_loader = DataLoader(train_source, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=collate_sequence)
    val_indices = val_data.representative_indices(int(cfg.trainer.validation_sequences))
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_sequence)

    action_dim = int(train_data.dataset.meta.features["action"]["shape"][0])
    base = load_smolvla(cfg.pretrained_policy, str(device), action_dim)
    init_checkpoint = str(cfg.trainer.initialization_checkpoint)
    if init_checkpoint and Path(init_checkpoint).expanduser().exists():
        payload = torch.load(Path(init_checkpoint).expanduser(), map_location="cpu")
        base.load_state_dict(payload["model"], strict=True)
    stats = dict(train_data.dataset.meta.stats)
    mapped_stats = dict(stats)
    for source, target in {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}.items():
        if source in mapped_stats:
            mapped_stats[target] = mapped_stats[source]
    preprocessor, _ = make_smolvla_pre_post_processors(base.config, dataset_stats=mapped_stats)
    unfreeze_vlm_layers = int(cfg.memory.unfreeze_vlm_layers)
    policy = MemorySmolVLAPolicy(
        base,
        MemorySpec(str(cfg.memory.variant), int(cfg.memory.num_tokens)),
        unfreeze_vlm_layers=unfreeze_vlm_layers,
    ).to(device)
    policy.train()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.trainer.learning_rate), weight_decay=float(cfg.trainer.weight_decay))
    output_dir = Path(cfg.trainer.output_dir).expanduser()
    if cfg.tracking.enabled:
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)

    train_iter = iter(train_loader) if int(cfg.memory.sequence_length) > 0 else iter(range(len(train_source)))
    def next_sequence(iterator):
        item = next(iterator)
        if int(cfg.memory.sequence_length) > 0:
            return item
        source_index = item if not isinstance(train_source, Subset) else train_source.indices[item]
        return train_data.iter_sequence(source_index)
    best = float("inf")
    step_times = []
    with mlflow.start_run(run_name=cfg.experiment_name) if cfg.tracking.enabled else _NullContext():
        if cfg.tracking.enabled:
            mlflow.log_params({"memory_variant": str(cfg.memory.variant), "memory_tokens": int(cfg.memory.num_tokens), "sequence_length": int(cfg.memory.sequence_length), "unfreeze_vlm_layers": unfreeze_vlm_layers, "trainable_parameters": sum(p.numel() for p in parameters)})
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, int(cfg.trainer.steps) + 1):
            started = time.perf_counter()
            total = 0.0
            for _ in range(int(cfg.trainer.gradient_accumulation_steps)):
                try:
                    raw_sequence = next_sequence(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader) if int(cfg.memory.sequence_length) > 0 else iter(range(len(train_source)))
                    raw_sequence = next_sequence(train_iter)
                sequence = prepare_sequence(raw_sequence, preprocessor, stats, device)
                loss = policy.forward_sequence(sequence)
                (loss / int(cfg.trainer.gradient_accumulation_steps)).backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            avg = sum(step_times[-20:]) / min(20, len(step_times))
            print(f"step={step} train_loss={total / int(cfg.trainer.gradient_accumulation_steps):.6f} step_time={elapsed:.1f}s avg_time={avg:.1f}s eta={(int(cfg.trainer.steps)-step)*avg/60:.1f}min")
            if cfg.tracking.enabled:
                mlflow.log_metric("train/loss", total / int(cfg.trainer.gradient_accumulation_steps), step=step)
            if step % int(cfg.trainer.validation_every) == 0:
                policy.eval()
                losses = []
                with torch.no_grad():
                    if int(cfg.memory.sequence_length) > 0:
                        validation_iterator = iter(val_loader)
                    else:
                        validation_iterator = (val_data.iter_sequence(index) for index in val_indices)
                    for raw_sequence in validation_iterator:
                        losses.append(float(policy.forward_sequence(prepare_sequence(raw_sequence, preprocessor, stats, device))))
                policy.train()
                val_loss = sum(losses) / len(losses)
                print(f"step={step} validation_loss={val_loss:.6f}")
                if cfg.tracking.enabled:
                    mlflow.log_metric("validation/loss", val_loss, step=step)
                if val_loss < best and cfg.trainer.save_checkpoints:
                    best = val_loss
                    save_checkpoint(output_dir / "best.pt", policy, optimizer, step, cfg)
            if step % int(cfg.trainer.checkpoint_every) == 0 and cfg.trainer.save_checkpoints:
                save_checkpoint(output_dir / "last.pt", policy, optimizer, step, cfg)
        if cfg.trainer.save_checkpoints:
            save_checkpoint(output_dir / "final.pt", policy, optimizer, int(cfg.trainer.steps), cfg)
            save_checkpoint(output_dir / "last.pt", policy, optimizer, int(cfg.trainer.steps), cfg)


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


if __name__ == "__main__":
    main()

