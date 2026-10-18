"""Minimal reproducible SmolVLA baseline trainer for the MIKASA manifest."""
from __future__ import annotations

import random
import time
import math
from pathlib import Path
from contextlib import contextmanager

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from smolvla_data.mikasa_dataset import MikasaWindowDataset
from smolvla_data.evaluation_sampling import select_stratified_window_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_mikasa_cameras(sample: dict, stats: dict) -> tuple[dict, dict]:
    sample = dict(sample)
    stats = dict(stats)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        sample[target] = sample.pop(source)
        if source in stats:
            stats[target] = stats[source]
    return sample, stats


def remove_processor_batch_dim(sample: dict) -> dict:
    result = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            result[key] = value.squeeze(0)
        else:
            result[key] = value
    return result


def preprocess_batch(raw_batch: dict, preprocessor, stats: dict) -> dict:
    batch_size = raw_batch["action"].shape[0]
    samples = []
    for index in range(batch_size):
        sample = {}
        for key, value in raw_batch.items():
            sample[key] = value[index] if torch.is_tensor(value) else value[index]
        sample, _ = map_mikasa_cameras(sample, stats)
        processed = preprocessor(sample)
        samples.append(remove_processor_batch_dim(processed))
    return default_collate(samples)


@contextmanager
def deterministic_validation(seed: int):
    """Keep validation noise/timesteps repeatable without changing train RNG."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate(policy, loader, preprocessor, stats, batches: int, seed: int) -> float:
    policy.eval()
    losses = []
    iterator = iter(loader)
    with deterministic_validation(seed), torch.no_grad():
        for _ in range(batches):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                break
            batch = preprocess_batch(raw_batch, preprocessor, stats)
            loss, _ = policy(batch)
            losses.append(float(loss))
    policy.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches.")
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, policy, optimizer, step: int, cfg: DictConfig, include_optimizer: bool) -> None:
    payload = {
        "step": step,
        "model": policy.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def flatten_config(cfg: DictConfig) -> dict[str, str | int | float | bool]:
    result = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        result[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return result


def unfreeze_last_vlm_layers(policy: SmolVLAPolicy, count: int) -> None:
    """Unfreeze only the final text-transformer layers for a capacity-matched control."""
    if count <= 0:
        return
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    for parameter in policy.model.vlm_with_expert.vlm.parameters():
        parameter.requires_grad = False
    for layer in text_model.layers[-min(count, len(text_model.layers)):]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


@hydra.main(version_base=None, config_path="../configs", config_name="config_baseline")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = torch.device(cfg.trainer.device if cfg.trainer.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.data.manifest).expanduser()
    dataset = MikasaWindowDataset(manifest, "train")
    validation_base_dataset = MikasaWindowDataset(manifest, "validation")
    loader = DataLoader(dataset, batch_size=int(cfg.trainer.batch_size), shuffle=True, num_workers=0)
    validation_indices = select_stratified_window_indices(
        validation_base_dataset.payload,
        "validation",
        int(cfg.trainer.validation_windows),
        full_only=bool(cfg.trainer.validation_full_only),
    )
    validation_dataset = torch.utils.data.Subset(validation_base_dataset, validation_indices)
    validation_loader = DataLoader(validation_dataset, batch_size=int(cfg.trainer.batch_size), shuffle=False, num_workers=0)
    stats = dict(dataset.dataset.meta.stats)
    mapped_stats = dict(stats)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        if source in mapped_stats:
            mapped_stats[target] = mapped_stats[source]

    policy = SmolVLAPolicy.from_pretrained(cfg.pretrained_policy, device=str(device))
    unfreeze_last_vlm_layers(policy, int(cfg.trainer.unfreeze_vlm_layers))
    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=mapped_stats)
    policy.train()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.trainer.learning_rate), weight_decay=float(cfg.trainer.weight_decay))

    if cfg.tracking.enabled:
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)

    output_dir = Path(cfg.trainer.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    accumulation_steps = int(cfg.trainer.gradient_accumulation_steps)
    checkpoint_every = int(cfg.trainer.checkpoint_every)
    validation_every = int(cfg.trainer.validation_every)
    validation_batches = math.ceil(
        int(cfg.trainer.validation_windows) / int(cfg.trainer.batch_size)
    )
    best_validation_loss = float("inf")
    total_steps = int(cfg.trainer.steps)
    samples_seen = 0
    step_times: list[float] = []
    with mlflow.start_run(run_name=cfg.experiment_name) if cfg.tracking.enabled else _nullcontext() as run:
        if cfg.tracking.enabled:
            mlflow.log_params(flatten_config(cfg))
            mlflow.log_param("device", str(device))
            mlflow.log_param("trainable_parameter_count", sum(parameter.numel() for parameter in parameters))
            mlflow.log_param("unfreeze_vlm_layers", int(cfg.trainer.unfreeze_vlm_layers))
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, total_steps + 1):
            step_started = time.perf_counter()
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                try:
                    raw_batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    raw_batch = next(iterator)
                batch = preprocess_batch(raw_batch, preprocessor, stats)
                loss, _ = policy(batch)
                (loss / accumulation_steps).backward()
                accumulated_loss += float(loss.detach())
                samples_seen += int(raw_batch["action"].shape[0])
            loss = torch.tensor(accumulated_loss / accumulation_steps)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_seconds = time.perf_counter() - step_started
            step_times.append(step_seconds)
            mean_step_seconds = sum(step_times[-20:]) / min(len(step_times), 20)
            eta_seconds = (total_steps - step) * mean_step_seconds
            fractional_epoch = samples_seen / len(dataset)
            if step % int(cfg.trainer.log_every) == 0:
                train_loss = float(loss)
                print(
                    f"step={step} train_loss={train_loss:.6f} "
                    f"step_time={step_seconds:.1f}s avg_time={mean_step_seconds:.1f}s "
                    f"epoch={fractional_epoch:.3f} eta={eta_seconds / 60:.1f}min"
                )
                if cfg.tracking.enabled:
                    mlflow.log_metric("train/loss", train_loss, step=step)
            if step % validation_every == 0:
                validation_loss = evaluate(
                    policy,
                    validation_loader,
                    preprocessor,
                    stats,
                    validation_batches,
                    int(cfg.seed) + 100_000,
                )
                print(f"step={step} validation_loss={validation_loss:.6f}")
                if cfg.tracking.enabled:
                    mlflow.log_metric("validation/loss", validation_loss, step=step)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    if cfg.trainer.save_checkpoints:
                        save_checkpoint(output_dir / "best.pt", policy, optimizer, step, cfg, include_optimizer=False)
                        if cfg.tracking.enabled:
                            mlflow.log_artifact(str(output_dir / "best.pt"), artifact_path="checkpoints")
            if step % checkpoint_every == 0 and cfg.trainer.save_checkpoints:
                save_checkpoint(output_dir / "last.pt", policy, optimizer, step, cfg, include_optimizer=True)

        if cfg.trainer.save_checkpoints:
            save_checkpoint(output_dir / "final.pt", policy, optimizer, total_steps, cfg, include_optimizer=False)
            save_checkpoint(output_dir / "last.pt", policy, optimizer, total_steps, cfg, include_optimizer=True)
        if cfg.tracking.enabled and cfg.trainer.save_checkpoints:
            mlflow.log_artifact(str(output_dir / "final.pt"), artifact_path="checkpoints")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
