"""Minimal reproducible SmolVLA baseline trainer for the MIKASA manifest."""
from __future__ import annotations

import random
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from smolvla_data.mikasa_dataset import MikasaWindowDataset


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


def flatten_config(cfg: DictConfig) -> dict[str, str | int | float | bool]:
    result = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        result[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config_baseline")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = torch.device(cfg.trainer.device if cfg.trainer.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.data.manifest).expanduser()
    dataset = MikasaWindowDataset(manifest, "train")
    loader = DataLoader(dataset, batch_size=int(cfg.trainer.batch_size), shuffle=True, num_workers=0)
    stats = dict(dataset.dataset.meta.stats)
    mapped_stats = dict(stats)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        if source in mapped_stats:
            mapped_stats[target] = mapped_stats[source]

    policy = SmolVLAPolicy.from_pretrained(cfg.pretrained_policy, device=str(device))
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
    with mlflow.start_run(run_name=cfg.experiment_name) if cfg.tracking.enabled else _nullcontext() as run:
        if cfg.tracking.enabled:
            mlflow.log_params(flatten_config(cfg))
            mlflow.log_param("device", str(device))
            mlflow.log_param("trainable_parameter_count", sum(parameter.numel() for parameter in parameters))
        for step in range(1, int(cfg.trainer.steps) + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            batch = preprocess_batch(raw_batch, preprocessor, stats)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy(batch)
            loss.backward()
            optimizer.step()
            if step % int(cfg.trainer.log_every) == 0:
                print(f"step={step} loss={float(loss.detach()):.6f}")
                if cfg.tracking.enabled:
                    mlflow.log_metric("train/loss", float(loss.detach()), step=step)
            if step % int(cfg.trainer.checkpoint_every) == 0 or step == int(cfg.trainer.steps):
                checkpoint = output_dir / f"step_{step:06d}.pt"
                torch.save({"step": step, "model": policy.state_dict(), "optimizer": optimizer.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, checkpoint)
                if cfg.tracking.enabled:
                    mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
