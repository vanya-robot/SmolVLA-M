"""Evaluate pretrained or fine-tuned SmolVLA on a fixed MIKASA split."""
from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

from evaluation.policies import load_policy, model_metadata
from smolvla_data.mikasa_dataset import MikasaWindowDataset
from smolvla_data.evaluation_sampling import select_stratified_window_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_sample(sample: dict, stats: dict) -> dict:
    sample = dict(sample)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        sample[target] = sample.pop(source)
    return sample


def remove_singleton_batch(sample: dict) -> dict:
    return {
        key: value.squeeze(0)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1
        else value
        for key, value in sample.items()
    }


def preprocess_batch(raw_batch, preprocessor, stats):
    samples = []
    for index in range(raw_batch["action"].shape[0]):
        sample = {
            key: value[index] if torch.is_tensor(value) else value[index]
            for key, value in raw_batch.items()
        }
        processed = preprocessor(remap_sample(sample, stats))
        samples.append(remove_singleton_batch(processed))
    return default_collate(samples)


@contextmanager
def deterministic_rng(seed: int):
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


def flatten_config(cfg: DictConfig) -> dict[str, str | int | float | bool]:
    result = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        result[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config_evaluate")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    requested = str(cfg.trainer.device) if "trainer" in cfg and "device" in cfg.trainer else "cuda"
    device = torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")
    base_dataset = MikasaWindowDataset(cfg.data.manifest, cfg.evaluation.split)
    selected_indices = select_stratified_window_indices(
        base_dataset.payload,
        str(cfg.evaluation.split),
        int(cfg.evaluation.max_windows),
        full_only=True,
    )
    dataset = torch.utils.data.Subset(base_dataset, selected_indices)
    loader = DataLoader(dataset, batch_size=int(cfg.evaluation.batch_size), shuffle=False, num_workers=0)
    stats = dict(dataset.dataset.meta.stats)
    mapped_stats = dict(stats)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        if source in mapped_stats:
            mapped_stats[target] = mapped_stats[source]

    policy = load_policy(cfg.model, device)
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=mapped_stats)
    losses = []
    valid_fractions = []
    iterator = iter(loader)
    with deterministic_rng(int(cfg.seed)), torch.no_grad():
        for _ in range((len(dataset) + int(cfg.evaluation.batch_size) - 1) // int(cfg.evaluation.batch_size)):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                break
            batch = preprocess_batch(raw_batch, preprocessor, mapped_stats)
            loss, _ = policy(batch)
            losses.append(float(loss))
            valid_fractions.append(float((~batch["action_is_pad"]).float().mean()))

    if not losses:
        raise RuntimeError("Evaluation loader produced no batches.")
    result = {
        "seed": int(cfg.seed),
        "split": str(cfg.evaluation.split),
        "windows": len(dataset),
        "batches": len(losses),
        "batch_size": int(cfg.evaluation.batch_size),
        "loss_mean": sum(losses) / len(losses),
        "loss_std": float(torch.tensor(losses).std(unbiased=False)),
        "valid_action_fraction": sum(valid_fractions) / len(valid_fractions),
        "model": model_metadata(cfg.model, policy),
    }
    output = Path(cfg.evaluation.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if cfg.tracking.enabled:
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)
        with mlflow.start_run(run_name=cfg.experiment_name):
            mlflow.log_params(flatten_config(cfg))
            mlflow.log_metrics({"eval/loss_mean": result["loss_mean"], "eval/loss_std": result["loss_std"], "eval/valid_action_fraction": result["valid_action_fraction"]})
            mlflow.log_artifact(str(output), artifact_path="evaluations")


if __name__ == "__main__":
    main()
