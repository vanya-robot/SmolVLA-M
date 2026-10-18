"""Sequential flow-loss evaluation for recurrent SmolVLA memory checkpoints."""
from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from memory.smolvla_memory import MemorySmolVLAPolicy, MemorySpec
from scripts.train_smolvla_baseline import preprocess_batch
from smolvla_data.mikasa_sequences import MikasaSequenceDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_sequence(batch):
    if len(batch) != 1:
        raise ValueError("Memory evaluation requires batch_size=1.")
    return batch[0]


def prepare_sequence(raw_sequence, preprocessor, stats, device):
    result = []
    for raw_sample in raw_sequence:
        processed = preprocess_batch(default_collate([raw_sample]), preprocessor, stats)
        result.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in processed.items()})
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config_memory_evaluate")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if str(cfg.evaluation.get("device", "cuda")) == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = MikasaSequenceDataset(Path(cfg.data.manifest).expanduser(), str(cfg.evaluation.split), int(cfg.memory.sequence_length))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_sequence)
    checkpoint_path = Path(cfg.evaluation.checkpoint_path).expanduser()
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_memory = payload.get("config", {}).get("memory", {})
    checkpoint_variant = str(checkpoint_memory.get("variant", ""))
    checkpoint_tokens = int(checkpoint_memory.get("num_tokens", cfg.memory.num_tokens))
    requested_variant = str(cfg.memory.variant)
    if requested_variant != "auto" and requested_variant != checkpoint_variant:
        raise ValueError(
            f"Config requests memory.variant={requested_variant!r}, but checkpoint was trained with "
            f"memory.variant={checkpoint_variant!r}. Use memory.variant=auto or the matching value."
        )
    variant = checkpoint_variant if requested_variant == "auto" else requested_variant
    num_tokens = checkpoint_tokens if requested_variant == "auto" else int(cfg.memory.num_tokens)
    base = SmolVLAPolicy.from_pretrained(cfg.pretrained_policy, device=str(device))
    stats = dict(dataset.dataset.meta.stats)
    mapped_stats = dict(stats)
    for source, target in {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}.items():
        if source in mapped_stats:
            mapped_stats[target] = mapped_stats[source]
    preprocessor, _ = make_smolvla_pre_post_processors(base.config, dataset_stats=mapped_stats)
    policy = MemorySmolVLAPolicy(base, MemorySpec(variant, num_tokens)).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    losses = []
    with torch.no_grad():
        for index, raw_sequence in enumerate(loader):
            if index >= int(cfg.evaluation.max_sequences):
                break
            losses.append(float(policy.forward_sequence(prepare_sequence(raw_sequence, preprocessor, stats, device))))
    result = {
        "seed": int(cfg.seed),
        "split": str(cfg.evaluation.split),
        "sequences": len(losses),
        "sequence_length": int(cfg.memory.sequence_length),
        "loss_mean": sum(losses) / len(losses),
        "loss_std": float(torch.tensor(losses).std(unbiased=False)),
        "memory": {"variant": variant, "num_tokens": num_tokens},
        "checkpoint_path": str(cfg.evaluation.checkpoint_path),
    }
    output = Path(cfg.evaluation.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if cfg.tracking.enabled:
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)
        with mlflow.start_run(run_name=cfg.experiment_name):
            mlflow.log_params({"memory_variant": str(cfg.memory.variant), "memory_tokens": int(cfg.memory.num_tokens), "sequence_length": int(cfg.memory.sequence_length)})
            mlflow.log_metrics({"eval/loss_mean": result["loss_mean"], "eval/loss_std": result["loss_std"]})
            mlflow.log_artifact(str(output), artifact_path="evaluations")


if __name__ == "__main__":
    main()

