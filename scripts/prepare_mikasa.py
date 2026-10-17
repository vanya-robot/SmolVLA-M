"""Build a deterministic MIKASA episode split and action-window manifest."""
from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from smolvla_data.mikasa_windows import make_episode_split, make_windows, write_manifest


def flatten_config(cfg: DictConfig) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        result[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config_mikasa")
def main(cfg: DictConfig) -> None:
    from lerobot.datasets import LeRobotDatasetMetadata

    root = Path(cfg.data.root).expanduser()
    metadata = LeRobotDatasetMetadata(cfg.data.repo_id, root=root)
    lengths = [int(value) for value in metadata.episodes["length"]]
    split = make_episode_split(len(lengths), int(cfg.seed), float(cfg.split.train_fraction), float(cfg.split.validation_fraction))
    episode_groups = {"train": split.train, "validation": split.validation, "test": split.test}
    windows = {name: make_windows(lengths, ids, int(cfg.data.horizon), bool(cfg.data.pad_to_horizon)) for name, ids in episode_groups.items()}
    manifest = Path(cfg.output.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    write_manifest(
        manifest,
        repo_id=cfg.data.repo_id,
        root=str(root),
        seed=int(cfg.seed),
        horizon=int(cfg.data.horizon),
        pad_to_horizon=bool(cfg.data.pad_to_horizon),
        episode_lengths=lengths,
        split=split,
        windows=windows,
        extra={"fps": int(metadata.info.fps), "total_episodes": int(metadata.info.total_episodes), "total_frames": int(metadata.info.total_frames)},
    )
    if cfg.tracking.enabled:
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)
        with mlflow.start_run(run_name=cfg.experiment_name):
            mlflow.log_params(flatten_config(cfg))
            mlflow.log_param("manifest", str(manifest))
            mlflow.log_metrics({f"windows/{name}": len(items) for name, items in windows.items()})
            mlflow.log_artifact(str(manifest), artifact_path="data")
    print(f"manifest: {manifest}")
    print(f"episodes: train={len(split.train)} val={len(split.validation)} test={len(split.test)}")
    print("windows:", {name: len(items) for name, items in windows.items()})


if __name__ == "__main__":
    main()
