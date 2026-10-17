from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from memory import CompressedMemory


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def flatten_config(config: DictConfig) -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in OmegaConf.to_container(config, resolve=True).items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.trainer.mode != "infrastructure_smoke":
        raise NotImplementedError(
            "Only infrastructure_smoke is implemented. "
            "The real LeRobot/SmolVLA trainer will be added after dataset inspection."
        )

    set_seed(cfg.seed)

    model = CompressedMemory(
        input_dim=cfg.data.input_dim,
        memory_dim=cfg.data.memory_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.learning_rate)

    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment)

    run_dir = Path.cwd()
    with mlflow.start_run(run_name=cfg.experiment_name):
        mlflow.log_params(flatten_config(cfg))

        for step in range(cfg.trainer.steps):
            history = torch.randn(
                cfg.data.batch_size,
                cfg.data.history_length,
                cfg.data.input_dim,
            )
            target = torch.zeros(cfg.data.batch_size, cfg.data.memory_dim)

            prediction = model(history)
            loss = (prediction - target).square().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("train/loss", loss.item(), step=step)
            print(f"step={step + 1} loss={loss.item():.6f}")

        checkpoint = run_dir / "infrastructure_smoke.pt"
        torch.save(model.state_dict(), checkpoint)
        mlflow.log_artifact(str(checkpoint))


if __name__ == "__main__":
    main()
