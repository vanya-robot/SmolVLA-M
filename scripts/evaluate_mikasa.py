"""Run a SmolVLA checkpoint in the MIKASA-Robo-VLA simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=("baseline", "prefix", "expert"), default="baseline")
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--task", default="RememberColor3-VLA-v0")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stats", required=True, help="LeRobot stats.json used during training")
    parser.add_argument("--output-dir", default="outputs/mikasa_eval")
    parser.add_argument("--save-videos", action="store_true")
    return parser.parse_args()


def make_env(task: str, save_videos: bool, output_dir: Path):
    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    env = gym.make(
        task,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        render_mode="all" if save_videos else None,
        sim_backend="gpu",
    )
    env = apply_mikasa_vla_wrappers(env, include_overlays=save_videos)
    if save_videos:
        from mani_skill.utils.wrappers import RecordEpisode

        env = RecordEpisode(
            env,
            output_dir=str(output_dir / "videos" / task),
            save_trajectory=False,
            max_steps_per_video=env.max_episode_steps,
        )
    return env


def load_policy(checkpoint: Path, stats_path: Path, model_type: str, memory_tokens: int, device: torch.device):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from smolvla_data.policy_factory import load_smolvla

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    action_dim = int(len(stats["action"]["mean"]))
    policy = load_smolvla("lerobot/smolvla_base", str(device), action_dim)
    payload = torch.load(checkpoint, map_location="cpu")
    if model_type == "baseline":
        policy.load_state_dict(payload["model"], strict=True)
    else:
        from memory.smolvla_memory import MemorySmolVLAPolicy, MemorySpec

        checkpoint_memory = payload.get("config", {}).get("memory", {})
        variant = str(checkpoint_memory.get("variant", model_type))
        if variant != model_type:
            raise ValueError(f"Checkpoint memory variant is {variant!r}, but --model-type={model_type!r}")
        tokens = int(checkpoint_memory.get("num_tokens", memory_tokens))
        policy = MemorySmolVLAPolicy(
            policy,
            MemorySpec(variant, tokens),
            unfreeze_vlm_layers=int(checkpoint_memory.get("unfreeze_vlm_layers", 0)),
        ).to(device)
        policy.load_state_dict(payload["model"], strict=True)
    for source, target in {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }.items():
        if source in stats:
            stats[target] = stats[source]
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats=stats,
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def observation_to_batch(obs: dict[str, Any], info: dict[str, Any], device: torch.device) -> dict[str, Any]:
    rgb = torch.as_tensor(obs["rgb"], device=device)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[-1] != 6:
        raise ValueError(f"Expected two RGB cameras concatenated in last dimension, got {tuple(rgb.shape)}")
    # MIKASA wrapper: (B,H,W,6), LeRobot: (B,3,H,W), float in [0, 1].
    top = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
    wrist = rgb[..., 3:].permute(0, 3, 1, 2).float() / 255.0
    state = torch.as_tensor(obs["proprio"], device=device).float()
    if state.ndim == 1:
        state = state.unsqueeze(0)
    instruction = info.get("language_instruction", "")
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0]
    return {
        "observation.images.camera1": top,
        "observation.images.camera2": wrist,
        "observation.state": state,
        "task": [str(instruction)],
    }


def scalar_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, preprocessor, postprocessor = load_policy(
        Path(args.checkpoint).expanduser(), Path(args.stats).expanduser(), args.model_type, args.memory_tokens, device
    )
    env = make_env(args.task, args.save_videos, output_dir)
    episodes = []
    try:
        for episode in range(args.num_episodes):
            seed = args.start_seed + episode
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            policy.reset()
            memory = policy.initial_memory(1, device) if args.model_type != "baseline" else None
            action_queue = []
            obs, info = env.reset(seed=seed)
            success = False
            total_return = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                raw_batch = observation_to_batch(obs, info, device)
                batch = preprocessor(raw_batch)
                with torch.inference_mode():
                    if args.model_type == "baseline":
                        action = policy.select_action(batch)
                    else:
                        if not action_queue:
                            action_chunk, memory = policy.predict_action_chunk(batch, memory)
                            action_queue = [item.unsqueeze(0) for item in action_chunk[0]]
                        action = action_queue.pop(0)
                # LeRobot 0.6 exposes policy postprocessing through the
                # tensor-oriented process_action API.
                action = postprocessor.process_action(action)
                action = action.squeeze(0).detach().float()
                expected_action_dim = int(env.action_space.shape[-1])
                if action.shape[-1] != expected_action_dim:
                    raise RuntimeError(
                        f"Action contract mismatch: policy produced {action.shape[-1]} dimensions, "
                        f"but MIKASA expects {expected_action_dim}."
                    )
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += float(torch.as_tensor(reward).detach().cpu().reshape(-1)[0])
                success = success or scalar_bool(info.get("success", False))
                steps += 1
            episodes.append({
                "episode": episode,
                "seed": seed,
                "success": success,
                "return": total_return,
                "steps": steps,
            })
            print(f"episode={episode} seed={seed} success={success} steps={steps} return={total_return:.4f}")
    finally:
        env.close()
    result = {
        "task": args.task,
        "num_episodes": len(episodes),
        "start_seed": args.start_seed,
        "success_rate": sum(item["success"] for item in episodes) / max(1, len(episodes)),
        "mean_return": sum(item["return"] for item in episodes) / max(1, len(episodes)),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "episodes": episodes,
    }
    output = output_dir / f"{args.task}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
