#!/usr/bin/env python3
"""Run a trained NEUBAY Go2 policy in MuJoCo and save an MP4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# These must be selected before importing JAX/MuJoCo.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND = ROOT / "external" / "mujoco_playground"
if str(PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(PLAYGROUND))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from memory.lru import StackedLRU
from mujoco_playground import registry
from online_rl.ac_modules import Alpha, RecurrentActor, RecurrentCritic


def load_policy(path: Path):
    """Rebuild the checkpoint PyTree and load its serialized leaves."""
    with path.open("rb") as checkpoint:
        config = json.loads(checkpoint.readline())
        key = random.PRNGKey(config["seed"])
        key, memory1_key, memory2_key, critic_key, actor_key = random.split(key, 5)
        model_config = config["model"]

        critic_memory = StackedLRU(**model_config["memory"], key=memory1_key)
        target_memory = StackedLRU(**model_config["memory"], key=memory1_key)
        actor_memory = StackedLRU(**model_config["memory"], key=memory2_key)
        observation_shape = (61,)  # 48-D state + previous 12-D action + reward.
        action_shape = (12,)

        critic = RecurrentCritic(
            observation_shape, action_shape, critic_memory, model_config, critic_key
        )
        target_critic = eqx.nn.inference_mode(
            RecurrentCritic(
                observation_shape,
                action_shape,
                target_memory,
                model_config,
                critic_key,
            )
        )
        actor = RecurrentActor(
            observation_shape, action_shape, actor_memory, model_config, actor_key
        )
        ratio = model_config["target_alpha_ratio"]
        target_entropy = None if ratio is None else -12.0 * ratio
        alpha = Alpha(model_config["init_alpha"], target_entropy)
        template = {
            "q_network": critic,
            "q_target": target_critic,
            "pi_network": actor,
            "alpha": alpha,
        }
        loaded = eqx.tree_deserialise_leaves(checkpoint, template)
    return eqx.nn.inference_mode(loaded["pi_network"]), config


def encode_mp4(frames, output: Path, fps: float, width: int, height: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("go2_policy.mp4"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="Simulator seed")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", default="track")
    args = parser.parse_args()

    actor, checkpoint_config = load_policy(args.checkpoint)
    env_config = registry.get_default_config("Go2JoystickFlatTerrain")
    env_config.impl = "jax"
    env_config.noise_config.level = 0.0
    env = registry.load("Go2JoystickFlatTerrain", config=env_config)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    command = jnp.array([args.vx, args.vy, args.yaw_rate], dtype=jnp.float32)
    state = reset(random.PRNGKey(args.seed))
    state.info["command"] = command
    state = state.replace(
        obs={**state.obs, "state": state.obs["state"].at[-3:].set(command)}
    )
    recurrent_state = actor.initial_state(1)
    previous_action = jnp.zeros(12, dtype=jnp.float32)
    previous_reward = jnp.array(0.0, dtype=jnp.float32)
    policy_key = random.PRNGKey(args.seed + 10_000)
    trajectory = [state]
    total_reward = 0.0

    for index in range(args.steps):
        policy_key, action_key = random.split(policy_key)
        augmented_observation = jnp.concatenate(
            [state.obs["state"], previous_action, previous_reward[None]]
        )
        action, recurrent_state, _ = actor(
            augmented_observation[None, None, :],
            recurrent_state,
            jnp.array([[index == 0]]),
            action_key,
            deterministic=True,
        )
        previous_action = action[0]
        state.info["command"] = command
        state = step(state, previous_action)
        state.info["command"] = command
        state = state.replace(
            obs={**state.obs, "state": state.obs["state"].at[-3:].set(command)}
        )
        previous_reward = state.reward
        total_reward += float(state.reward)
        trajectory.append(state)
        if bool(state.done):
            print(f"Episode terminated at step {index + 1}.")
            break

    sampled = trajectory[:: args.render_every]
    frames = env.render(
        sampled,
        height=args.height,
        width=args.width,
        camera=args.camera,
    )
    fps = 1.0 / (env.dt * args.render_every)
    encode_mp4(frames, args.output, fps, args.width, args.height)
    duration = len(frames) / fps
    print(f"Checkpoint seed: {checkpoint_config['seed']}")
    print(f"Steps: {len(trajectory) - 1}; total reward: {total_reward:.3f}")
    print(f"Video: {args.output.resolve()} ({duration:.2f}s at {fps:.2f} fps)")


if __name__ == "__main__":
    main()
