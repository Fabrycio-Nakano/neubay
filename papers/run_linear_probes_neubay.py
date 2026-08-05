#!/usr/bin/env python3
"""
Run linear/MLP probes on NEUBAY continuous world-model hidden states.

Example:
    python papers/run_linear_probes_neubay.py \
        --domain d4rl_loco \
        --dataset-name hopper-medium-v2 \
        --model-seed 0 \
        --sample-size 10000 \
        --aggregate member \
        --member-index 0

This script is intended to run inside the NEUBAY environment/container, where
JAX, Equinox, D4RL/NeoRL, MuJoCo, scikit-learn, pandas, etc. are installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "offline_world").is_dir() and (parent / "experience").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing offline_world/ and experience/.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing NEUBAY runtime dependency. Run this script inside the NEUBAY "
        "Conda environment or Apptainer/Singularity container. "
        f"Missing module: {exc.name}"
    ) from exc

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experience.wrapper import make_env
from experience.world_buffer import get_dataset
from offline_world.modules import EnsembleContModel, Scaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear/MLP probes on NEUBAY world-model hidden states."
    )
    parser.add_argument("--domain", default="d4rl_loco", help="Domain/config family.")
    parser.add_argument("--dataset-name", default="hopper-medium-v2", help="Dataset/env name.")
    parser.add_argument(
        "--hdf5-dataset",
        default=None,
        help="Optional HDF5 dataset path. If set, bypasses make_env/D4RL loading.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory containing ensemble_seed*.eqx or latest_seed*.eqx.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional exact .eqx checkpoint path. Takes precedence over --checkpoint-dir.",
    )
    parser.add_argument("--model-seed", type=int, default=0, help="Checkpoint seed index.")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Number of transitions to sample. Use -1 for all.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling/splits.")
    parser.add_argument(
        "--aggregate",
        choices=["member", "mean", "stack"],
        default="member",
        help="How to convert ensemble activations into probe features.",
    )
    parser.add_argument("--member-index", type=int, default=0, help="Ensemble member for --aggregate member.")
    parser.add_argument("--batch-size", type=int, default=4096, help="JAX inference batch size.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge L2 strength.")
    parser.add_argument("--skip-mlp", action="store_true", help="Only run Ridge probes.")
    parser.add_argument(
        "--probe-seeds",
        default="0,1,2",
        help="MLP initialization seeds separated by comma or colon. Metrics are mean/std.",
    )
    parser.add_argument("--mlp-max-epochs", type=int, default=500)
    parser.add_argument("--mlp-patience", type=int, default=30)
    parser.add_argument(
        "--run-tag",
        default="scaled_mlp_v2",
        help="Tag included in output filenames so this run never overwrites legacy results.",
    )
    parser.add_argument(
        "--random-model-if-missing",
        action="store_true",
        help="Smoke-test mode: if no checkpoint exists, create an untrained random EnsembleContModel.",
    )
    parser.add_argument(
        "--random-ensemble-size",
        type=int,
        default=4,
        help="Ensemble size for --random-model-if-missing.",
    )
    parser.add_argument(
        "--random-hidden-size",
        type=int,
        default=200,
        help="Hidden size for --random-model-if-missing.",
    )
    parser.add_argument("--goal-x", type=float, default=None, help="Optional goal x for AntMaze-style targets.")
    parser.add_argument("--goal-y", type=float, default=None, help="Optional goal y for AntMaze-style targets.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "papers" / "probe_outputs"),
        help="Directory for CSV outputs.",
    )
    return parser.parse_args()


def build_hdf5_dataset_with_ids(path: str) -> dict[str, np.ndarray]:
    import h5py

    obs_list, next_obs_list, action_list, reward_list = [], [], [], []
    terminal_list, timeout_list, episode_ids, timesteps = [], [], [], []
    info_chunks: dict[str, list[np.ndarray]] = {}

    with h5py.File(path, "r") as f:
        episode_names = sorted(
            [key for key in f.keys() if key.startswith("episode_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if not episode_names:
            raise ValueError(f"No episode_* groups found in {path}")

        for episode_id, ep_name in enumerate(episode_names):
            ep = f[ep_name]
            obs = np.asarray(ep["observations"], dtype=np.float32)
            actions = np.asarray(ep["actions"], dtype=np.float32)
            rewards = np.asarray(ep["rewards"], dtype=np.float32)
            terminations = np.asarray(ep["terminations"], dtype=np.float32).astype(bool)
            truncations = np.asarray(ep["truncations"], dtype=bool)

            t_steps = actions.shape[0]
            if obs.shape[0] != t_steps + 1:
                raise ValueError(
                    f"{ep_name}: expected observations length actions+1, got "
                    f"{obs.shape[0]} and {t_steps}"
                )

            obs_list.append(obs[:-1])
            next_obs_list.append(obs[1:])
            action_list.append(actions)
            reward_list.append(rewards)
            terminal_list.append(terminations)
            timeout_list.append(truncations)
            episode_ids.append(np.full(t_steps, episode_id, dtype=np.int64))
            timesteps.append(np.arange(t_steps, dtype=np.int64))

            if "infos" in ep:
                infos = ep["infos"]
                for key in infos.keys():
                    arr = np.asarray(infos[key])
                    if arr.shape[0] != t_steps:
                        continue
                    if arr.dtype == np.dtype("O"):
                        continue
                    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
                        continue
                    info_chunks.setdefault(key, []).append(arr.astype(np.float32))

    data = {
        "observations": np.concatenate(obs_list, axis=0).astype(np.float32),
        "actions": np.concatenate(action_list, axis=0).astype(np.float32),
        "next_observations": np.concatenate(next_obs_list, axis=0).astype(np.float32),
        "rewards": np.concatenate(reward_list, axis=0).astype(np.float32),
        "terminals": np.concatenate(terminal_list, axis=0).astype(bool),
        "timeouts": np.concatenate(timeout_list, axis=0).astype(bool),
        "episode_id": np.concatenate(episode_ids, axis=0).astype(np.int64),
        "timestep": np.concatenate(timesteps, axis=0).astype(np.int64),
        "infos": {},
    }

    for key, chunks in info_chunks.items():
        if len(chunks) == len(episode_ids):
            data["infos"][key] = np.concatenate(chunks, axis=0).astype(np.float32)

    return data


def build_world_dataset_with_ids(domain: str, env, terminate_on_end: bool = False) -> dict[str, np.ndarray]:
    raw = get_dataset(domain, env)
    has_next_obs = "next_observations" in raw
    if "timeouts" not in raw:
        raise KeyError("Dataset must contain a 'timeouts' array to reconstruct episodes.")

    obs_list, next_obs_list, action_list, reward_list = [], [], [], []
    terminal_list, timeout_list, episode_ids, timesteps = [], [], [], []

    episode_id = 0
    timestep = 0
    n = raw["rewards"].shape[0]

    for i in range(n - 1):
        obs = raw["observations"][i].astype(np.float32)
        if has_next_obs:
            next_obs = raw["next_observations"][i].astype(np.float32)
        else:
            next_obs = raw["observations"][i + 1].astype(np.float32)
        action = raw["actions"][i].astype(np.float32)
        reward = np.asarray(raw["rewards"][i], dtype=np.float32)

        done_bool = bool(raw["terminals"][i])
        final_timestep = bool(raw["timeouts"][i])

        skip = False
        if (not terminate_on_end) and final_timestep:
            skip = True
        if done_bool or final_timestep:
            if not has_next_obs:
                skip = True

        if not skip:
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            reward_list.append(float(reward))
            terminal_list.append(done_bool)
            timeout_list.append(final_timestep)
            episode_ids.append(episode_id)
            timesteps.append(timestep)

        if done_bool or final_timestep:
            episode_id += 1
            timestep = 0
        else:
            timestep += 1

    data = {
        "observations": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.float32),
        "next_observations": np.asarray(next_obs_list, dtype=np.float32),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "terminals": np.asarray(terminal_list, dtype=bool),
        "timeouts": np.asarray(timeout_list, dtype=bool),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "timestep": np.asarray(timesteps, dtype=np.int64),
    }

    if "antmaze" in domain and data["rewards"].max() == 0.0 and data["rewards"].min() == 0.0:
        data["rewards"] -= 1.0

    return data


def load_ensemble_checkpoint(
    domain: str,
    dataset_name: str,
    model_seed: int,
    obs_dim: int | None = None,
    act_dim: int | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_path: str | None = None,
    random_model_if_missing: bool = False,
    random_ensemble_size: int = 4,
    random_hidden_size: int = 200,
):
    save_dir = Path(checkpoint_dir) if checkpoint_dir else ROOT / "offline_world" / "ckpt" / domain / dataset_name
    if checkpoint_path:
        paths = [Path(checkpoint_path)]
    else:
        paths = sorted(save_dir.glob("ensemble_seed*.eqx"))
        if not paths:
            paths = sorted(save_dir.glob("latest_seed*.eqx"))

    if obs_dim is None or act_dim is None:
        env = make_env(domain, dataset_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

    key = jax.random.PRNGKey(128 + model_seed)
    _, model_key = jax.random.split(key)

    if not paths:
        if not random_model_if_missing:
            raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        hparams = {
            "ensemble_size": random_ensemble_size,
            "hidden_size": random_hidden_size,
            "has_ln": True,
            "random_untrained": True,
        }
        ensemble = EnsembleContModel(
            ensemble_size=hparams["ensemble_size"],
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hparams["hidden_size"],
            has_ln=hparams["has_ln"],
            key=model_key,
        )
        model_path = save_dir / "<random-untrained-model>"
        print(
            "WARNING: no checkpoint found; using an untrained random model for smoke testing only.",
            flush=True,
        )
        return eqx.nn.inference_mode(ensemble), hparams, model_path

    model_path = paths[model_seed % len(paths)]

    with open(model_path, "rb") as f:
        hparams = json.loads(f.readline().decode())
        template = EnsembleContModel(
            ensemble_size=hparams["ensemble_size"],
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hparams["hidden_size"],
            has_ln=hparams["has_ln"],
            key=model_key,
        )
        ensemble = eqx.tree_deserialise_leaves(f, template)

    return eqx.nn.inference_mode(ensemble), hparams, model_path


@eqx.filter_jit
def hidden_states_same_data(ensemble, x):
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def _apply(member, data):
        h1 = member.block1(data)
        h2 = member.block2(h1)
        h3 = member.block3(h2)
        h4 = member.block4(h3)
        return h1, h2, h3, h4

    return _apply(ensemble.members, x)


def extract_hidden_states(ensemble, x: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {"h1": [], "h2": [], "h3": [], "h4": []}
    for start in range(0, x.shape[0], batch_size):
        xb = jnp.asarray(x[start : start + batch_size])
        hs = hidden_states_same_data(ensemble, xb)
        for name, h in zip(["h1", "h2", "h3", "h4"], hs):
            chunks[name].append(np.asarray(h))
        print(f"  extracted hidden batch {start}:{min(start + batch_size, x.shape[0])}", flush=True)

    return {name: np.concatenate(parts, axis=1) for name, parts in chunks.items()}


def make_probe_features(h, y, episode_ids, aggregate: str, member_index: int):
    if aggregate == "member":
        h_out = h[member_index]
        y_out = y
        groups = episode_ids
    elif aggregate == "mean":
        h_out = h.mean(axis=0)
        y_out = y
        groups = episode_ids
    elif aggregate == "stack":
        ensemble_size, batch_size, hidden_dim = h.shape
        h_out = h.reshape(ensemble_size * batch_size, hidden_dim)
        y_out = np.tile(y, (ensemble_size, 1))
        groups = np.tile(episode_ids, ensemble_size)
    else:
        raise ValueError(f"Unknown aggregate mode: {aggregate}")

    return h_out.astype(np.float32), y_out.astype(np.float32), groups


def group_train_test_indices(groups, test_size=0.2, random_state=42):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(groups)), test_size=test_size, random_state=random_state
        )
    return train_idx, test_idx


def group_train_val_test_indices(groups, test_size=0.2, val_size=0.15, random_state=42):
    """Return disjoint train/validation/test indices, grouped by episode when possible."""
    train_val_idx, test_idx = group_train_test_indices(
        groups, test_size=test_size, random_state=random_state
    )
    train_val_groups = groups[train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)
    train_rel, val_rel = group_train_test_indices(
        train_val_groups,
        test_size=relative_val_size,
        random_state=random_state + 1,
    )
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    r2_per_dim = np.asarray(
        r2_score(y_true, y_pred, multioutput="raw_values"), dtype=np.float64
    ).reshape(-1)
    std_per_dim = np.std(y_true, axis=0)
    rmse_per_dim = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "r2_median": float(np.median(r2_per_dim)),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted")
        ),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "nrmse": float(np.mean(rmse_per_dim / np.maximum(std_per_dim, 1e-8))),
    }


def train_regression_probe(
    h,
    y,
    groups,
    ridge_alpha: float,
    random_state: int,
    run_mlp: bool,
    probe_seeds: list[int],
    mlp_max_epochs: int,
    mlp_patience: int,
) -> dict[str, float]:
    started = time.perf_counter()
    train_idx, val_idx, test_idx = group_train_val_test_indices(
        groups, random_state=random_state
    )
    h_train, h_val, h_test = h[train_idx], h[val_idx], h[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    ridge.fit(h_train, y_train)
    pred = ridge.predict(h_test)
    ridge_metrics = evaluate_regression(y_test, pred)

    result = {
        "ridge_r2": ridge_metrics["r2"],
        "ridge_mae": ridge_metrics["mae"],
        "ridge_rmse": ridge_metrics["rmse"],
        "ridge_r2_median": ridge_metrics["r2_median"],
        "ridge_r2_variance_weighted": ridge_metrics["r2_variance_weighted"],
        "ridge_nrmse": ridge_metrics["nrmse"],
        "ridge_elapsed_seconds": time.perf_counter() - started,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
    }

    if run_mlp:
        mlp_runs = []
        mlp_started = time.perf_counter()
        x_scaler = StandardScaler().fit(h_train)
        y_scaler = StandardScaler().fit(y_train)
        x_train = x_scaler.transform(h_train)
        x_val = x_scaler.transform(h_val)
        x_test = x_scaler.transform(h_test)
        y_train_scaled = y_scaler.transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)

        for seed in probe_seeds:
            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=3e-4,
                batch_size=min(256, len(train_idx)),
                max_iter=1,
                warm_start=True,
                early_stopping=False,
                random_state=seed,
            )
            best_model = None
            best_val_loss = np.inf
            stale_epochs = 0
            epochs_run = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for epoch in range(1, mlp_max_epochs + 1):
                    mlp.fit(x_train, y_train_scaled)
                    val_pred = np.asarray(mlp.predict(x_val)).reshape(y_val_scaled.shape)
                    val_loss = float(np.mean((val_pred - y_val_scaled) ** 2))
                    epochs_run = epoch
                    if val_loss < best_val_loss - 1e-7:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(mlp)
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    if stale_epochs >= mlp_patience:
                        break
            pred_scaled = np.asarray(best_model.predict(x_test)).reshape(y_test.shape)
            pred_mlp = y_scaler.inverse_transform(pred_scaled)
            metrics = evaluate_regression(y_test, pred_mlp)
            metrics["epochs"] = epochs_run
            metrics["best_val_loss"] = best_val_loss
            mlp_runs.append(metrics)

        def mean_std(key):
            values = np.asarray([run[key] for run in mlp_runs], dtype=np.float64)
            return float(values.mean()), float(values.std(ddof=0))

        mlp_r2, mlp_r2_std = mean_std("r2")
        mlp_mae, mlp_mae_std = mean_std("mae")
        mlp_rmse, mlp_rmse_std = mean_std("rmse")
        mlp_nrmse, mlp_nrmse_std = mean_std("nrmse")
        mlp_r2_median, _ = mean_std("r2_median")
        mlp_r2_vw, _ = mean_std("r2_variance_weighted")
        mlp_epochs, mlp_epochs_std = mean_std("epochs")
        result.update(
            {
                "mlp_r2": mlp_r2,
                "mlp_r2_std": mlp_r2_std,
                "mlp_mae": mlp_mae,
                "mlp_mae_std": mlp_mae_std,
                "mlp_rmse": mlp_rmse,
                "mlp_rmse_std": mlp_rmse_std,
                "mlp_nrmse": mlp_nrmse,
                "mlp_nrmse_std": mlp_nrmse_std,
                "mlp_r2_median": mlp_r2_median,
                "mlp_r2_variance_weighted": mlp_r2_vw,
                "mlp_epochs": mlp_epochs,
                "mlp_epochs_std": mlp_epochs_std,
                "mlp_num_seeds": len(probe_seeds),
                "mlp_elapsed_seconds": time.perf_counter() - mlp_started,
                "delta_mlp_minus_ridge": mlp_r2 - ridge_metrics["r2"],
            }
        )

    result["probe_elapsed_seconds"] = time.perf_counter() - started
    return result


def prepare_target(y, variance_epsilon=1e-10):
    """Drop constant dimensions; return target plus metadata for auditing."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    variances = np.var(y, axis=0)
    active = np.isfinite(variances) & (variances > variance_epsilon)
    metadata = {
        "target_dim_original": int(y.shape[1]),
        "target_dim_active": int(active.sum()),
        "target_dim_constant": int((~active).sum()),
        "target_variance_mean": float(np.mean(variances)),
    }
    if y.shape[1] == 1:
        metadata["target_unique_values"] = int(np.unique(y[:, 0]).size)
    else:
        metadata["target_unique_values"] = np.nan
    return y[:, active], metadata


def build_targets(obs, actions, next_obs, rewards, goal_xy=None, infos=None):
    targets = {
        "delta_obs": (next_obs - obs).astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "reward": rewards[:, None].astype(np.float32),
        "obs_sanity": obs.astype(np.float32),
    }

    if obs.shape[1] >= 2:
        targets["xy_t_sanity"] = obs[:, :2].astype(np.float32)
        targets["xy_tp1"] = next_obs[:, :2].astype(np.float32)
        targets["delta_xy"] = (next_obs[:, :2] - obs[:, :2]).astype(np.float32)

    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        dist_t = np.linalg.norm(obs[:, :2] - goal_xy[None, :], axis=1)
        dist_tp1 = np.linalg.norm(next_obs[:, :2] - goal_xy[None, :], axis=1)
        targets["dist_to_goal_t"] = dist_t[:, None].astype(np.float32)
        targets["dist_to_goal_tp1"] = dist_tp1[:, None].astype(np.float32)
        targets["progress_to_goal"] = (dist_t - dist_tp1)[:, None].astype(np.float32)

    if infos:
        preferred = [
            "command",
            "phase",
            "phase_dt",
            "push",
            "pert_dir",
            "pert_duration",
            "pert_duration_seconds",
            "pert_mag",
            "pert_steps",
            "steps_since_last_pert",
            "steps_until_next_cmd",
            "steps_until_next_pert",
            "feet_air_time",
            "swing_peak",
            "last_act",
            "last_last_act",
            "motor_targets",
        ]
        for key in preferred:
            if key not in infos:
                continue
            arr = infos[key]
            if arr.ndim == 1:
                arr = arr[:, None]
            # Keep very high-dimensional info arrays out by default except action-like targets.
            if arr.shape[1] > 64:
                continue
            targets[f"info_{key}"] = arr.astype(np.float32)

    return targets


def world_model_diagnostics_targets(ensemble, x, next_obs, rewards, scaler, batch_size):
    pred_chunks = []
    unc_chunks = {"epi_mean": [], "ale_max": [], "total_var": []}

    for start in range(0, x.shape[0], batch_size):
        xb = jnp.asarray(x[start : start + batch_size])
        (mu, _), unc = ensemble.forward_same_data(xb)
        pred_chunks.append(np.asarray(mu).mean(axis=0))
        for key in unc_chunks:
            unc_chunks[key].append(np.asarray(unc[key]))

    pred_scaled = np.concatenate(pred_chunks, axis=0)
    pred_raw = scaler.inverse_transform_outputs(pred_scaled)
    pred_next_obs = pred_raw[:, :-1]
    pred_reward = pred_raw[:, -1]

    out = {
        "wm_obs_error_norm": np.linalg.norm(pred_next_obs - next_obs, axis=1, keepdims=True).astype(np.float32),
        "wm_reward_abs_error": np.abs(pred_reward - rewards)[:, None].astype(np.float32),
    }
    for key, parts in unc_chunks.items():
        out[f"unc_{key}"] = np.concatenate(parts, axis=0)[:, None].astype(np.float32)

    return out


def run_controls(
    x, y, groups, hidden_dim, random_state, ridge_alpha,
    probe_seeds, mlp_max_epochs, mlp_patience
):
    rng = np.random.default_rng(random_state)
    rows = []

    metrics = train_regression_probe(
        x, y, groups, ridge_alpha, random_state, False,
        probe_seeds, mlp_max_epochs, mlp_patience
    )
    rows.append({"control": "input_obs_action", **metrics})

    perm = rng.permutation(len(y))
    metrics = train_regression_probe(
        x, y[perm], groups, ridge_alpha, random_state, False,
        probe_seeds, mlp_max_epochs, mlp_patience
    )
    rows.append({"control": "labels_shuffled", **metrics})

    h_random = rng.normal(size=(len(y), hidden_dim)).astype(np.float32)
    metrics = train_regression_probe(
        h_random, y, groups, ridge_alpha, random_state, False,
        probe_seeds, mlp_max_epochs, mlp_patience
    )
    rows.append({"control": "random_features", **metrics})

    return pd.DataFrame(rows)


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    probe_seed_text = args.probe_seeds.replace(":", ",")
    probe_seeds = [
        int(value.strip()) for value in probe_seed_text.split(",") if value.strip()
    ]
    if not probe_seeds and not args.skip_mlp:
        raise ValueError("--probe-seeds must contain at least one integer when MLP is enabled.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", ROOT)
    print("Args:", vars(args))

    if args.hdf5_dataset:
        data = build_hdf5_dataset_with_ids(args.hdf5_dataset)
    else:
        env = make_env(args.domain, args.dataset_name)
        data = build_world_dataset_with_ids(args.domain, env)
    print("Dataset transitions:", data["observations"].shape[0])
    print("Obs/action dims:", data["observations"].shape[1], data["actions"].shape[1])
    print("Episodes:", len(np.unique(data["episode_id"])))

    ensemble, hparams, model_path = load_ensemble_checkpoint(
        args.domain,
        args.dataset_name,
        args.model_seed,
        obs_dim=data["observations"].shape[1],
        act_dim=data["actions"].shape[1],
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        random_model_if_missing=args.random_model_if_missing,
        random_ensemble_size=args.random_ensemble_size,
        random_hidden_size=args.random_hidden_size,
    )
    print("Loaded checkpoint:", model_path)
    print("Checkpoint hparams:", hparams)

    rng = np.random.default_rng(args.random_state)
    n = data["observations"].shape[0]
    sample_idx = np.arange(n)
    if args.sample_size is not None and args.sample_size > 0 and args.sample_size < n:
        sample_idx = np.sort(rng.choice(n, size=args.sample_size, replace=False))
    print("Using transitions:", len(sample_idx))

    obs = data["observations"][sample_idx]
    actions = data["actions"][sample_idx]
    next_obs = data["next_observations"][sample_idx]
    rewards = data["rewards"][sample_idx]
    episode_ids = data["episode_id"][sample_idx]

    scaler = Scaler(data["observations"], data["actions"], data["rewards"])
    raw_inputs = np.concatenate([obs, actions], axis=-1)
    x = scaler.transform_inputs(raw_inputs).astype(np.float32)

    goal_xy = None
    if args.goal_x is not None or args.goal_y is not None:
        if args.goal_x is None or args.goal_y is None:
            raise ValueError("Provide both --goal-x and --goal-y, or neither.")
        goal_xy = np.array([args.goal_x, args.goal_y], dtype=np.float32)

    infos = {}
    if data.get("infos"):
        infos = {key: value[sample_idx] for key, value in data["infos"].items()}
    targets = build_targets(obs, actions, next_obs, rewards, goal_xy=goal_xy, infos=infos)
    print("Targets:", {k: v.shape for k, v in targets.items()})

    print("Extracting hidden states...")
    hidden = extract_hidden_states(ensemble, x, batch_size=args.batch_size)
    for name, h in hidden.items():
        print(name, h.shape)

    rows = []
    for layer_name, h in hidden.items():
        for target_name, y in targets.items():
            print(f"Probe layer={layer_name} target={target_name}", flush=True)
            h_probe, y_probe, groups = make_probe_features(
                h,
                y,
                episode_ids,
                aggregate=args.aggregate,
                member_index=args.member_index,
            )
            y_probe, target_metadata = prepare_target(y_probe)
            if y_probe.shape[1] == 0:
                print(f"  skipped constant target: {target_name}", flush=True)
                rows.append(
                    {
                        "domain": args.domain,
                        "dataset_name": args.dataset_name,
                        "model_seed": args.model_seed,
                        "layer": layer_name,
                        "target": target_name,
                        "target_dim": 0,
                        "aggregate": args.aggregate,
                        "member_index": args.member_index if args.aggregate == "member" else np.nan,
                        "status": "skipped_constant_target",
                        **target_metadata,
                    }
                )
                continue
            metrics = train_regression_probe(
                h_probe,
                y_probe,
                groups,
                ridge_alpha=args.ridge_alpha,
                random_state=args.random_state,
                run_mlp=not args.skip_mlp,
                probe_seeds=probe_seeds,
                mlp_max_epochs=args.mlp_max_epochs,
                mlp_patience=args.mlp_patience,
            )
            rows.append(
                {
                    "domain": args.domain,
                    "dataset_name": args.dataset_name,
                    "model_seed": args.model_seed,
                    "layer": layer_name,
                    "target": target_name,
                    "target_dim": y_probe.shape[1],
                    "aggregate": args.aggregate,
                    "member_index": args.member_index if args.aggregate == "member" else np.nan,
                    "status": "ok",
                    **target_metadata,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)

    diagnostic_targets = world_model_diagnostics_targets(
        ensemble, x, next_obs, rewards, scaler, batch_size=args.batch_size
    )
    diag_rows = []
    for layer_name, h in hidden.items():
        for target_name, y in diagnostic_targets.items():
            print(f"Diagnostic probe layer={layer_name} target={target_name}", flush=True)
            h_probe, y_probe, groups = make_probe_features(
                h,
                y,
                episode_ids,
                aggregate=args.aggregate,
                member_index=args.member_index,
            )
            y_probe, target_metadata = prepare_target(y_probe)
            if y_probe.shape[1] == 0:
                print(f"  skipped constant target: {target_name}", flush=True)
                diag_rows.append(
                    {
                        "domain": args.domain,
                        "dataset_name": args.dataset_name,
                        "model_seed": args.model_seed,
                        "layer": layer_name,
                        "target": target_name,
                        "target_dim": 0,
                        "aggregate": args.aggregate,
                        "member_index": args.member_index if args.aggregate == "member" else np.nan,
                        "status": "skipped_constant_target",
                        **target_metadata,
                    }
                )
                continue
            metrics = train_regression_probe(
                h_probe,
                y_probe,
                groups,
                ridge_alpha=args.ridge_alpha,
                random_state=args.random_state,
                run_mlp=not args.skip_mlp,
                probe_seeds=probe_seeds,
                mlp_max_epochs=args.mlp_max_epochs,
                mlp_patience=args.mlp_patience,
            )
            diag_rows.append(
                {
                    "domain": args.domain,
                    "dataset_name": args.dataset_name,
                    "model_seed": args.model_seed,
                    "layer": layer_name,
                    "target": target_name,
                    "target_dim": y_probe.shape[1],
                    "aggregate": args.aggregate,
                    "member_index": args.member_index if args.aggregate == "member" else np.nan,
                    "status": "ok",
                    **target_metadata,
                    **metrics,
                }
            )

    diagnostic_results = pd.DataFrame(diag_rows)

    control_target = "delta_obs"
    controls = run_controls(
        x,
        targets[control_target],
        episode_ids,
        hidden_dim=hidden["h1"].shape[-1],
        random_state=args.random_state,
        ridge_alpha=args.ridge_alpha,
        probe_seeds=probe_seeds,
        mlp_max_epochs=args.mlp_max_epochs,
        mlp_patience=args.mlp_patience,
    )
    controls.insert(0, "target", control_target)
    controls.insert(0, "dataset_name", args.dataset_name)
    controls.insert(0, "domain", args.domain)

    dataset_slug = args.dataset_name
    if args.hdf5_dataset:
        dataset_slug = Path(args.hdf5_dataset).parents[1].name
    safe_run_tag = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in args.run_tag
    )
    stem = (
        f"{args.domain}__{dataset_slug}__seed{args.model_seed}__"
        f"{args.aggregate}__{safe_run_tag}"
    )
    results_path = output_dir / f"{stem}__probe_results.csv"
    diag_path = output_dir / f"{stem}__diagnostic_probe_results.csv"
    controls_path = output_dir / f"{stem}__controls.csv"
    summary_path = output_dir / f"{stem}__summary.json"

    results.to_csv(results_path, index=False)
    diagnostic_results.to_csv(diag_path, index=False)
    controls.to_csv(controls_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "root": str(ROOT),
                "checkpoint": str(model_path),
                "hparams": hparams,
                "num_transitions_total": int(n),
                "num_transitions_used": int(len(sample_idx)),
                "probe_seeds": probe_seeds,
                "elapsed_seconds": time.perf_counter() - run_started,
                "outputs": {
                    "probe_results": str(results_path),
                    "diagnostic_probe_results": str(diag_path),
                    "controls": str(controls_path),
                },
            },
            indent=2,
        )
        + "\n"
    )

    print("\nSaved:")
    print(" ", results_path)
    print(" ", diag_path)
    print(" ", controls_path)
    print(" ", summary_path)
    print(f"\nTotal elapsed: {time.perf_counter() - run_started:.2f} seconds")
    print("\nTop Ridge R2 by target:")
    valid_results = results[results["status"] == "ok"]
    print(valid_results.sort_values("ridge_r2", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
