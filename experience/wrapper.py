import json
from pathlib import Path

import numpy as np
import gym, d4rl, neorl
from gym import spaces
from gym.wrappers import TimeLimit
from gym.core import ActType, ObsType
from typing import Tuple


class HDF5OfflineEnv(gym.Env):
    """Minimal Gym metadata/dataset adapter for Minari-style HDF5 datasets."""

    def __init__(self, dataset_name: str, dataset_path: str = None):
        self.dataset_name = dataset_name
        self.dataset_path = self._resolve_dataset_path(dataset_name, dataset_path)
        metadata_path = self.dataset_path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}

        obs_shape = self._metadata_shape(metadata.get("observation_space"))
        act_shape = self._metadata_shape(metadata.get("action_space"))
        if obs_shape is None or act_shape is None:
            import h5py

            with h5py.File(self.dataset_path, "r") as handle:
                episode_names = sorted(
                    (name for name in handle.keys() if name.startswith("episode_")),
                    key=lambda name: int(name.split("_")[-1]),
                )
                if not episode_names:
                    raise ValueError(f"No episode_* groups found in {self.dataset_path}")
                first_episode = episode_names[0]
                obs_shape = tuple(handle[first_episode]["observations"].shape[1:])
                act_shape = tuple(handle[first_episode]["actions"].shape[1:])

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=act_shape, dtype=np.float32
        )
        self._max_episode_steps = 1000
        self.max_episode_steps = self._max_episode_steps
        self._dataset = None

    @staticmethod
    def _metadata_shape(value):
        if not value:
            return None
        parsed = json.loads(value) if isinstance(value, str) else value
        return tuple(parsed["shape"])

    @staticmethod
    def _resolve_dataset_path(dataset_name, dataset_path):
        if dataset_path:
            candidate = Path(dataset_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()
        candidates = [
            Path("datasets") / dataset_name / "data" / "main_data.hdf5",
            Path("datasets/akcit-rl_playground_backup")
            / dataset_name
            / "data"
            / "main_data.hdf5",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "Go2 HDF5 dataset not found. Set dataset_path in configs/go2/base.yaml."
        )

    def get_dataset(self, **kwargs):
        if self._dataset is not None:
            return self._dataset

        import h5py

        chunks = {key: [] for key in [
            "observations", "actions", "next_observations", "rewards",
            "terminals", "timeouts"
        ]}
        with h5py.File(self.dataset_path, "r") as handle:
            episode_names = sorted(
                (name for name in handle.keys() if name.startswith("episode_")),
                key=lambda name: int(name.split("_")[-1]),
            )
            if not episode_names:
                raise ValueError(f"No episode_* groups found in {self.dataset_path}")
            for episode_name in episode_names:
                episode = handle[episode_name]
                observations = np.asarray(episode["observations"], dtype=np.float32)
                actions = np.asarray(episode["actions"], dtype=np.float32)
                rewards = np.asarray(episode["rewards"], dtype=np.float32)
                terminals = np.asarray(episode["terminations"]).astype(bool)
                timeouts = np.asarray(episode["truncations"]).astype(bool)
                transition_count = actions.shape[0]
                if observations.shape[0] != transition_count + 1:
                    raise ValueError(
                        f"{episode_name}: observations must have T+1 rows"
                    )
                for field_name, values in [
                    ("rewards", rewards),
                    ("terminations", terminals),
                    ("truncations", timeouts),
                ]:
                    if values.shape[0] != transition_count:
                        raise ValueError(
                            f"{episode_name}: {field_name} must have T rows"
                        )
                if transition_count and not (terminals[-1] or timeouts[-1]):
                    timeouts[-1] = True
                chunks["observations"].append(observations[:-1])
                chunks["next_observations"].append(observations[1:])
                chunks["actions"].append(actions)
                chunks["rewards"].append(rewards)
                chunks["terminals"].append(terminals)
                chunks["timeouts"].append(timeouts)

        self._dataset = {
            key: np.concatenate(parts, axis=0) for key, parts in chunks.items()
        }
        return self._dataset


def make_env(domain: str, dataset_name: str, dataset_path: str = None):
    if "go2" in domain.lower():
        env = HDF5OfflineEnv(dataset_name, dataset_path=dataset_path)
    elif "neorl" in domain:
        env = neorl.make(dataset_name)
    elif "antmaze" in domain:
        env = AntMazeWrapper(gym.make(dataset_name))
    else:
        env = gym.make(dataset_name)
    return env


class AntMazeWrapper(gym.Wrapper):
    """
    Shift reward by -1 in antmaze following LEQ and IQL papers.
    As our policy is reward-conditioned, we need to have a wrapper on it
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert "antmaze" in env.spec.id
        self._max_episode_steps = env._max_episode_steps

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        reward -= 1.0
        return obs, reward, done, info

    def get_normalized_score(self, score):
        # add back the shift before calling this function
        return self.env.get_normalized_score(score)


def find_time_limit_wrapper(env: gym.Env):
    """
    Walk down .env links until we hit a TimeLimit wrapper.
    Returns the wrapper or None if it does not exist.
    """
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, TimeLimit):
            return current
        current = current.env  # move one level deeper
    return None


class MarkovWrapper(gym.Wrapper):
    """Used in ablation study for Markov agent."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def max_episode_steps(self):
        """
        Returns the max episode step if a TimeLimit wrapper is present,
        otherwise None. Usually this wrapper is the last one in the chain.
        """
        tl = find_time_limit_wrapper(self)
        return None if tl is None else tl._max_episode_steps  # private attr


class ActionRewardWrapper(gym.Wrapper):
    """
    Appends the *previous action* (one‑hot if Discrete, raw if Box) and the
    *current reward* to every observation.

    Resulting observation: concat([obs_flat, action_vec, reward_scalar])
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box)
        obs_low = env.observation_space.low.flatten().astype(np.float32)
        obs_high = env.observation_space.high.flatten().astype(np.float32)

        if isinstance(env.action_space, spaces.Discrete):
            # one-hot float encoding
            self._encode = lambda a: np.eye(env.action_space.n, dtype=np.float32)[
                int(a)
            ]
            act_low = np.zeros(env.action_space.n, dtype=np.float32)
            act_high = np.ones(env.action_space.n, dtype=np.float32)
            null_action = 0
        elif isinstance(env.action_space, spaces.Box):
            self._encode = lambda a: np.asarray(a, dtype=np.float32).flatten()
            act_low = env.action_space.low.flatten().astype(np.float32)
            act_high = env.action_space.high.flatten().astype(np.float32)
            null_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        else:
            raise NotImplementedError(
                "Only Discrete and Box action spaces are supported"
            )

        self.null_action_vec = self._encode(null_action)

        # ----- augmented observation space -----
        low = np.concatenate([obs_low, act_low, np.array([-np.inf], dtype=np.float32)])
        high = np.concatenate(
            [obs_high, act_high, np.array([np.inf], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def max_episode_steps(self):
        """
        Returns the max episode step if a TimeLimit wrapper is present,
        otherwise None. Usually this wrapper is the last one in the chain.
        """
        tl = find_time_limit_wrapper(self)
        return None if tl is None else tl._max_episode_steps  # private attr

    def reset(self, **kwargs) -> ObsType:
        obs = self.env.reset(**kwargs)
        augmented = np.concatenate(
            [
                obs.flatten().astype(np.float32),
                self.null_action_vec,
                np.array([0.0], dtype=np.float32),
            ]
        )
        return augmented

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        augmented = np.concatenate(
            [
                obs.flatten().astype(np.float32),
                self._encode(action),
                np.array([reward], dtype=np.float32),
            ]
        )
        return augmented, reward, done, info
