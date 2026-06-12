from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
import random as rnd
from collections import defaultdict
import gymnasium as gym


class Agent(ABC):
    """Abstract base class that all RL agents must implement.

    Enforces a common interface so run_episode, Trainer, and any future
    tooling (sweepers, visualisers) work with any agent without isinstance
    checks or special-casing.
    """

    @abstractmethod
    def get_action(self, obs) -> int:
        """Select an action given the current (preprocessed) observation."""
        ...

    @abstractmethod
    def train_step(self, obs, action: int, reward: float,
                   next_obs, done: bool) -> "int | None":
        """Apply one training update.

        Returns the next action for on-policy agents (SARSA), or None
        for off-policy agents so run_episode calls get_action normally.
        """
        ...

    @abstractmethod
    def preprocess(self, raw_obs: np.ndarray, obs_cfg: dict):
        """Convert a raw env observation to the agent's expected input format."""
        ...

    @abstractmethod
    def decay_epsilon(self) -> None:
        """Step the exploration schedule forward by one episode."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent weights/state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent weights/state from disk."""
        ...


class BaseAgent(Agent):
    """Shared state and helpers for all tabular Q-learning agents.

    Subclasses only need to implement update(); all other Agent methods
    are handled here.
    """

    def __init__(self, hp: dict, policy: str, env: gym.Env):
        self.env = env
        self.lr = hp["learning_rate"]
        self.gamma = hp["discount_factor"]
        self.epsilon = hp["epsilon"]
        self.epsilon_decay = hp["exploration_decay"]
        self.epsilon_min = hp["final_epsilon"]
        self.temperature = hp.get("temperature", 1.0)
        self.policy_type = policy

        self.q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.counts = defaultdict(lambda: np.zeros(env.action_space.n))
        self.td_errors: list = []
        self.episode_rewards: list = []

    # ── Action selection ──────────────────────────────────────────────────────

    def get_action(self, obs: tuple) -> int:
        return {
            "epsilon_greedy": self._eps_greedy,
            "softmax": self._softmax,
            "ucb": self._ucb,
        }.get(self.policy_type, self._eps_greedy)(obs)

    def _eps_greedy(self, obs):
        if rnd.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q[obs]))

    def _softmax(self, obs):
        q = self.q[obs]
        q_s = q - np.max(q)
        exp_q = np.exp(q_s / max(self.temperature, 1e-8))
        return int(np.random.choice(len(q), p=exp_q / exp_q.sum()))

    def _ucb(self, obs, c: float = 2.0):
        n_total = self.counts[obs].sum() + 1
        bonus = c * np.sqrt(np.log(n_total) / (self.counts[obs] + 1))
        return int(np.argmax(self.q[obs] + bonus))

    # ── Agent interface ───────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def preprocess(self, raw_obs: np.ndarray, obs_cfg: dict):
        """Convert a raw observation to the form expected by this agent."""
        from utils.preprocessing import preprocess
        return preprocess(raw_obs, obs_cfg)

    def train_step(self, obs, action: int, reward: float,
                   next_obs, done: bool) -> None:
        """Apply one training update. Off-policy: returns None."""
        self.update(obs, action, reward, next_obs, done=done)
        return None

    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        raise NotImplementedError

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"q": dict(self.q)}, f)
        print(f"[save] {path}.pkl")

    def load(self, path: str) -> None:
        pkl = path + ".pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                self.q.update(pickle.load(f)["q"])
            print(f"[load] {pkl}")
