import random as rnd
import os
import numpy as np
from collections import deque

from agents.base import Agent

# PyTorch is only required for DQN — tabular agents work without it
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CNN(nn.Module):
    """
    Convolutional Q-network.
    Input:  (B, 3, 96, 96)
    Output: (B, n_actions)
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x / 255.0)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool):
        self.buf.append((
            obs.transpose(2, 0, 1).astype(np.uint8),
            int(action),
            float(reward),
            next_obs.transpose(2, 0, 1).astype(np.uint8),
            float(done),
        ))

    def sample(self, batch_size: int, device: "torch.device"):
        batch = rnd.sample(self.buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)

        # Pre-allocate contiguous arrays — faster than np.array(list_of_arrays)
        C, H, W = obs[0].shape
        obs_arr = np.empty((batch_size, C, H, W), dtype=np.uint8)
        nxt_arr = np.empty((batch_size, C, H, W), dtype=np.uint8)
        for i in range(batch_size):
            obs_arr[i] = obs[i]
            nxt_arr[i] = next_obs[i]

        def t(x, dtype):
            return torch.tensor(x, dtype=dtype, device=device)

        return (
            t(obs_arr, torch.float32),
            t(np.array(acts, dtype=np.int64), torch.long),
            t(np.array(rews, dtype=np.float32), torch.float32),
            t(nxt_arr, torch.float32),
            t(np.array(dones, dtype=np.float32), torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent(Agent):
    def __init__(self, hp: dict, policy: str, env, double: bool = False):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not found. Install with: pip install torch")

        self.env = env
        self.lr = hp["learning_rate"]
        self.gamma = hp["discount_factor"]
        self.epsilon = hp["epsilon"]
        self.epsilon_decay = hp["exploration_decay"]
        self.epsilon_min = hp["final_epsilon"]
        self.policy_type = policy
        self.double = double

        self.batch_size = hp.get("batch_size", 32)
        self.train_start = hp.get("train_start", 10_000)
        self.target_update_freq = hp.get("target_update_freq", 1_000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n = env.action_space.n
        self.online = CNN(n).to(self.device)
        self.target = CNN(n).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(hp.get("replay_buffer_size", 50_000))

        self.steps = 0
        self.td_errors: deque = deque(maxlen=10_000)
        self.episode_rewards: list = []

    # ── Agent interface ───────────────────────────────────────────────────────

    def preprocess(self, raw_obs: np.ndarray, obs_cfg: dict) -> np.ndarray:
        """DQN operates on raw pixel observations — no tabular preprocessing."""
        return raw_obs

    def train_step(self, obs: np.ndarray, action: int, reward: float,
                   next_obs: np.ndarray, done: bool) -> None:
        """Push transition to replay buffer and run one gradient update."""
        self.update(obs, action, reward, next_obs, done)
        return None

    def get_action(self, obs: np.ndarray) -> int:
        if rnd.random() < self.epsilon:
            return self.env.action_space.sample()
        t = torch.from_numpy(obs.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online(t).argmax(dim=1).item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool = False, next_action=None):
        self.replay.push(obs, action, reward, next_obs, done)
        self.steps += 1

        if len(self.replay) < self.train_start:
            return

        obs_t, act_t, rew_t, next_t, done_t = self.replay.sample(self.batch_size, self.device)
        q_curr = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                best_a = self.online(next_t).argmax(dim=1)
                q_next = self.target(next_t).gather(1, best_a.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.target(next_t).max(dim=1).values
            td_target = rew_t + self.gamma * q_next * (1.0 - done_t)

        loss = nn.functional.huber_loss(q_curr, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.td_errors.append(float((q_curr - td_target).abs().mean()))

        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        if self.steps % 1000 == 0 and self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path + ".pt")
        print(f"[save] {path}.pt")

    def load(self, path: str) -> None:
        pt = path + ".pt"
        if os.path.exists(pt):
            data = torch.load(pt, map_location=self.device)
            self.online.load_state_dict(data["online"])
            self.target.load_state_dict(data["target"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.steps = data.get("steps", self.steps)
            print(f"[load] {pt}")
