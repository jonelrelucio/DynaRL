import argparse
import csv
import gymnasium as gym
import numpy as np
import random as rnd
import yaml
import pickle
import os
from collections import defaultdict, deque

# PyTorch is only required for DQN — tabular agents work without it
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Action sets for continuous mode -----------------------------------------

ACTION_SETS = {
    "standard": [
        [ 0.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.8],
        [-0.5,  0.5,  0.0],
        [ 0.5,  0.5,  0.0],
    ],
    "aggressive": [
        [ 0.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [-1.0,  0.5,  0.0],
        [ 1.0,  0.5,  0.0],
        [-0.5,  0.0,  0.5],
        [ 0.5,  0.0,  0.5],
    ],
    "smooth": [
        [ 0.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-0.5,  0.0,  0.0],
        [ 0.5,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.5,  0.0],
        [ 0.0,  0.0,  0.8],
        [-0.5,  0.5,  0.0],
        [ 0.5,  0.5,  0.0],
        [-1.0,  0.5,  0.0],
        [ 1.0,  0.5,  0.0],
    ],
}


class DiscretizedEnv(gym.Wrapper):
    """Wrap a continuous Box action env with a fixed discrete action set."""

    def __init__(self, env: gym.Env, action_set_name: str = "standard"):
        super().__init__(env)
        if action_set_name not in ACTION_SETS:
            raise ValueError(f"Unknown action set '{action_set_name}'. Choose from: {list(ACTION_SETS)}")
        self._actions = np.array(ACTION_SETS[action_set_name], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        self._action_set_name = action_set_name

    def step(self, action: int):
        return self.env.step(self._actions[action])


# --- Observation preprocessing (tabular agents only) -------------------------

def extract_features(obs: np.ndarray, n_bins: int = 4, n_regions: int = 4) -> tuple:
    r, g, b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    road_mask = (
        (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 20) &
        (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 20) &
        (gray > 80) & (gray < 180)
    )
    h, strip_h = road_mask.shape[0], road_mask.shape[0] // n_regions
    features = []
    for i in range(n_regions):
        strip = road_mask[i * strip_h:(i + 1) * strip_h, :]
        features.append(min(int(float(np.mean(strip)) * n_bins), n_bins - 1))
    return tuple(features)


def to_tuple(a):
    """Recursively convert a nested array/iterable into a nested tuple.

    Used by the 'raw' observation method to make numpy arrays hashable
    for tabular Q-tables. Note: the raw method is impractical at scale
    (state space explodes); prefer the default 'feature' method.
    """
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def preprocess(raw_obs: np.ndarray, obs_cfg: dict) -> tuple:
    if obs_cfg.get("method", "feature") == "feature":
        return extract_features(raw_obs, obs_cfg.get("n_bins", 4), obs_cfg.get("n_regions", 4))
    return to_tuple(raw_obs)


# --- Neural network components (DQN) -----------------------------------------

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
        self.buf = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool):
        self.buf.append((
            obs.transpose(2, 0, 1).astype(np.uint8),
            int(action),
            float(reward),
            next_obs.transpose(2, 0, 1).astype(np.uint8),
            float(done),
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = rnd.sample(self.buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)

        # Pre-allocate contiguous arrays instead of letting np.array copy
        # each element individually — meaningfully faster for large batches.
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



# --- Tabular base agent -------------------------------------------------------

class BaseAgent:
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
        self.td_errors = []
        self.episode_rewards = []

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

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def preprocess(self, raw_obs: np.ndarray, obs_cfg: dict):
        """Convert a raw observation to the form expected by this agent."""
        return preprocess(raw_obs, obs_cfg)

    def train_step(self, obs, action: int, reward: float,
                   next_obs, done: bool) -> int | None:
        """Apply one training update and return the next action (or None).

        Off-policy agents return None; on-policy agents (SARSA) return
        the next action so it can be reused on the next step.
        """
        self.update(obs, action, reward, next_obs, done=done)
        return None

    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        raise NotImplementedError

    def save(self, path: str):
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"q": dict(self.q)}, f)
        print(f"[save] {path}.pkl")

    def load(self, path: str):
        pkl = path + ".pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                self.q.update(pickle.load(f)["q"])
            print(f"[load] {pkl}")


class QLearningAgent(BaseAgent):
    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        td_err = reward + self.gamma * np.max(self.q[next_obs]) - self.q[obs][action]
        self.q[obs][action] += self.lr * td_err
        self.counts[obs][action] += 1
        self.td_errors.append(abs(td_err))


class DoubleQLearningAgent(BaseAgent):
    def __init__(self, hp, policy, env):
        super().__init__(hp, policy, env)
        self.q2 = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        combined = self.q[obs] + self.q2[obs]
        if self.policy_type == "epsilon_greedy":
            if rnd.random() < self.epsilon:
                return self.env.action_space.sample()
            return int(np.argmax(combined))
        if self.policy_type == "softmax":
            q_s = combined - np.max(combined)
            exp_q = np.exp(q_s / max(self.temperature, 1e-8))
            return int(np.random.choice(len(combined), p=exp_q / exp_q.sum()))
        return int(np.argmax(combined))

    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        # Zero out the next-state value at terminal transitions
        discount = 0.0 if done else self.gamma
        if rnd.random() < 0.5:
            best = int(np.argmax(self.q[next_obs]))
            td_err = reward + discount * self.q2[next_obs][best] - self.q[obs][action]
            self.q[obs][action] += self.lr * td_err
        else:
            best = int(np.argmax(self.q2[next_obs]))
            td_err = reward + discount * self.q[next_obs][best] - self.q2[obs][action]
            self.q2[obs][action] += self.lr * td_err
        self.counts[obs][action] += 1
        self.td_errors.append(abs(td_err))

    def save(self, path: str):
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"q": dict(self.q), "q2": dict(self.q2)}, f)
        print(f"[save] {path}.pkl")

    def load(self, path: str):
        pkl = path + ".pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            self.q.update(data["q"])
            self.q2.update(data.get("q2", {}))
            print(f"[load] {pkl}")


class SARSAAgent(BaseAgent):
    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        if next_action is None:
            next_action = self.get_action(next_obs)
        # At terminal transitions the next-state value must be zero
        next_q = 0.0 if done else self.q[next_obs][next_action]
        td_err = reward + self.gamma * next_q - self.q[obs][action]
        self.q[obs][action] += self.lr * td_err
        self.counts[obs][action] += 1
        self.td_errors.append(abs(td_err))
        return next_action

    def train_step(self, obs, action: int, reward: float,
                   next_obs, done: bool) -> int:
        """SARSA is on-policy: returns the next action to reuse on the next step."""
        return self.update(obs, action, reward, next_obs, done=done)


class ExpectedSARSAAgent(BaseAgent):
    def _probs(self, obs):
        n, q = self.env.action_space.n, self.q[obs]
        if self.policy_type == "softmax":
            q_s = q - np.max(q)
            exp_q = np.exp(q_s / max(self.temperature, 1e-8))
            return exp_q / exp_q.sum()
        probs = np.full(n, self.epsilon / n)
        probs[int(np.argmax(q))] += 1.0 - self.epsilon
        return probs

    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        td_err = reward + self.gamma * float(np.dot(self._probs(next_obs), self.q[next_obs])) - self.q[obs][action]
        self.q[obs][action] += self.lr * td_err
        self.counts[obs][action] += 1
        self.td_errors.append(abs(td_err))


# --- DQN agent ----------------------------------------------------------------

class DQNAgent:
    def __init__(self, hp: dict, policy: str, env: gym.Env, double: bool = False):
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
        # Use bounded deques to avoid manual trimming
        self.td_errors = deque(maxlen=10_000)
        self.episode_rewards = []

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

        if self.steps % 1000 == 0:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path + ".pt")
        print(f"[save] {path}.pt")

    def load(self, path: str):
        pt = path + ".pt"
        if os.path.exists(pt):
            data = torch.load(pt, map_location=self.device)
            self.online.load_state_dict(data["online"])
            self.target.load_state_dict(data["target"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.steps = data.get("steps", self.steps)
            print(f"[load] {pt}")


TABULAR_ALGORITHMS = {
    "q_learning": QLearningAgent,
    "double_q_learning": DoubleQLearningAgent,
    "sarsa": SARSAAgent,
    "expected_sarsa": ExpectedSARSAAgent,
}


def make_agent(mode_cfg: dict, env: gym.Env):
    name = mode_cfg.get("algorithm", "q_learning")
    hp = mode_cfg["hyperparameters"]
    pol = mode_cfg.get("policy", "epsilon_greedy")

    if name == "dqn":
        return DQNAgent(hp, pol, env, double=False)
    if name == "double_dqn":
        return DQNAgent(hp, pol, env, double=True)
    if name in TABULAR_ALGORITHMS:
        return TABULAR_ALGORITHMS[name](hp=hp, policy=pol, env=env)
    raise ValueError(f"Unknown algorithm '{name}'")


def run_episode(env, agent, obs_cfg: dict, training: bool = True, seed: int | None = None) -> float:
    """Run one episode, optionally training the agent.

    Uses each agent's preprocess() and train_step() methods so no
    isinstance checks are needed here — adding a new agent type only
    requires implementing those two methods.
    """
    reset_kw = {"seed": seed} if seed is not None else {}
    raw_obs, _ = env.reset(**reset_kw)

    obs = agent.preprocess(raw_obs, obs_cfg)
    action = agent.get_action(obs)
    done = False
    total_reward = 0.0

    while not done:
        raw_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = agent.preprocess(raw_next, obs_cfg)
        total_reward += reward

        if training:
            next_action = agent.train_step(obs, action, reward, next_obs, done)
            # On-policy agents (e.g. SARSA) return the next action to reuse;
            # off-policy agents return None so we query get_action normally.
            action = next_action if next_action is not None else agent.get_action(next_obs)
        else:
            action = agent.get_action(next_obs)

        obs = next_obs

    return total_reward


def build_env(cfg: dict):
    mode = cfg["environment"].get("mode", "discrete")
    env_name = cfg["environment"]["name"]
    render = cfg["environment"].get("render_mode", "rgb_array")
    base_kw = dict(
        lap_complete_percent=cfg["environment"].get("lap_complete_percent", 0.95),
        domain_randomize=cfg["environment"].get("domain_randomize", False),
    )

    if mode == "dqn":
        mode_cfg = cfg["dqn"]
        use_continuous = mode_cfg.get("use_continuous", False)
        raw_env = gym.make(env_name, render_mode=render, continuous=use_continuous, **base_kw)
        if use_continuous:
            aset = mode_cfg.get("action_set", "smooth")
            env = DiscretizedEnv(raw_env, aset)
            label = f"dqn (CNN) / continuous -> '{aset}' ({env.action_space.n} actions)"
        else:
            env = raw_env
            label = "dqn (CNN) / discrete -> 5 actions"

    elif mode == "continuous":
        mode_cfg = cfg["continuous"]
        aset = mode_cfg.get("action_set", "standard")
        raw_env = gym.make(env_name, render_mode=render, continuous=True, **base_kw)
        env = DiscretizedEnv(raw_env, aset)
        label = f"tabular / continuous -> '{aset}' ({env.action_space.n} actions)"

    else:
        mode_cfg = cfg["discrete"]
        env = gym.make(env_name, render_mode=render, continuous=False, **base_kw)
        label = "tabular / discrete -> 5 actions"

    return env, mode_cfg, label


def _stats_suffix(agent) -> str:
    if isinstance(agent, DQNAgent):
        buf = len(agent.replay)
        ready = buf >= agent.train_start
        return f"buf {buf:6d}{'  training' if ready else '  warming up'}"
    return f"states {len(agent.q)}"


def _open_csv_logger(save_path: str | None):
    """Open a CSV file for per-episode logging next to the checkpoint."""
    if not save_path:
        return None, None
    csv_path = save_path + "_log.csv"
    existed = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    if not existed:
        writer.writerow(["episode", "reward", "epsilon", "avg_td_error"])
    return f, writer


def main():
    parser = argparse.ArgumentParser(description="Train a RL agent on CarRacing.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML config file (default: config.yaml)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed built-in, numpy, and torch RNGs for reproducibility.
    seed = cfg.get("seed", 42)
    rnd.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env, mode_cfg, mode_label = build_env(cfg)
    agent = make_agent(mode_cfg, env)
    obs_cfg = cfg.get("observation", {})
    save_path = cfg.get("save_path")

    if save_path:
        agent.load(save_path)

    n_episodes = cfg["environment"]["n_episodes"]
    log_every = cfg["environment"].get("log_interval", 25)
    checkpoint_every = cfg["environment"].get("checkpoint_interval", 0)

    csv_file, csv_writer = _open_csv_logger(save_path)

    print(f"{'-'*62}")
    print(f"  Mode      : {mode_label}")
    print(f"  Algorithm : {mode_cfg.get('algorithm', 'q_learning')}")
    print(f"  Policy    : {mode_cfg.get('policy', 'epsilon_greedy')}")
    if not isinstance(agent, DQNAgent):
        print(f"  Obs mode  : {obs_cfg.get('method', 'feature')}")
    else:
        print(f"  Device    : {agent.device}")
    print(f"  Episodes  : {n_episodes}")
    if checkpoint_every:
        print(f"  Checkpoint: every {checkpoint_every} episodes")
    print(f"{'-'*62}")

    for ep in range(n_episodes):
        reward = run_episode(env, agent, obs_cfg, training=True, seed=ep)
        agent.episode_rewards.append(reward)
        if len(agent.episode_rewards) > 5_000:
            del agent.episode_rewards[:2_500]
        agent.decay_epsilon()
        avg_td = float(np.mean(list(agent.td_errors)[-1000:])) if agent.td_errors else 0.0

        if csv_writer:
            csv_writer.writerow([ep + 1, f"{reward:.4f}",
                                  f"{agent.epsilon:.6f}", f"{avg_td:.6f}"])
            csv_file.flush()

        print(
            f"\r  ep {ep+1:4d}/{n_episodes}  "
            f"reward {reward:8.2f}  "
            f"ε {agent.epsilon:.4f}  "
            f"{_stats_suffix(agent)}",
            end="", flush=True,
        )

        if (ep + 1) % log_every == 0:
            window = agent.episode_rewards[-log_every:]
            print(
                f"\n  ├ avg_reward {np.mean(window):8.2f}"
                f"  TD_err {avg_td:.4f}"
            )

        if checkpoint_every and save_path and (ep + 1) % checkpoint_every == 0:
            agent.save(f"{save_path}_ep{ep+1}")

    print()
    if save_path:
        agent.save(save_path)

    if csv_file:
        csv_file.close()

    env.close()


if __name__ == "__main__":
    main()
