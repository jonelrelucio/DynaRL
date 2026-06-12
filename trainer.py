import numpy as np
from typing import Sequence

from agents.base import Agent
from agents.dqn import DQNAgent
from utils.logging import Logger


def run_episode(env, agent: Agent, obs_cfg: dict,
                training: bool = True, seed: "int | None" = None) -> float:
    """Run one episode, optionally training the agent.

    Uses each agent's preprocess() and train_step() methods so no
    isinstance checks are needed here — adding a new agent type only
    requires implementing those two methods on the Agent ABC.
    """
    reset_kw = {"seed": seed} if seed is not None else {}
    raw_obs, _ = env.reset(**reset_kw)

    obs    = agent.preprocess(raw_obs, obs_cfg)
    action = agent.get_action(obs)
    done   = False
    total_reward = 0.0

    while not done:
        raw_next, reward, terminated, truncated, _ = env.step(action)
        done     = terminated or truncated
        next_obs = agent.preprocess(raw_next, obs_cfg)
        total_reward += reward

        if training:
            next_action = agent.train_step(obs, action, reward, next_obs, done)
            # On-policy agents (SARSA) return the next action to reuse;
            # off-policy agents return None so we call get_action normally.
            action = next_action if next_action is not None else agent.get_action(next_obs)
        else:
            action = agent.get_action(next_obs)

        obs = next_obs

    return total_reward


def _stats_suffix(agent: Agent) -> str:
    if isinstance(agent, DQNAgent):
        buf   = len(agent.replay)
        ready = buf >= agent.train_start
        return f"buf {buf:6d}{'  training' if ready else '  warming up'}"
    return f"states {len(agent.q)}"


class Trainer:
    """Encapsulates the full training loop.

    Separating training into a class makes it easy to:
    - Construct and run programmatically (e.g. hyperparameter sweeps).
    - Unit-test individual methods without running the full loop.
    - Swap loggers, agents, or envs without touching training code.

    Args:
        cfg:        Full parsed YAML config dict.
        env:        A ready-to-use gym environment.
        agent:      Any Agent-conforming object.
        loggers:    Zero or more Logger instances (e.g. CSVLogger, ConsoleLogger).
        mode_label: Human-readable description of the current mode (for the header).
    """

    def __init__(
        self,
        cfg: dict,
        env,
        agent: Agent,
        loggers: Sequence[Logger] = (),
        mode_label: str = "",
    ):
        self.cfg        = cfg
        self.env        = env
        self.agent      = agent
        self.loggers    = list(loggers)
        self.mode_label = mode_label

        self.obs_cfg          = cfg.get("observation", {})
        self.n_episodes: int  = cfg["environment"]["n_episodes"]
        self.log_every: int   = cfg["environment"].get("log_interval", 25)
        self.checkpoint_every = cfg["environment"].get("checkpoint_interval", 0)
        self.save_path        = cfg.get("save_path")

        mode     = cfg["environment"].get("mode", "discrete")
        mode_cfg = cfg.get(mode, {})
        self._algorithm = mode_cfg.get("algorithm", "q_learning")
        self._policy    = mode_cfg.get("policy", "epsilon_greedy")

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        self._print_header()

        for ep in range(self.n_episodes):
            reward = run_episode(
                self.env, self.agent, self.obs_cfg, training=True, seed=ep
            )
            self.agent.episode_rewards.append(reward)
            if len(self.agent.episode_rewards) > 5_000:
                del self.agent.episode_rewards[:2_500]
            self.agent.decay_epsilon()

            td_errs = list(self.agent.td_errors)
            avg_td  = float(np.mean(td_errs[-1_000:])) if td_errs else 0.0

            for lg in self.loggers:
                lg.log(ep + 1, reward, self.agent.epsilon, avg_td)

            print(
                f"\r  ep {ep+1:4d}/{self.n_episodes}  "
                f"reward {reward:8.2f}  "
                f"ε {self.agent.epsilon:.4f}  "
                f"{_stats_suffix(self.agent)}",
                end="", flush=True,
            )

            if self.checkpoint_every and self.save_path \
                    and (ep + 1) % self.checkpoint_every == 0:
                self.agent.save(f"{self.save_path}_ep{ep+1}")

        print()
        if self.save_path:
            self.agent.save(self.save_path)

        for lg in self.loggers:
            lg.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _print_header(self) -> None:
        print(f"{'─'*62}")
        print(f"  Mode      : {self.mode_label}")
        print(f"  Algorithm : {self._algorithm}")
        print(f"  Policy    : {self._policy}")
        if not isinstance(self.agent, DQNAgent):
            print(f"  Obs mode  : {self.obs_cfg.get('method', 'feature')}")
        else:
            print(f"  Device    : {self.agent.device}")
        print(f"  Episodes  : {self.n_episodes}")
        if self.checkpoint_every:
            print(f"  Checkpoint: every {self.checkpoint_every} episodes")
        print(f"{'─'*62}")
