from agents.base import Agent, BaseAgent
from agents.tabular import (
    QLearningAgent,
    DoubleQLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
)
from agents.dqn import DQNAgent, CNN, ReplayBuffer

__all__ = [
    "Agent", "BaseAgent",
    "QLearningAgent", "DoubleQLearningAgent", "SARSAAgent", "ExpectedSARSAAgent",
    "DQNAgent", "CNN", "ReplayBuffer",
    "make_agent",
]

# ── Agent registry ────────────────────────────────────────────────────────────
# Maps algorithm name → class. Add new agents here without touching make_agent.

AGENT_REGISTRY: dict = {
    "q_learning":       QLearningAgent,
    "double_q_learning": DoubleQLearningAgent,
    "sarsa":            SARSAAgent,
    "expected_sarsa":   ExpectedSARSAAgent,
}


def make_agent(mode_cfg: dict, env) -> Agent:
    name = mode_cfg.get("algorithm", "q_learning")
    hp   = mode_cfg["hyperparameters"]
    pol  = mode_cfg.get("policy", "epsilon_greedy")

    if name == "dqn":
        return DQNAgent(hp, pol, env, double=False)
    if name == "double_dqn":
        return DQNAgent(hp, pol, env, double=True)
    if name in AGENT_REGISTRY:
        return AGENT_REGISTRY[name](hp=hp, policy=pol, env=env)

    raise ValueError(
        f"Unknown algorithm '{name}'. "
        f"Available: {list(AGENT_REGISTRY) + ['dqn', 'double_dqn']}"
    )
