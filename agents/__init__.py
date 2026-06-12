from agents.base import Agent, BaseAgent
from agents.tabular import (
    QLearningAgent,
    DoubleQLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
)
# DQNAgent is imported lazily inside make_agent (and on explicit request)
# to avoid triggering PyTorch/CUDA initialisation in tabular worker processes.

__all__ = [
    "Agent", "BaseAgent",
    "QLearningAgent", "DoubleQLearningAgent", "SARSAAgent", "ExpectedSARSAAgent",
    "DQNAgent", "CNN", "ReplayBuffer",
    "make_agent",
]


def __getattr__(name: str):
    """Lazy-load DQN symbols so they are still importable as
    ``from agents import DQNAgent`` without eagerly importing torch."""
    if name in ("DQNAgent", "CNN", "ReplayBuffer"):
        from agents.dqn import DQNAgent, CNN, ReplayBuffer  # noqa: F811
        globals()["DQNAgent"] = DQNAgent
        globals()["CNN"] = CNN
        globals()["ReplayBuffer"] = ReplayBuffer
        return globals()[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")

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

    if name in ("dqn", "double_dqn"):
        from agents.dqn import DQNAgent  # lazy — only loaded for DQN configs
        return DQNAgent(hp, pol, env, double=(name == "double_dqn"))
    if name in AGENT_REGISTRY:
        return AGENT_REGISTRY[name](hp=hp, policy=pol, env=env)

    raise ValueError(
        f"Unknown algorithm '{name}'. "
        f"Available: {list(AGENT_REGISTRY) + ['dqn', 'double_dqn']}"
    )
