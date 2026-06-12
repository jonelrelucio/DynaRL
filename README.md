# DynaRL — Reinforcement Learning on CarRacing-v3

A modular RL framework for the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) Gymnasium environment, supporting tabular and deep Q-learning algorithms.

## Algorithms

| Algorithm | Type | Config `mode` |
|-----------|------|--------------|
| Q-Learning | Tabular | `discrete` / `continuous` |
| Double Q-Learning | Tabular | `discrete` / `continuous` |
| SARSA | Tabular (on-policy) | `discrete` / `continuous` |
| Expected SARSA | Tabular | `discrete` / `continuous` |
| DQN (CNN) | Deep | `dqn` |
| Double DQN (CNN) | Deep | `dqn` |

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py                        # uses config.yaml
python train.py --config my_cfg.yaml   # custom config
```

Training logs are written to `<save_path>_log.csv` and periodic checkpoints are saved every `checkpoint_interval` episodes (configurable in `config.yaml`).

## Evaluation

```bash
python evaluate.py                           # 5 episodes, rendered
python evaluate.py --episodes 10 --no-render
```

## Configuration

Edit `config.yaml` to switch algorithms, tune hyperparameters, and control training:

```yaml
environment:
  mode: "dqn"              # discrete | continuous | dqn
  n_episodes: 3000
  checkpoint_interval: 500 # 0 = disabled
  domain_randomize: false
  lap_complete_percent: 0.95

seed: 42

dqn:
  algorithm: "double_dqn"  # dqn | double_dqn
  use_continuous: true
  action_set: "smooth"
  hyperparameters:
    learning_rate: 0.0001
    replay_buffer_size: 50000
    ...
```

## Action Sets (continuous mode)

| Name | Actions | Description |
|------|---------|-------------|
| `standard` | 7 | Balanced steering + throttle |
| `smooth` | 12 | Fine-grained control |
| `aggressive` | 9 | High-speed / braking focused |

## Project Structure

```
agents/
├── base.py          Agent ABC + BaseAgent (shared tabular state)
├── tabular.py       QLearning, DoubleQ, SARSA, ExpectedSARSA
├── dqn.py           DQNAgent, CNN, ReplayBuffer
└── __init__.py      AGENT_REGISTRY + make_agent factory

envs/
├── wrappers.py      DiscretizedEnv, ACTION_SETS, build_env
└── __init__.py

utils/
├── preprocessing.py extract_features, preprocess, to_tuple
├── logging.py       Logger ABC, CSVLogger, ConsoleLogger, CompositeLogger
└── __init__.py

trainer.py           Trainer class + run_episode
train.py             Entry point (argparse → Trainer)
evaluate.py          Load checkpoint and render evaluation episodes
config.yaml          All hyperparameters and training settings
requirements.txt     Pinned dependencies
```

