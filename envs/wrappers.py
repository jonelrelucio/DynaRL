import numpy as np
import gymnasium as gym

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
            raise ValueError(
                f"Unknown action set '{action_set_name}'. "
                f"Choose from: {list(ACTION_SETS)}"
            )
        self._actions = np.array(ACTION_SETS[action_set_name], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        self._action_set_name = action_set_name

    def step(self, action: int):
        return self.env.step(self._actions[action])


def build_env(cfg: dict):
    """Construct the gym environment and return (env, mode_cfg, label)."""
    mode     = cfg["environment"].get("mode", "discrete")
    env_name = cfg["environment"]["name"]
    render   = cfg["environment"].get("render_mode", "rgb_array")
    base_kw  = dict(
        lap_complete_percent=cfg["environment"].get("lap_complete_percent", 0.95),
        domain_randomize=cfg["environment"].get("domain_randomize", False),
    )

    if mode == "dqn":
        mode_cfg       = cfg["dqn"]
        use_continuous = mode_cfg.get("use_continuous", False)
        raw_env        = gym.make(env_name, render_mode=render,
                                  continuous=use_continuous, **base_kw)
        if use_continuous:
            aset  = mode_cfg.get("action_set", "smooth")
            env   = DiscretizedEnv(raw_env, aset)
            label = f"dqn (CNN) / continuous -> '{aset}' ({env.action_space.n} actions)"
        else:
            env   = raw_env
            label = "dqn (CNN) / discrete -> 5 actions"

    elif mode == "continuous":
        mode_cfg = cfg["continuous"]
        aset     = mode_cfg.get("action_set", "standard")
        raw_env  = gym.make(env_name, render_mode=render, continuous=True, **base_kw)
        env      = DiscretizedEnv(raw_env, aset)
        label    = f"tabular / continuous -> '{aset}' ({env.action_space.n} actions)"

    else:
        mode_cfg = cfg["discrete"]
        env      = gym.make(env_name, render_mode=render, continuous=False, **base_kw)
        label    = "tabular / discrete -> 5 actions"

    return env, mode_cfg, label
