import numpy as np
import gymnasium as gym
from collections import deque

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
    # Replicates Gymnasium CarRacing-v3 Discrete(5) actions via continuous mode.
    # This bypasses a Gymnasium 1.0 bug where discrete mode applies .astype(float64)
    # then fails Discrete(5).contains() check.
    # Actions: [do nothing, left, right, gas, brake]
    "car_racing_discrete": [
        [ 0.0,  0.0,  0.0],   # 0: do nothing
        [-0.6,  0.0,  0.0],   # 1: steer left
        [ 0.6,  0.0,  0.0],   # 2: steer right
        [ 0.0,  0.2,  0.0],   # 3: gas
        [ 0.0,  0.0,  0.8],   # 4: brake
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

class DiscreteInt64Wrapper(gym.Wrapper):
    """Gymnasium 1.0 CarRacing-v3 calls action.astype() unconditionally.

    Python `int` (returned by argmax/sample casts) doesn't have .astype(),
    but np.int64 does. This wrapper ensures discrete actions are np.int64.
    """

    def step(self, action):
        return self.env.step(np.int64(action))


class CarRacingPreprocessor(gym.Wrapper):
    """Preprocessing wrapper for DQN on CarRacing.

    Applies in order:
      1. Grayscale conversion  (96, 96, 3) -> (96, 96)
      2. Crop bottom 12 rows   (96, 96)    -> (84, 96)   removes dashboard
      3. Crop 6px each side     (84, 96)    -> (84, 84)   square frame
      4. Frame stacking         (84, 84)    -> (84, 84, N) last N frames
      5. Action repeat          repeat each action for K env steps

    No external dependencies — pure numpy.

    Output observation shape: (84, 84, frame_stack) dtype uint8
    """

    def __init__(self, env: gym.Env, frame_stack: int = 4,
                 action_repeat: int = 4):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self._frames: deque = deque(maxlen=frame_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, frame_stack),
            dtype=np.uint8,
        )

    @staticmethod
    def _process_frame(frame: np.ndarray) -> np.ndarray:
        """RGB (96,96,3) -> grayscale, cropped (84,84) uint8."""
        # Luminance-weighted grayscale
        gray = (0.299 * frame[:, :, 0]
                + 0.587 * frame[:, :, 1]
                + 0.114 * frame[:, :, 2]).astype(np.uint8)
        # Crop: remove 12-row dashboard at bottom, 6px from each side
        return gray[:84, 6:90]

    def _get_obs(self) -> np.ndarray:
        """Stack buffered frames along the last axis -> (84, 84, N)."""
        return np.stack(self._frames, axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._process_frame(obs)
        # Fill the buffer with copies of the first frame
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        self._frames.append(self._process_frame(obs))
        return self._get_obs(), total_reward, terminated, truncated, info


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

        # Apply DQN frame preprocessing (grayscale, crop, stack, repeat)
        fstack  = mode_cfg.get("frame_stack", 4)
        arepeat = mode_cfg.get("action_repeat", 4)
        env = CarRacingPreprocessor(env, frame_stack=fstack,
                                    action_repeat=arepeat)
        label += f"  stack={fstack} repeat={arepeat}"

    elif mode == "continuous":
        mode_cfg = cfg["continuous"]
        aset     = mode_cfg.get("action_set", "standard")
        raw_env  = gym.make(env_name, render_mode=render, continuous=True, **base_kw)
        env      = DiscretizedEnv(raw_env, aset)
        label    = f"tabular / continuous -> '{aset}' ({env.action_space.n} actions)"

    else:
        mode_cfg = cfg["discrete"]
        # Use continuous=True + DiscretizedEnv to work around a Gymnasium 1.0 bug
        # where discrete mode calls action.astype(float64) then fails contains() check.
        raw_env  = gym.make(env_name, render_mode=render, continuous=True, **base_kw)
        env      = DiscretizedEnv(raw_env, "car_racing_discrete")
        label    = "tabular / discrete -> 5 actions"

    return env, mode_cfg, label
