import random as rnd
import numpy as np
import os
import pickle
from collections import defaultdict

from agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    def update(self, obs, action, reward, next_obs, next_action=None, done=False):
        discount = 0.0 if done else self.gamma
        td_err = reward + discount * np.max(self.q[next_obs]) - self.q[obs][action]
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

    def save(self, path: str) -> None:
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"q": dict(self.q), "q2": dict(self.q2)}, f)
        print(f"[save] {path}.pkl")

    def load(self, path: str) -> None:
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
        next_val = 0.0 if done else float(np.dot(self._probs(next_obs), self.q[next_obs]))
        td_err = reward + self.gamma * next_val - self.q[obs][action]
        self.q[obs][action] += self.lr * td_err
        self.counts[obs][action] += 1
        self.td_errors.append(abs(td_err))
