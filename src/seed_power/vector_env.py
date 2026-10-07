"""Explicit-reset batches; classic dynamics adapted from Gymnasium 1.2.1.

Gymnasium: Copyright (c) 2016 OpenAI, (c) 2022 Farama Foundation (MIT).
Acrobot also derives from RLPy, Copyright (c) 2013 Alborz Geramifard,
Robert H. Klein, Christoph Dann, William Dabney and Jonathan P. How (BSD-3-Clause).
Source links, numerical contract and complete notices: docs/ENVIRONMENTS.md.
"""

from __future__ import annotations

import numpy as np

_TASKS = {
    "CartPole-v1": (4, 2, 500),
    "Acrobot-v1": (6, 3, 500),
}


class _Batch:
    def __init__(self, task: str, num_envs: int):
        if task not in _TASKS:
            raise ValueError(f"Unsupported task: {task!r}")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.observation_size, self.action_size, self.max_steps = _TASKS[task]
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self._done = np.zeros(num_envs, dtype=bool)
        self._ready = False
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Environment is closed")

    def _check_ready(self):
        self._check_open()
        if not self._ready:
            raise RuntimeError("Call reset before stepping or partially resetting")

    def _actions(self, actions, active):
        self._check_ready()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or actions.dtype.kind not in "iu":
            raise ValueError("actions must be an integer array of shape (num_envs,)")
        if np.any(actions < 0) or np.any(actions >= self.action_size):
            raise ValueError(f"actions must lie in [0, {self.action_size})")
        if active is not None:
            active = np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype.kind != "b":
                raise ValueError("active must be a boolean array of shape (num_envs,)")
            if np.all(active):
                active = None
        selected = slice(None) if active is None else active
        if np.any(self._done[selected]):
            raise RuntimeError("Reset finished active slots with reset_done before stepping")
        return actions, selected

    @staticmethod
    def _seeds(seeds: list[int], count: int):
        if not isinstance(seeds, list) or len(seeds) != count:
            raise ValueError(f"seeds must be a list of {count} nonnegative Python integers")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
            raise ValueError("seeds must contain nonnegative Python integers")
        return seeds

    def reset(self, seeds: list[int]) -> np.ndarray:
        self._check_open()
        seeds = self._seeds(seeds, self.num_envs)
        self._reset_slots(np.arange(self.num_envs), seeds)
        self.steps.fill(0)
        self._done.fill(False)
        self._ready = True
        return self._observation()

    def reset_done(self, mask, seeds: list[int]) -> np.ndarray:
        """Reset selected slots, with one seed per True entry in index order."""
        self._check_ready()
        mask = np.asarray(mask)
        if mask.shape != (self.num_envs,) or mask.dtype.kind != "b":
            raise ValueError("mask must be a boolean array of shape (num_envs,)")
        indices = np.flatnonzero(mask)
        seeds = self._seeds(seeds, len(indices))
        self._reset_slots(indices, seeds)
        self.steps[indices] = 0
        self._done[indices] = False
        return self._observation()

    def close(self):
        self._closed = True


class _ClassicBatch(_Batch):
    def __init__(self, task: str, num_envs: int):
        super().__init__(task, num_envs)
        self.state = np.empty((num_envs, 4), dtype=np.float64)
        # Cache observations to preserve Acrobot's float32 trigonometry on reset.
        self._obs = np.empty((num_envs, self.observation_size), dtype=np.float32)

    def _observation(self):
        return self._obs.copy()

    def step(self, actions, active=None):
        actions, selected = self._actions(actions, active)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        if actions[selected].size:
            rewards[selected], terminated[selected] = self._advance(actions[selected], selected)
            self.steps[selected] += 1
            truncated[selected] = self.steps[selected] >= self.max_steps
            self._done[selected] = terminated[selected] | truncated[selected]
        return self._observation(), rewards, terminated, truncated


class CartPoleBatch(_ClassicBatch):
    def __init__(self, num_envs: int):
        super().__init__("CartPole-v1", num_envs)

    def _reset_slots(self, indices, seeds):
        for index, seed in zip(indices, seeds, strict=True):
            self.state[index] = np.random.default_rng(seed).uniform(-0.05, 0.05, size=4)
            self._obs[index] = self.state[index]

    def _advance(self, actions, selected):
        state = self.state[selected]
        x, x_dot, theta, theta_dot = state.T
        force = np.where(actions == 1, 10.0, -10.0)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + 0.05 * np.square(theta_dot) * sintheta) / 1.1
        thetaacc = (9.8 * sintheta - costheta * temp) / (
            0.5 * (4.0 / 3.0 - 0.1 * np.square(costheta) / 1.1)
        )
        xacc = temp - 0.05 * thetaacc * costheta / 1.1
        x += 0.02 * x_dot
        x_dot += 0.02 * xacc
        theta += 0.02 * theta_dot
        theta_dot += 0.02 * thetaacc
        threshold = 12 * 2 * np.pi / 360
        terminated = (x < -2.4) | (x > 2.4) | (theta < -threshold) | (theta > threshold)
        self.state[selected] = state
        self._obs[selected] = state
        return np.ones(len(actions), dtype=np.float64), terminated


def _wrap_angles(angles):
    """Gymnasium's inclusive [-pi, pi] wrap, preserving both endpoints."""
    above = angles > np.pi
    while np.any(above):
        np.subtract(angles, 2 * np.pi, out=angles, where=above)
        above = angles > np.pi
    below = angles < -np.pi
    while np.any(below):
        np.add(angles, 2 * np.pi, out=angles, where=below)
        below = angles < -np.pi


class AcrobotBatch(_ClassicBatch):
    def __init__(self, num_envs: int):
        super().__init__("Acrobot-v1", num_envs)
        self._rk = np.empty((4, num_envs, 4), dtype=np.float64)
        self._work = np.empty_like(self.state)

    def _reset_slots(self, indices, seeds):
        for index, seed in zip(indices, seeds, strict=True):
            # Upstream rounds the INITIAL internal state, but RK4 thereafter uses float64.
            s = np.random.default_rng(seed).uniform(-0.1, 0.1, size=4).astype(np.float32)
            self.state[index] = s
            self._obs[index] = [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]]

    @staticmethod
    def _derivative(state, torque, out):
        theta1, theta2, dtheta1, dtheta2 = state.T
        # Keep Gymnasium's expression ordering with its default physical constants.
        c2, s2 = np.cos(theta2), np.sin(theta2)
        d1 = 0.25 + (1.0 + 0.25 + c2) + 1.0 + 1.0
        d2 = (0.25 + 0.5 * c2) + 1.0
        phi2 = 0.5 * 9.8 * np.cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -0.5 * dtheta2**2 * s2
            - 2 * 0.5 * dtheta2 * dtheta1 * s2
            + 1.5 * 9.8 * np.cos(theta1 - np.pi / 2)
            + phi2
        )
        ddtheta2 = (torque + d2 / d1 * phi1 - 0.5 * dtheta1**2 * s2 - phi2) / (
            0.25 + 1.0 - d2**2 / d1
        )
        out[:, 0] = dtheta1
        out[:, 1] = dtheta2
        out[:, 2] = -(d2 * ddtheta2 + phi1) / d1
        out[:, 3] = ddtheta2

    def _advance(self, actions, selected):
        state = self.state[selected]
        torque = actions.astype(np.float64) - 1.0
        k1, k2, k3, k4 = self._rk[:, : len(actions)]
        work = self._work[: len(actions)]
        self._derivative(state, torque, k1)
        np.multiply(k1, 0.1, out=work)
        work += state
        self._derivative(work, torque, k2)
        np.multiply(k2, 0.1, out=work)
        work += state
        self._derivative(work, torque, k3)
        np.multiply(k3, 0.2, out=work)
        work += state
        self._derivative(work, torque, k4)
        state += 0.2 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        _wrap_angles(state[:, :2])
        np.clip(state[:, 2], -4 * np.pi, 4 * np.pi, out=state[:, 2])
        np.clip(state[:, 3], -9 * np.pi, 9 * np.pi, out=state[:, 3])
        self.state[selected] = state
        theta1, theta2 = state[:, 0], state[:, 1]
        terminated = -np.cos(theta1) - np.cos(theta1 + theta2) > 1.0
        self._obs[selected, 0] = np.cos(theta1)
        self._obs[selected, 1] = np.sin(theta1)
        self._obs[selected, 2] = np.cos(theta2)
        self._obs[selected, 3] = np.sin(theta2)
        self._obs[selected, 4:] = state[:, 2:]
        return np.where(terminated, 0.0, -1.0), terminated


class GymnasiumBatch(_Batch):
    """Slow independent Gymnasium environments; no autoreset wrapper.

    Used for LunarLander and as the classic-control equivalence reference.
    Box2D retains its native precision; only NumPy classic dynamics use float64.
    """

    def __init__(self, task: str, num_envs: int):
        super().__init__(task, num_envs)
        import gymnasium as gym

        self.envs = []
        try:
            for _ in range(num_envs):
                self.envs.append(gym.make(task))
        except Exception:
            self.close()
            raise
        self._obs = np.empty((num_envs, self.observation_size), dtype=np.float32)

    def _reset_slots(self, indices, seeds):
        for index, seed in zip(indices, seeds, strict=True):
            self._obs[index], _ = self.envs[index].reset(seed=seed)

    def _observation(self):
        return self._obs.copy()

    def step(self, actions, active=None):
        actions, selected = self._actions(actions, active)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        indices = range(self.num_envs) if isinstance(selected, slice) else np.flatnonzero(selected)
        for index in indices:
            obs, reward, term, trunc, _ = self.envs[index].step(int(actions[index]))
            self._obs[index] = obs
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
        self.steps[selected] += 1
        self._done[selected] = terminated[selected] | truncated[selected]
        return self._observation(), rewards, terminated, truncated

    def close(self):
        for env in self.envs:
            env.close()
        super().close()


def make_vector(task: str, num_envs: int) -> _Batch:
    """Construct a task batch. reset_done consumes seeds only for selected slots."""
    if task == "CartPole-v1":
        return CartPoleBatch(num_envs)
    if task == "Acrobot-v1":
        return AcrobotBatch(num_envs)
    return GymnasiumBatch(task, num_envs)
