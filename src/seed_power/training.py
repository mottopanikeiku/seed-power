"""Fixed-budget affine ARS/CEM, adapted from control-clock (MIT).

Source: control_clock/search.py at ee9c303de8e7698ca25569fc585e173e3e172bb8.
Unlike that stopping-time experiment, I fix iteration counts and evaluate once.
Every batch member has its own optimizer, training resets, and evaluation resets.
"""

from __future__ import annotations

import numpy as np

from seed_power.vector_env import make_vector


def actions(weights, observations, scales):
    features = np.concatenate(
        (observations / scales, np.ones((*observations.shape[:-1], 1))), axis=-1
    )
    return np.einsum("...ao,...o->...a", weights, features).argmax(axis=-1)


def episode_returns(task, weights, reset_seeds, scales, max_steps=500):
    """One complete episode per policy; no post-terminal transitions."""
    env = make_vector(task, len(weights))
    try:
        observations = env.reset([int(seed) for seed in reset_seeds])
        active = np.ones(len(weights), dtype=bool)
        returns = np.zeros(len(weights))
        transitions = np.zeros(len(weights), dtype=np.int64)
        for _ in range(max_steps):
            if not active.any():
                break
            observations, rewards, terminated, truncated = env.step(
                actions(weights, observations, scales), active=active
            )
            returns += rewards
            transitions += active
            active &= ~(terminated | truncated)
        if active.any():
            raise RuntimeError("episode horizon ended before the environment's time limit")
        return returns, transitions
    finally:
        env.close()


def ars_update(weights, directions, positive, negative, top, learning_rate):
    """Independent ARS updates; normalization never mixes training seeds."""
    selected = np.argsort(np.maximum(positive, negative), axis=1)[:, -top:]
    positive = np.take_along_axis(positive, selected, axis=1)
    negative = np.take_along_axis(negative, selected, axis=1)
    chosen = np.take_along_axis(directions, selected[:, :, None, None], axis=1)
    score_std = np.std(np.concatenate((positive, negative), axis=1), axis=1)
    update = np.einsum("sn,snao->sao", positive - negative, chosen)
    scale = np.divide(
        learning_rate, top * score_std, out=np.zeros_like(score_std), where=score_std >= 1e-8
    )
    return weights + scale[:, None, None] * update


def cem_update(mean, std, candidates, scores, elites):
    selected = np.argsort(scores, axis=1, kind="stable")[:, -elites:]
    chosen = np.take_along_axis(candidates, selected[:, :, None, None], axis=1)
    return (
        0.5 * mean + 0.5 * chosen.mean(axis=1),
        np.maximum(0.05, 0.5 * std + 0.5 * chosen.std(axis=1)),
    )


def train_batch(task, configuration, seeds, evaluation_episodes=16):
    """Return independent per-training-seed scores; batching is only execution."""
    method = configuration["method"]
    if method not in ("ars", "cem"):
        raise ValueError("method must be ars or cem")
    population = configuration.get("population", 32)
    iterations = configuration["iterations"]
    if population < 8 or population % 8 or iterations < 1 or evaluation_episodes < 1:
        raise ValueError("invalid population, iterations or evaluation count")
    if len(seeds) == 0 or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be nonempty and unique within a batch")
    streams = [np.random.SeedSequence(int(seed)).spawn(3) for seed in seeds]
    optimizers = [np.random.default_rng(s[0]) for s in streams]
    training_rngs = [np.random.default_rng(s[1]) for s in streams]
    evaluation_rngs = [np.random.default_rng(s[2]) for s in streams]
    probe = make_vector(task, 1)
    shape = (probe.action_size, probe.observation_size + 1)
    scales = np.ones(probe.observation_size)
    if task == "Acrobot-v1":
        scales[-2:] = [4 * np.pi, 9 * np.pi]
    probe.close()
    count = len(seeds)
    weights = np.zeros((count, *shape))
    std = np.ones_like(weights)
    training_steps = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        if method == "ars":
            directions = np.stack(
                [rng.standard_normal((population // 2, *shape)) for rng in optimizers]
            )
            candidates = np.concatenate(
                (weights[:, None] + 0.5 * directions, weights[:, None] - 0.5 * directions),
                axis=1,
            )
        else:
            candidates = np.stack(
                [rng.normal(weights[i], std[i], size=(population, *shape))
                 for i, rng in enumerate(optimizers)]
            )
        # One common reset within each optimizer population; fresh at every iteration.
        resets = [int(rng.integers(0, 2**32)) for rng in training_rngs]
        scores, steps = episode_returns(
            task, candidates.reshape(-1, *shape), np.repeat(resets, population), scales
        )
        scores = scores.reshape(count, population)
        training_steps += steps.reshape(count, population).sum(axis=1)
        if method == "ars":
            weights = ars_update(
                weights, directions, scores[:, :population // 2], scores[:, population // 2:],
                population // 4, configuration.get("learning_rate", 0.2),
            )
        else:
            weights, std = cem_update(weights, std, candidates, scores, population // 8)
    reset_matrix = np.asarray(
        [rng.integers(0, 2**32, size=evaluation_episodes).tolist() for rng in evaluation_rngs]
    )
    returns, _ = episode_returns(
        task, np.repeat(weights, evaluation_episodes, axis=0), reset_matrix.ravel(), scales
    )
    returns = returns.reshape(count, evaluation_episodes)
    return [
        {"seed": int(seed), "task": task, "configuration": dict(configuration),
         "score": float(returns[i].mean()), "episode_returns": returns[i].tolist(),
         "evaluation_reset_seeds": reset_matrix[i].tolist(),
         "training_transitions": int(training_steps[i])}
        for i, seed in enumerate(seeds)
    ]
