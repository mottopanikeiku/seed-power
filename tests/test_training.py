import numpy as np
import pytest

from seed_power.training import ars_update, cem_update, train_batch
from seed_power.vector_env import GymnasiumBatch, make_vector


@pytest.mark.parametrize("task", ["CartPole-v1", "Acrobot-v1"])
def test_batch_matches_gymnasium_with_independent_masks(task):
    batch, reference = make_vector(task, 4), GymnasiumBatch(task, 4)
    try:
        seeds = [14, 27, 39, 51]
        np.testing.assert_allclose(batch.reset(seeds), reference.reset(seeds), atol=2e-6, rtol=2e-7)
        rng = np.random.default_rng(19)
        active = np.ones(4, dtype=bool)
        for step in range(80):
            mask = active & np.asarray([True, True, step % 3 != 0, True])
            actions = rng.integers(batch.action_size, size=4)
            got = batch.step(actions, active=mask)
            expected = reference.step(actions, active=mask)
            np.testing.assert_allclose(got[0], expected[0], atol=2e-6, rtol=2e-7)
            for actual, wanted in zip(got[1:], expected[1:], strict=True):
                np.testing.assert_array_equal(actual, wanted)
            active &= ~(got[2] | got[3])
            if step == 40:
                reset_mask = np.array([True, False, True, False])
                np.testing.assert_allclose(
                    batch.reset_done(reset_mask, [100, 200]),
                    reference.reset_done(reset_mask, [100, 200]), atol=2e-6, rtol=2e-7,
                )
                active[reset_mask] = True
    finally:
        batch.close()
        reference.close()


def test_updates_match_independent_scalar_references():
    rng = np.random.default_rng(12)
    weights = rng.normal(size=(3, 2, 5))
    directions = rng.normal(size=(3, 8, 2, 5))
    positive, negative = rng.normal(size=(2, 3, 8))
    expected = []
    for seed in range(3):
        selected = sorted(range(8), key=lambda i: max(positive[seed, i], negative[seed, i]))[-4:]
        sd = np.std([value for i in selected for value in (positive[seed, i], negative[seed, i])])
        delta = sum((positive[seed, i] - negative[seed, i]) * directions[seed, i]
                    for i in selected)
        expected.append(weights[seed] + 0.2 / (4 * sd) * delta)
    np.testing.assert_allclose(ars_update(weights, directions, positive, negative, 4, 0.2), expected)
    np.testing.assert_array_equal(
        ars_update(weights, directions, np.ones((3, 8)), np.ones((3, 8)), 4, 0.2), weights
    )
    candidates = rng.normal(size=(3, 16, 2, 5))
    scores = rng.normal(size=(3, 16))
    std = np.ones_like(weights)
    got_mean, got_std = cem_update(weights, std, candidates, scores, 2)
    for seed in range(3):
        selected = sorted(range(16), key=lambda i: scores[seed, i])[-2:]
        elite = np.stack([candidates[seed, i] for i in selected])
        np.testing.assert_allclose(got_mean[seed], 0.5 * weights[seed] + 0.5 * elite.mean(axis=0))
        np.testing.assert_allclose(got_std[seed], np.maximum(0.05, 0.5 + 0.5 * elite.std(axis=0)))


@pytest.mark.parametrize("method", ["ars", "cem"])
def test_training_seed_results_do_not_depend_on_batch_neighbors(method):
    config = {"method": method, "iterations": 1, "population": 8}
    together = train_batch("CartPole-v1", config, [41, 73], evaluation_episodes=2)
    separately = [train_batch("CartPole-v1", config, [seed], evaluation_episodes=2)[0]
                  for seed in [41, 73]]
    assert together == separately
    for run in together:
        assert run["score"] == np.mean(run["episode_returns"])
        assert run["training_transitions"] > 0
