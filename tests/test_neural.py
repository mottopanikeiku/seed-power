"""Optional CPU correctness tests; install the neural extra to exercise PPO."""

import numpy as np
import pytest

neural = pytest.importorskip("seed_power.neural", exc_type=ImportError)
jax = neural.jax
jnp = neural.jnp


@pytest.fixture(autouse=True)
def cpu_execution():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def trajectory(values, rewards=None, dones=None, actions=None, log_probs=None):
    values = jnp.asarray(values, dtype=jnp.float32)
    shape = values.shape
    return neural.Transition(
        jnp.zeros((*shape, 4)),
        jnp.zeros(shape, dtype=jnp.int32) if actions is None else jnp.asarray(actions),
        jnp.zeros(shape) if log_probs is None else jnp.asarray(log_probs),
        values,
        jnp.zeros(shape) if rewards is None else jnp.asarray(rewards, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.bool_) if dones is None else jnp.asarray(dones),
    )


def test_gae_matches_scalar_reference_and_stops_at_terminal():
    values = np.array([[0.2, -0.1, 0.3], [0.5, 0.4, -0.2],
                       [0.6, 0.1, 0.8], [-0.1, 0.7, 0.9]], dtype=np.float32)
    rewards = np.array([[1, -1, 2], [2, 0, 1], [3, -2, 0], [4, 1, -1]], dtype=np.float32)
    dones = np.array([[False, False, True], [True, False, False],
                      [False, True, False], [False, False, True]])
    last_value = np.array([0.8, -0.4, 100.0], dtype=np.float32)
    gamma, gae_lambda = 0.91, 0.83
    expected = np.zeros_like(values)
    for environment in range(values.shape[1]):
        previous, next_value = 0.0, last_value[environment]
        for time in range(len(values) - 1, -1, -1):
            mask = 1.0 - dones[time, environment]
            delta = (rewards[time, environment] + gamma * next_value * mask
                     - values[time, environment])
            previous = delta + gamma * gae_lambda * mask * previous
            expected[time, environment] = previous
            next_value = values[time, environment]
    batch = trajectory(values, rewards, dones)
    actual, targets = jax.jit(neural.advantages)(batch, jnp.asarray(last_value), gamma, gae_lambda)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(targets, expected + values, rtol=2e-6, atol=2e-6)
    # A terminal's reset-state value must never influence its preceding episode.
    assert actual[1, 0] == pytest.approx(rewards[1, 0] - values[1, 0])
    assert actual[-1, 2] == pytest.approx(rewards[-1, 2] - values[-1, 2])
    gradient = jax.grad(lambda v: neural.advantages(
        batch._replace(value=v), jnp.asarray(last_value), gamma, gae_lambda
    )[1].sum())(jnp.asarray(values))
    np.testing.assert_array_equal(gradient, np.zeros_like(values))


def test_ppo_loss_matches_scalar_reference_with_both_clips_and_entropy():
    logits = np.array([[0.2, -0.5], [1.0, -0.2], [-0.1, 0.6], [0.3, 0.1]], dtype=np.float32)
    actions = np.array([0, 1, 1, 0], dtype=np.int32)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    new_log_probs = log_probs[np.arange(4), actions]
    ratios = np.array([1.8, 0.4, 1.05, 1.2], dtype=np.float32)
    old_log_probs = new_log_probs - np.log(ratios)
    old_values = np.array([0.0, 0.2, -0.5, 0.3], dtype=np.float32)
    values = np.array([0.9, -0.6, -0.3, 0.35], dtype=np.float32)
    targets = np.array([1.0, -1.0, 0.4, -0.2], dtype=np.float32)
    advantage = np.array([2.0, -2.0, 0.5, 0.0], dtype=np.float32)
    normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    clip, entropy_coefficient, value_coefficient = 0.2, 0.07, 0.6
    actor, critic, entropy = 0.0, 0.0, 0.0
    for index in range(4):
        actor -= min(ratios[index] * normalized[index],
                     min(1 + clip, max(1 - clip, ratios[index])) * normalized[index]) / 4
        clipped = old_values[index] + min(clip, max(-clip, values[index] - old_values[index]))
        critic += 0.5 * max((values[index] - targets[index]) ** 2,
                            (clipped - targets[index]) ** 2) / 4
        for action in range(2):
            entropy -= np.exp(log_probs[index, action]) * log_probs[index, action] / 4
    batch = trajectory(old_values, actions=actions, log_probs=old_log_probs)
    actual = neural.ppo_loss(
        jnp.asarray(logits), jnp.asarray(values), batch, jnp.asarray(advantage),
        jnp.asarray(targets), clip, entropy_coefficient, value_coefficient,
    )
    assert float(actual) == pytest.approx(
        actor + value_coefficient * critic - entropy_coefficient * entropy, abs=2e-6
    )


class ArgmaxPolicy:
    def apply(self, parameters, observations):
        del parameters
        # Lowest-index argmax ties must choose action zero.
        return jnp.zeros((observations.shape[0], 2)), jnp.zeros(observations.shape[0])


class AutoResetEnvironment:
    """First episode lasts 2 or 4 steps; later episodes carry huge rewards."""
    def reset(self, key, parameters):
        del parameters
        duration = jnp.where(key[1] % 2 == 0, 2, 4)
        state = (jnp.array(0), jnp.array(0), duration)
        return jnp.zeros(1), state

    def step(self, key, state, action, parameters):
        del key, parameters
        elapsed, episode, duration = state
        elapsed += 1
        done = elapsed >= duration
        reward = jnp.where(episode == 0, elapsed.astype(jnp.float32), 10000.0)
        # A nonzero action makes accidental sampling or tie-breaking visible.
        reward += 1000.0 * action
        state = (jnp.where(done, 0, elapsed), episode + done, duration)
        return jnp.zeros(1), state, reward, done, {}


def test_evaluation_masks_auto_reset_and_includes_termination_reward():
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(16, dtype=jnp.uint32))
    evaluate = jax.jit(lambda: neural.evaluate_policy(
        ArgmaxPolicy(), {}, AutoResetEnvironment(), None, keys, max_steps=500
    ))
    actual = evaluate()
    # Sum 1+2 for even seeds; sum 1+2+3+4 for odd seeds.
    np.testing.assert_array_equal(actual, np.tile([3.0, 10.0], 8))
    # A horizon before termination still includes each step up to that horizon.
    truncated = neural.evaluate_policy(
        ArgmaxPolicy(), {}, AutoResetEnvironment(), None, keys, max_steps=3
    )
    np.testing.assert_array_equal(truncated, np.tile([3.0, 6.0], 8))


@pytest.mark.parametrize("task", ["CartPole-v1", "Acrobot-v1"])
def test_tiny_training_is_independent_of_batch_neighbors_and_order(task):
    config = {
        "method": "ppo", "population": 32, "num_envs": 2, "num_steps": 8,
        "minibatches": 2, "epochs": 2, "learning_rate": 0.0004,
        "entropy_coefficient": 0.03, "hidden_size": 8, "total_steps": 32,
        "gamma": 0.97, "gae_lambda": 0.91, "clip_epsilon": 0.15,
        "value_coefficient": 0.7, "max_gradient_norm": 0.4,
        "variant_label": "tiny-test",
    }
    evaluation_seeds = list(range(10000, 10016))
    together = neural.train_batch(task, config, [41, 73], evaluation_seeds)
    reverse = neural.train_batch(task, config, [73, 41], evaluation_seeds)
    alone = neural.train_batch(task, config, [41], evaluation_seeds)[0]
    assert together == list(reversed(reverse))
    assert together[0] == alone
    for record in together:
        assert record["configuration"] == config
        assert record["training_transitions"] == 32
        assert record["evaluation_reset_seeds"] == evaluation_seeds
        assert len(record["episode_returns"]) == 16
        assert record["score"] == pytest.approx(np.mean(record["episode_returns"]), abs=1e-5)
        assert np.isfinite(record["episode_returns"]).all()
        if task == "CartPole-v1":
            assert all(1 <= value <= 500 for value in record["episode_returns"])
        else:
            assert all(-500 <= value <= 0 for value in record["episode_returns"])


def test_hidden_width_controls_all_four_trainable_hidden_layers():
    network = neural.ActorCritic(actions=3, hidden_size=7)
    params = network.init(jax.random.PRNGKey(19), jnp.zeros((2, 6)))["params"]
    for branch in ("actor", "critic"):
        assert params[f"{branch}_0"]["kernel"].shape == (6, 7)
        assert params[f"{branch}_1"]["kernel"].shape == (7, 7)
    assert params["actor_out"]["kernel"].shape == (7, 3)
    assert params["critic_out"]["kernel"].shape == (7, 1)


@pytest.mark.parametrize("override, message", [
    ({"total_steps": 1025}, "multiple"),
    ({"minibatches": 3}, "divisible"),
    ({"epochs": 0}, "positive integer"),
    ({"hidden_size": 2.5}, "positive integer"),
    ({"learning_rate": 0.0}, "positive"),
    ({"entropy_coefficient": float("nan")}, "finite"),
    ({"gamma": 1.1}, "exceed one"),
    ({"method": "ars"}, "method must be ppo"),
])
def test_invalid_configuration_rejected_before_compilation(override, message, monkeypatch):
    def unexpected_compile(*args):
        pytest.fail("invalid input reached training compilation")
    monkeypatch.setattr(neural, "_compiled_training", unexpected_compile)
    with pytest.raises(ValueError, match=message):
        neural.train_batch("CartPole-v1", override, [1], range(16))


@pytest.mark.parametrize("training_seeds, evaluation_seeds, message", [
    ([], range(16), "training seeds"),
    ([-1], range(16), "training seeds"),
    ([2**32], range(16), "training seeds"),
    ([1], range(15), "exactly 16"),
    ([1], [0] * 16, "distinct"),
    ([1], [0.5] + list(range(15)), "evaluation seeds"),
])
def test_invalid_seed_inputs_rejected_before_compilation(
    training_seeds, evaluation_seeds, message, monkeypatch
):
    def unexpected_compile(*args):
        pytest.fail("invalid input reached training compilation")
    monkeypatch.setattr(neural, "_compiled_training", unexpected_compile)
    with pytest.raises(ValueError, match=message):
        neural.train_batch("CartPole-v1", {}, training_seeds, evaluation_seeds)


def test_trained_weights_are_seed_isolated_and_configuration_changes_updates(monkeypatch):
    # Inspect real trained weights instead of scores: tiny-budget policies can
    # choose identical argmax actions even when their optimizer states differ.
    def weights_instead_of_evaluation(network, parameters, env, env_params, evaluation_keys):
        del network, env, env_params, evaluation_keys
        return jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(parameters)])

    monkeypatch.setattr(neural, "evaluate_policy", weights_instead_of_evaluation)
    config = neural._configuration("CartPole-v1", {
        "num_envs": 2, "num_steps": 8, "minibatches": 2, "epochs": 2,
        "hidden_size": 4, "total_steps": 32,
    })
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(16, dtype=jnp.uint32))

    def compiled(overrides):
        settings = dict(config, **overrides)
        settings.pop("population")
        # Avoid polluting the production closure cache with the test evaluator.
        return neural._compiled_training.__wrapped__("CartPole-v1", tuple(sorted(settings.items())))

    train = compiled({})
    together = train(jnp.array([41, 73], dtype=jnp.uint32), keys)
    alone = train(jnp.array([41], dtype=jnp.uint32), keys)
    reversed_weights = train(jnp.array([73, 41], dtype=jnp.uint32), keys)
    np.testing.assert_allclose(together[0], alone[0], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(together, reversed_weights[::-1], rtol=2e-5, atol=2e-6)
    assert not np.allclose(together[0], together[1])
    for override in ({"epochs": 1}, {"learning_rate": 0.01}, {"total_steps": 16},
                     {"entropy_coefficient": 1.0}, {"max_gradient_norm": 0.001}):
        altered = compiled(override)(jnp.array([41], dtype=jnp.uint32), keys)
        assert np.max(np.abs(np.asarray(altered[0] - alone[0]))) > 1e-7
