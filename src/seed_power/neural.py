"""Fixed-transition neural PPO, adapted from control-clock (MIT, Alp Cetin).

Source: control_clock/jax_ppo.py at b090dad8a3b53f937782676c6033ba3557cedfa1:
https://github.com/mottopanikeiku/control-clock/blob/
b090dad8a3b53f937782676c6033ba3557cedfa1/control_clock/jax_ppo.py
Inspired by Chris Lu's PureJaxRL (2023), Apache-2.0:
https://github.com/luchris429/purejaxrl (third_party/PureJaxRL-LICENSE).
I changed the implementation to train independent seeds in one compiled vmap,
use configurable fixed budgets, and evaluate once in Gymnax on common resets.
Evaluation episodes describe each policy; they are not training replicates.
"""

from functools import lru_cache
from numbers import Integral
from typing import NamedTuple

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


DEFAULTS = {
    "method": "ppo",
    "population": 32,  # Unused; shared experiment job planning expects this field.
    "num_envs": 16,
    "num_steps": 64,
    "minibatches": 4,
    "epochs": 4,
    "learning_rate": 0.0003,
    "entropy_coefficient": 0.01,
    "hidden_size": 32,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coefficient": 0.5,
    "max_gradient_norm": 0.5,
}
TASK_STEPS = {"CartPole-v1": 32768, "Acrobot-v1": 65536}


class ActorCritic(nn.Module):
    actions: int
    hidden_size: int

    @nn.compact
    def __call__(self, observations):
        outputs = []
        for name, size, scale in (("actor", self.actions, 0.01), ("critic", 1, 1.0)):
            x = observations
            for index in range(2):
                x = nn.tanh(
                    nn.Dense(
                        self.hidden_size,
                        kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        name=f"{name}_{index}",
                    )(x)
                )
            outputs.append(
                nn.Dense(
                    size, kernel_init=nn.initializers.orthogonal(scale), name=f"{name}_out"
                )(x)
            )
        return outputs[0], outputs[1].squeeze(-1)


class Transition(NamedTuple):
    observation: jax.Array
    action: jax.Array
    log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def log_probability(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]


def advantages(trajectory, last_value, gamma, gae_lambda):
    """Reverse-time GAE; terminal transitions do not bootstrap across auto-reset."""
    def step(carry, transition):
        previous, next_value = carry
        mask = 1.0 - transition.done.astype(jnp.float32)
        delta = transition.reward + gamma * next_value * mask - transition.value
        advantage = delta + gamma * gae_lambda * mask * previous
        return (advantage, transition.value), advantage

    _, result = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value), trajectory, reverse=True
    )
    return jax.lax.stop_gradient(result), jax.lax.stop_gradient(result + trajectory.value)


def ppo_loss(logits, value, transitions, advantage, target, clip_epsilon,
             entropy_coefficient, value_coefficient):
    """Clipped policy/value loss, normalized within one seed's minibatch only."""
    ratio = jnp.exp(log_probability(logits, transitions.action) - transitions.log_probability)
    normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor = -jnp.minimum(
        ratio * normalized,
        jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * normalized,
    ).mean()
    clipped = transitions.value + jnp.clip(value - transitions.value, -clip_epsilon, clip_epsilon)
    critic = 0.5 * jnp.maximum((value - target) ** 2, (clipped - target) ** 2).mean()
    log_probs = jax.nn.log_softmax(logits)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    return actor + value_coefficient * critic - entropy_coefficient * entropy


def evaluate_policy(network, parameters, env, env_params, evaluation_keys, max_steps=500):
    """Deterministic argmax, counting rewards through first done, including done.

    Gymnax steps automatically reset finished environments. The active mask is
    permanent, so neither that reset nor later episodes can contribute rewards.
    Reset and step randomness depend only on the fixed evaluation seed, not on
    the policy's training seed or its batch position.
    """
    observations, states = jax.vmap(env.reset, in_axes=(0, None))(evaluation_keys, env_params)
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    initial = (
        jnp.array(0), states, observations, evaluation_keys,
        jnp.ones(evaluation_keys.shape[0], dtype=jnp.bool_),
        jnp.zeros(evaluation_keys.shape[0], dtype=jnp.float32),
    )

    def continuing(carry):
        steps, _, _, _, active, _ = carry
        return (steps < max_steps) & active.any()

    def step(carry):
        steps, states, observations, keys, active, returns = carry
        split_keys = jax.vmap(jax.random.split)(keys)
        keys, step_keys = split_keys[:, 0], split_keys[:, 1]
        logits, _ = network.apply(parameters, observations)
        actions = jnp.argmax(logits, axis=-1)
        observations, states, rewards, dones, _ = step_env(
            step_keys, states, actions, env_params
        )
        returns = returns + jnp.where(active, rewards, 0.0)
        active = active & ~dones
        return steps + 1, states, observations, keys, active, returns

    return jax.lax.while_loop(continuing, step, initial)[-1]


@lru_cache(maxsize=32)
def _compiled_training(task, settings):
    """Cache compiled train/evaluate closures across batches with equal settings."""
    config = dict(settings)
    env, env_params = gymnax.make(task)
    env_params = env_params.replace(max_steps_in_episode=500)
    network = ActorCritic(env.num_actions, config["hidden_size"])
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    rollout_size = config["num_envs"] * config["num_steps"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["max_gradient_norm"]),
        optax.adam(config["learning_rate"], eps=1e-5),
    )

    def single_seed(seed, evaluation_keys):
        key, model_key, env_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        parameters = network.init(model_key, jnp.zeros((1, env.obs_shape[0])))
        train = TrainState.create(apply_fn=network.apply, params=parameters, tx=optimizer)
        observations, states = reset(jax.random.split(env_key, config["num_envs"]), env_params)

        def update(runner, _):
            def collect(carry, _):
                train, states, observations, key = carry
                key, action_key, env_key = jax.random.split(key, 3)
                logits, values = network.apply(train.params, observations)
                actions = jax.random.categorical(action_key, logits)
                next_obs, states, rewards, dones, _ = step_env(
                    jax.random.split(env_key, config["num_envs"]), states, actions, env_params
                )
                transition = Transition(
                    observations, actions, log_probability(logits, actions), values, rewards, dones
                )
                return (train, states, next_obs, key), transition

            runner, trajectory = jax.lax.scan(
                collect, runner, None, length=config["num_steps"]
            )
            train, states, observations, key = runner
            _, last_value = network.apply(train.params, observations)
            gae, targets = advantages(trajectory, last_value, config["gamma"], config["gae_lambda"])
            batch = jax.tree.map(
                lambda x: x.reshape((rollout_size,) + x.shape[2:]), (trajectory, gae, targets)
            )

            def epoch(carry, _):
                train, key = carry
                key, shuffle_key = jax.random.split(key)
                order = jax.random.permutation(shuffle_key, rollout_size)
                minibatches = jax.tree.map(
                    lambda x: x[order].reshape((config["minibatches"], -1) + x.shape[1:]), batch
                )

                def optimize(train, minibatch):
                    transitions, advantage, target = minibatch

                    def loss(parameters):
                        logits, values = network.apply(parameters, transitions.observation)
                        return ppo_loss(
                            logits, values, transitions, advantage, target,
                            config["clip_epsilon"], config["entropy_coefficient"],
                            config["value_coefficient"],
                        )

                    gradients = jax.grad(loss)(train.params)
                    return train.apply_gradients(grads=gradients), None

                train, _ = jax.lax.scan(optimize, train, minibatches)
                return (train, key), None

            (train, key), _ = jax.lax.scan(epoch, (train, key), None, length=config["epochs"])
            return (train, states, observations, key), None

        runner, _ = jax.lax.scan(
            update, (train, states, observations, key), None,
            length=config["total_steps"] // rollout_size,
        )
        return evaluate_policy(network, runner[0].params, env, env_params, evaluation_keys)

    return jax.jit(jax.vmap(single_seed, in_axes=(0, None)))


def _configuration(task, supplied):
    if task not in TASK_STEPS:
        raise ValueError("task must be CartPole-v1 or Acrobot-v1")
    config = dict(DEFAULTS, total_steps=TASK_STEPS[task])
    config.update(supplied)
    if config["method"] != "ppo":
        raise ValueError("method must be ppo")
    for name in ("num_envs", "num_steps", "minibatches", "epochs", "hidden_size", "total_steps"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        config[name] = int(value)
    rollout_size = config["num_envs"] * config["num_steps"]
    if config["total_steps"] % rollout_size:
        raise ValueError("total_steps must be a multiple of num_envs * num_steps")
    if rollout_size % config["minibatches"]:
        raise ValueError("num_envs * num_steps must be divisible by minibatches")
    for name in ("learning_rate", "entropy_coefficient", "gamma", "gae_lambda", "clip_epsilon",
                 "value_coefficient", "max_gradient_norm"):
        value = float(config[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if name in ("learning_rate", "max_gradient_norm") and value == 0:
            raise ValueError(f"{name} must be positive")
        if name in ("gamma", "gae_lambda", "clip_epsilon") and value > 1:
            raise ValueError(f"{name} must not exceed one")
        config[name] = value
    return config


def _seeds(values, name):
    result = list(values)
    if not result or any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32
        for seed in result
    ):
        raise ValueError(f"{name} must contain unsigned 32-bit integer seeds")
    return [int(seed) for seed in result]


def train_batch(task, configuration, seeds, evaluation_seeds):
    """Train all roots together; return one raw 16-episode outcome per root.

    No seed loop performs training or evaluation on the host. Each vmapped
    training body owns its model, Adam state, RNG, advantage normalization,
    and gradient clipping. Supplied configuration fields are retained in every
    record, including the unused population field used by experiment planning.
    Batch length can vary; the caller may pad to a preferred execution shape.
    """
    config = _configuration(task, configuration)
    seeds = _seeds(seeds, "training seeds")
    evaluation_seeds = _seeds(evaluation_seeds, "evaluation seeds")
    if len(evaluation_seeds) != 16 or len(set(evaluation_seeds)) != 16:
        raise ValueError("evaluation seeds must contain exactly 16 distinct fixed reset seeds")
    settings = tuple(sorted((name, config[name]) for name in DEFAULTS if name != "population"))
    settings += (("total_steps", config["total_steps"]),)
    train = _compiled_training(task, settings)
    evaluation_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(evaluation_seeds, dtype=jnp.uint32))
    returns = np.asarray(jax.device_get(train(jnp.asarray(seeds, dtype=jnp.uint32), evaluation_keys)))
    return [
        {"seed": seed, "task": task, "configuration": dict(config),
         "score": float(returns[index].mean()), "episode_returns": returns[index].tolist(),
         "evaluation_reset_seeds": list(evaluation_seeds),
         "training_transitions": config["total_steps"]}
        for index, seed in enumerate(seeds)
    ]
