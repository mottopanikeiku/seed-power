# Batch environments

I adapt `src/seed_power/vector_env.py` from
[control-clock](https://github.com/mottopanikeiku/control-clock/blob/ee9c303de8e7698ca25569fc585e173e3e172bb8/control_clock/vector_env.py)
(MIT, copyright 2026 Alp Cetin). It batches the default CartPole-v1 and
Acrobot-v1 dynamics from Gymnasium 1.2.1. Both training and evaluation use
these batches; `GymnasiumBatch` is the slow independent reference in tests.

## Interface and episode handling

`make_vector(task, num_envs)` exposes `num_envs`, `observation_size`, and
`action_size`, plus:

- `reset(seeds)` takes a list containing one nonnegative Python integer per slot.
- `step(actions, active=None)` takes integer actions of shape `(num_envs,)` and
  returns `(observations, rewards, terminated, truncated)`. Observations have
  shape `(num_envs, observation_size)` and dtype `float32`; flags are boolean.
- `reset_done(mask, seeds)` takes a boolean mask of shape `(num_envs,)` and
  **one seed per selected slot**, in ascending slot-index order. It returns the
  full updated observation batch. Unselected state and elapsed counters are
  unchanged. An empty selection accepts `[]` and changes nothing. Selected
  slots need not already be done, allowing explicit episode restarts.
- `close()` releases Gymnasium environments and prevents subsequent use.

There is **no automatic reset**. The observation returned for a terminal or
truncated transition is the final observation, not the next episode's start.
Returned observation arrays are snapshots, so a subsequent partial reset does
not overwrite observations retained by a rollout buffer. Training must reset
finished slots separately. Stepping a finished active slot raises an error
rather than using Gymnasium's undefined post-termination behavior.

Policy search can supply a boolean `active` mask of shape `(num_envs,)`.
Inactive slots retain both their observations and elapsed counters, receive
reward zero and return false termination/truncation flags for that call. Their
previous done status is retained internally. Completed candidates wait while
other candidates finish. Omitting the mask, or supplying all true, steps every slot.

Episode truncation is `elapsed_steps >= 500`. Termination and truncation are
independent: both may be true on the same transition. Partial resets clear only
the selected counters and done status. Runtime validation rejects wrong action
shapes, noninteger/out-of-range actions, invalid mask shapes/dtypes and seed
lists with incorrect lengths or invalid elements.

## Numerical contract

Each classic-control reset draws four uniforms from a newly seeded
`np.random.default_rng(seed)`, matching Gymnasium's PCG64 generator. CartPole
uses reset bounds `[-0.05, 0.05]`, its default explicit Euler integrator and
float64 internal dynamics. Its default reward is one, including the transition
that terminates the episode.

Acrobot uses reset bounds `[-0.1, 0.1]`. An important upstream detail: Gymnasium
rounds **the initial internal state to float32**. This implementation retains
that rounding, then promotes the state to float64. Its reset observation also
uses float32 trigonometry, exactly as upstream; later observations are rounded
from float64 dynamics. Each action applies torque `-1`, `0` or `1`; the default
"book" equations use a single RK4 step of duration `0.2`. After integration,
angles are wrapped into inclusive `[-pi, pi]` (both endpoints are retained),
and velocities are clipped to `[-4*pi, 4*pi]` and `[-9*pi, 9*pi]`. Reward is
zero only when `-cos(theta1)-cos(theta1+theta2) > 1`, otherwise minus one.
The nondefault NIPS equations, torque noise and custom reset bounds are not
supported by this API.

The fixed observation and internal-state comparison is `numpy.testing.assert_allclose` with
**`atol=2e-6`, `rtol=2e-7`**. This permits float32 output rounding and small
scalar-versus-array ufunc differences without feeding rounded observations
back into the integrator. The maximum permitted discrepancy is
`2e-6 + 2e-7 * abs(reference_value)`; at Acrobot's largest velocity it is
under `8e-6`. Rewards, termination and truncation flags are compared **exactly**;
there is no tolerance for decisions or episode lengths.

`tests/test_training.py` compares seeded batch transitions with independent
Gymnasium environments, including masks and resets. Rewards and episode flags
must match exactly. Float observations use the stated tolerance. The reference
does not share this repository's physics implementation.

## Sources and licenses

Dynamics, integration order, reset behavior and episode semantics are adapted
from the following pinned primary sources:

- [Gymnasium 1.2.1 CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/cartpole.py),
  originally based on Rich Sutton's [pole.c](https://perma.cc/C9ZM-652R).
- [Gymnasium 1.2.1 Acrobot](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/acrobot.py),
  including its RK4, inclusive wrap, velocity bounds and default book equations.
- [Gymnasium 1.2.1 seeding](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/utils/seeding.py)
  and [task registrations](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/__init__.py).

Gymnasium is MIT-licensed. Its Acrobot source additionally credits the RLPy
implementation under BSD-3-Clause: Alborz Geramifard, Robert H. Klein,
Christoph Dann, William Dabney and Jonathan P. How; author Christoph Dann.
Both notices are retained below. The batch layout and explicit-reset interface
come from control-clock; I reuse them rather than claiming new dynamics.

### Gymnasium MIT notice

[Upstream license](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/LICENSE):

The MIT License

Copyright (c) 2016 OpenAI  
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

### Acrobot / RLPy BSD notice

[Upstream license](https://github.com/rlpy/rlpy/blob/master/LICENSE.txt):

Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, Christoph Dann,
William Dabney and Jonathan P. How  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of ACL nor the names of its contributors may be used to
endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
