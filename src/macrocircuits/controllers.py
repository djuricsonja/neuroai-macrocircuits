"""The steering controllers a run may plug into NCAP's turn inputs, and the registry
both trainers pick one from.

NCAP's circuit swims, but it cannot sense food or obstacles: the tasks that add a
`to_target` / `to_obstacle` vector to the observation (see `envs.TASKS`) leave that
vector unused unless something turns it into the circuit's `right_control` /
`left_control` signals. A run chooses that something with `controller=`:

| `controller=` | what steers | learned? |
|---|---|---|
| `None` | nothing -- the circuit swims straight, only its reward changes | -- |
| `'foraging'` / `'obstacle_avoidance'` | a fixed reflex (`reflex_steering`) | no |
| `'foraging_learnable'` | the foraging reflex's formula, gains learned (`LearnableForagingReflex`) | partly |
| `'mlp_foraging'` / `'mlp_obstacle_avoidance'` | a small MLP (`MLPController`) | yes |

That is the comparison the tasks exist for: how much of the steering has to be learned
once the swimming itself is given by the architecture.

All of these are the same shape to the rest of the code -- a callable
`controller(observations) -> (right, left, speed)`, each `(..., 1)` in `[0, 1]`, or
`None`. The fixed reflexes are plain closures; `LearnableForagingReflex` and
`MLPController` are `nn.Module`s, so assigning either to `SwimmerActor.controller` (RL)
or `NCAPSwimmerPolicy.controller` (ES) registers it as a submodule and its parameters
are trained/evolved along with the circuit's own. Nothing else has to know which kind
it got.
"""

import torch
import torch.nn as nn

from macrocircuits.envs import TASKS
from macrocircuits.reflex_steering import (
    make_foraging_reflex,
    make_foraging_reflex_adaptive,
    make_foraging_reflex_learnable,
    make_obstacle_avoidance_reflex,
)


# ==================================================================================================
# Learned controllers.

def foraging_state(observations, n_joints):
    """Joint angles plus the head-egocentric [forward, lateral] vector to the food.

    Assumes observation layout: joints, to_target, body_velocities -- i.e. a task with
    enable_foraging (or enable_single_target) on and enable_obstacles off.
    """
    joints = observations[..., :n_joints]
    to_target = observations[..., n_joints:n_joints + 2]
    return torch.cat((joints, to_target), dim=-1)


def obstacle_state(observations, n_joints):
    """Joint angles plus the head-egocentric [forward, lateral] vector to the nearest
    obstacle.

    Assumes observation layout: joints, to_obstacle, body_velocities -- i.e. a task with
    enable_obstacles on and enable_foraging off.
    """
    joints = observations[..., :n_joints]
    to_obstacle = observations[..., n_joints:n_joints + 2]
    return torch.cat((joints, to_obstacle), dim=-1)


class MLPController(nn.Module):
    """Learns the sensed-vector -> steering-command mapping the reflexes hand-derive.

    `state_fn` slices the inputs out of the raw observation (which is why the controller
    needs `n_joints`), and the head outputs `right`, `left`, `speed`, squashed to [0, 1]
    to match the range the circuit's turn inputs expect.

    Note that `speed` only does anything when the circuit was built with
    `include_speed_control=True` (via a run's `swimmer_kwargs`); otherwise
    `SwimmerModule` ignores the signal and that head simply gets no gradient.
    """

    def __init__(self, n_joints, state_fn, hidden_size=16):
        super().__init__()
        self.n_joints = n_joints
        self.state_fn = state_fn
        self.net = nn.Sequential(
            nn.Linear(n_joints + 2, hidden_size),  # joints + [forward, lateral]
            nn.Tanh(),
            nn.Linear(hidden_size, 3),  # right, left, speed (pre-activation)
        )

    def forward(self, observations):
        out = torch.sigmoid(self.net(self.state_fn(observations, self.n_joints)))
        right, left, speed = out.split(1, dim=-1)  # each keeps shape (..., 1)
        return right, left, speed


def make_foraging_mlp(n_joints, hidden_size=16):
    """Learned counterpart of `make_foraging_reflex`: steer from the vector to the food."""
    return MLPController(n_joints, foraging_state, hidden_size=hidden_size)


def make_obstacle_avoidance_mlp(n_joints, hidden_size=16):
    """Learned counterpart of `make_obstacle_avoidance_reflex`: steer from the vector to
    the nearest obstacle."""
    return MLPController(n_joints, obstacle_state, hidden_size=hidden_size)


# ==================================================================================================
# The registry both trainers select from.

# Each controller a run may name, mapped onto its factory and the tasks whose observation
# layout that factory assumes (see envs.TASKS). The factory is held as a *name* so
# training.run_config can spell it into the agent source string it eval's;
# make_controller resolves it to the real thing for callers that just want the object.
# The reflex and MLP entries deliberately cover the same tasks, so the three controller
# choices are comparable on one environment.
CONTROLLERS = {
    'foraging': ('make_foraging_reflex', ('foraging', 'swim_to_ball')),
    'foraging_learnable': ('make_foraging_reflex_learnable', ('foraging', 'swim_to_ball')),
    'foraging_adaptive': ('make_foraging_reflex_adaptive', ('foraging', 'swim_to_ball')),
    'obstacle_avoidance': ('make_obstacle_avoidance_reflex', ('evasion',)),
    'mlp_foraging': ('make_foraging_mlp', ('foraging', 'swim_to_ball')),
    'mlp_obstacle_avoidance': ('make_obstacle_avoidance_mlp', ('evasion',)),
}


def check_controller(controller, network, task):
    """Raise unless `controller` can actually steer this (network, task) combination.

    Called by both trainers before anything is built, so a mismatch fails with an
    explanation rather than as a bare AssertionError inside SwimmerModule (turn control
    on with no signal to feed it) or as a silently useless controller reading whichever
    numbers happen to sit where its vector should be.
    """
    if controller is None:
        return
    if controller not in CONTROLLERS:
        raise ValueError(
            f'controller must be None or one of {sorted(CONTROLLERS)}, got {controller!r}'
        )
    if network != 'ncap':
        raise ValueError(
            f"controller={controller!r} steers NCAP's turn inputs, which the {network!r} "
            f"baseline does not have; drop it or use network='ncap'."
        )
    tasks = CONTROLLERS[controller][1]
    if task not in tasks:
        raise ValueError(
            f'controller={controller!r} reads the {TASKS[tasks[0]]!r} vector that '
            f'{sorted(tasks)} add to the observation, but this run is on task={task!r}.'
        )


def make_controller(controller, n_joints):
    """Build the named controller for an `n_joints` body; None passes through as None."""
    if controller is None:
        return None
    if controller not in CONTROLLERS:
        raise ValueError(
            f'controller must be None or one of {sorted(CONTROLLERS)}, got {controller!r}'
        )
    return globals()[CONTROLLERS[controller][0]](n_joints)
