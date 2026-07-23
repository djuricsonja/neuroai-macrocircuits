"""The steering controllers a run may plug into NCAP's turn inputs, and the registry
both trainers pick one from.

NCAP's circuit swims, but it cannot sense food or obstacles: the tasks that add a
`to_target` / `to_obstacle` vector to the observation (see `envs.TASKS`) leave that
vector unused unless something turns it into the circuit's `right_control` /
`left_control` signals. A run chooses that something with `controller=`:

| `controller=` | what steers | learned? |
|---|---|---|
| `None` | nothing -- the circuit swims straight, only its reward changes | -- |
| `'foraging'` / `'obstacle_avoidance'` | a fixed P+D reflex (`reflex_steering`) | no |
| `'steer_to_food'` | a fixed proportional turn toward food | no |
| `'turn_left'` / `'turn_right'` | a constant turn, sensor-free (an actuator test) | no |
| `'learned_steering'` | a tiny MLP on the bearing, driving the fixed turn primitive | yes |
| `'mlp_foraging'` / `'mlp_obstacle_avoidance'` | a small MLP on joints + vector (`MLPController`) | yes |

That is the comparison the tasks exist for: how much of the steering has to be learned
once the swimming itself is given by the architecture -- with `None` as the floor it is
all measured against.

All of these are the same shape to the rest of the code -- a callable
`controller(observations) -> (right, left, speed)`, each `(..., 1)` in `[0, 1]`, or
`None`. The reflexes are plain closures; `LearnedSteering` and `MLPController` are
`nn.Module`s, so assigning either to `SwimmerActor.controller` (RL) or
`NCAPSwimmerPolicy.controller` (ES) registers it as a submodule and its parameters are
trained/evolved along with the circuit's own. Nothing else has to know which kind it got.
"""

import numpy as np
import torch
import torch.nn as nn

from macrocircuits.envs import TASKS
from macrocircuits.reflex_steering import (
    make_foraging_reflex,
    make_obstacle_avoidance_reflex,
    make_steer_to_food_reflex,
    make_turn_left_reflex,
    make_turn_right_reflex,
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


class LearnedSteering(nn.Module):
    """Learn *how* to steer toward food on top of the fixed turn primitive.

    Pipeline: the egocentric direction to the food -> a tiny MLP -> one turn command,
    handed to NCAP's left/right inputs at a fixed, pre-calibrated strength. Far smaller
    and better-posed than the from-scratch `MLPController`, which reads raw joints + raw
    vector and would have to discover the geometry, the sign convention, the turn
    strength and the decision all at once.

    Egocentric convention, MEASURED directly (see the check_convention diagnostic), not
    inherited from the older steering code which had these axes SWAPPED (the bug that
    kept the whole foraging effort from ever steering):
        lateral = to_target[0]   (+ve => food is to the worm's LEFT)
        forward = -to_target[1]  (+ve => food is AHEAD)
    Driving `left` turns the worm to its own left.

    Bounded with `tanh`, never a hard [0, 1] clamp: a clamp saturates ~95% of steps and
    starves the parameters of gradient (a failure this project already hit); tanh keeps a
    live gradient every step, and turn_strength <= the calibrated ceiling means no upper
    clamp is ever needed.

    Warm start: from random init, PPO here never even discovered the correct turn *sign*
    -- it collapsed to a constant weak turn (~7% success) while the same steering rule
    hand-set navigates at ~90%. So the net is first behaviour-cloned to reproduce that
    correct-sign hardcoded steerer, and PPO then only has to *refine* it. This is the
    "good init + learning" philosophy the NCAP paper itself uses for the circuit weights,
    not a reused hand-tuned magic number.
    """

    def __init__(self, n_joints, hidden_size=8, turn_strength=0.75,
                 warm_start=True, warm_gain=3.0, warm_steps=500):
        super().__init__()
        self.n_joints = n_joints
        self.target_slice = slice(n_joints, n_joints + 2)
        self.turn_strength = turn_strength
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),   # unit [forward, lateral] -> hidden
            nn.Tanh(),
            nn.Linear(hidden_size, 1),   # -> one turn command (pre-activation)
        )
        if warm_start:
            self._behaviour_clone(warm_gain, warm_steps)

    def _behaviour_clone(self, gain, steps, seed=0):
        """Fit the net to the correct-sign hardcoded steerer over random bearings, so RL
        starts from a working ~90% policy rather than from noise (and, critically, with
        the turn sign already right)."""
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.02)
        for _ in range(steps):
            phi = (torch.rand(256, 1, generator=gen) * 2 - 1) * np.pi   # bearing, 0 = ahead
            feat = torch.cat((torch.cos(phi), torch.sin(phi)), dim=-1)  # [fwd_hat, lat_hat]
            target = torch.tanh(gain * phi)                             # correct-sign turn
            loss = ((torch.tanh(self.net(feat)) - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def forward(self, observations):
        to_target = observations[..., self.target_slice]
        lateral = to_target[..., 0, None]
        forward = -to_target[..., 1, None]
        dist = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
        features = torch.cat((forward / dist, lateral / dist), dim=-1)  # unit bearing

        u = torch.tanh(self.net(features)) * self.turn_strength  # in (-strength, strength)
        left = u.clamp(min=0)      # u > 0  => food to the left  => turn left
        right = (-u).clamp(min=0)  # u < 0  => food to the right => turn right
        speed = torch.ones_like(u)
        return right, left, speed


def make_learned_steering(n_joints, hidden_size=8, turn_strength=0.75, warm_start=True):
    """Option 5 controller -- learns the steering decision on top of the fixed turn
    primitive, warm-started at the correct-sign hand solution. See `LearnedSteering`."""
    return LearnedSteering(n_joints, hidden_size=hidden_size, turn_strength=turn_strength,
                           warm_start=warm_start)


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
    'obstacle_avoidance': ('make_obstacle_avoidance_reflex', ('evasion',)),
    'mlp_foraging': ('make_foraging_mlp', ('foraging', 'swim_to_ball')),
    'mlp_obstacle_avoidance': ('make_obstacle_avoidance_mlp', ('evasion',)),
    # Hardcoded turn tests: sensor-free, so valid on every task (they read no
    # to_target/to_obstacle vector, they just hold the turn signal to one side).
    'turn_left': ('make_turn_left_reflex', tuple(TASKS)),
    'turn_right': ('make_turn_right_reflex', tuple(TASKS)),
    # Hardcoded correct-convention navigator (~90% on foraging, no learning).
    'steer_to_food': ('make_steer_to_food_reflex', ('foraging', 'swim_to_ball')),
    # Learns only the steering decision on top of the fixed turn primitive (Option 5).
    'learned_steering': ('make_learned_steering', ('foraging', 'swim_to_ball')),
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


def make_controller(controller, n_joints, **controller_kwargs):
    """Build the named controller for an `n_joints` body; None passes through as None.

    Extra keyword arguments go straight to the factory, e.g.
    make_controller('learned_steering', 5, turn_strength=0.5).
    """
    if controller is None:
        if controller_kwargs:
            raise ValueError(
                f'controller=None takes no options, got {sorted(controller_kwargs)}'
            )
        return None
    if controller not in CONTROLLERS:
        raise ValueError(
            f'controller must be None or one of {sorted(CONTROLLERS)}, got {controller!r}'
        )
    return globals()[CONTROLLERS[controller][0]](n_joints, **controller_kwargs)
