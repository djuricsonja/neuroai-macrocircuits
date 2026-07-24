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
| `'mlp_foraging'` / `'mlp_obstacle_avoidance'` | a small MLP (`MLPController`) | yes |

That is the comparison the tasks exist for: how much of the steering has to be learned
once the swimming itself is given by the architecture.

All three are the same shape to the rest of the code -- a callable
`controller(observations) -> (right, left, speed)`, each `(..., 1)` in `[0, 1]`, or
`None`. The reflexes are plain closures; `MLPController` is an `nn.Module`, so
assigning it to `SwimmerActor.controller` (RL) or `NCAPSwimmerPolicy.controller` (ES)
registers it as a submodule and its parameters are trained/evolved along with the
circuit's own. Nothing else has to know which kind it got.
"""


TAU = 5

import torch
import torch.nn as nn

from macrocircuits.envs import TASKS
from macrocircuits.reflex_steering import (
    make_foraging_reflex,
    make_obstacle_avoidance_reflex,
)

from macrocircuits.constraints import (
    excitatory_uniform, 
    inhibitory_uniform, 
    unsigned_uniform,
    excitatory,
    inhibitory,
    unsigned
)


# ==================================================================================================
# Biologically Inspired controllers.
def distance_to_food_target(observations, n_joints):
    to_target = observations[..., n_joints:n_joints + 2]
    return torch.norm(to_target, dim=-1)

class NaivePiouretteController(nn.Module):

    def __init__(self, n_joints, state_fn=distance_to_food_target, tau=20):
        super().__init__()
        self.tau = tau
        self.counter = 0
        self.n_joints = n_joints
        self.state_fn = state_fn
        self.concentration = torch.zeros((2,))
        self.controls = torch.tensor([0.0, 0.0, 1.0])
        self.dist = torch.distributions.Uniform(0.0, 1.0)

    def forward(self, observations, n_joints=None):
        to_target = self.state_fn(observations, self.n_joints)
        # concentration proxy: negative distance, so higher = closer to food
        x = -torch.norm(to_target, dim=-1)
        if x.dim() > 0:
            x = x[-1]

        with torch.no_grad():
            if (self.counter == 0) or (self.counter < (self.tau - 1)):
                if self.counter == 0:
                    self.concentration[0] = x
                self.counter += 1
            else:
                self.concentration[1] = x
                conc_gradient = self.concentration[1] - self.concentration[0]

                if conc_gradient < 0:
                    # concentration decreasing (getting farther) -> pirouette:
                    # pick a single turn direction and magnitude, keep speed fixed
                    magnitude = self.dist.sample((1,))
                    go_right = torch.rand(1) < 0.5
                    left = torch.where(go_right, torch.zeros(1), magnitude)
                    right = torch.where(go_right, magnitude, torch.zeros(1))
                    speed = torch.ones(1)
                    self.controls = torch.cat([left, right, speed])
                # else: concentration flat/increasing -> keep current heading (run)

                self.counter = 0

            left, right, speed = self.controls
            return right, left, speed
        
def make_foraging_naive_piourette(n_joints, tau=TAU):
    return NaivePiouretteController(n_joints, state_fn=distance_to_food_target, tau=tau)


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
istance_to_food_target
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



class MLPBased_PiouretteController(NaivePiouretteController):

    def __init__(self, n_joints, state_fn=foraging_state, tau=20):
        super().__init__(n_joints, state_fn, tau)
        # self.weight = nn.Parameter(torch.zeros((self.n_joints + 2, 3)))
        self.weight = excitatory_uniform((self.n_joints + 2, 3))
        self.tau_layer = nn.Parameter(torch.zeros(self.n_joints + 2,))

        self.prev_controls = torch.tensor([0., 0., 1.0])
        # Per-env state, lazily (re)sized on first call / batch-size change.
        self.counter_t = None
        self.tau_t = None
        self.prev_x = None
        self.conc_gradient = None

    def _ensure_batch_state(self, batch_size, device):
        if self.counter_t is None or self.counter_t.shape[0] != batch_size:
            self.counter_t = torch.zeros(batch_size, 1, device=device)
            self.prev_x = torch.zeros(batch_size, 1, device=device)
            self.conc_gradient = torch.zeros(batch_size, 1, device=device)
            self.controls = torch.tensor([0., 0., 1.0], device=device).expand(batch_size, 3).clone()
            self.prev_controls = self.controls.clone()

    def forward(self, observations, n_joints=None):
        with torch.no_grad():
            obs = self.state_fn(observations, self.n_joints)
            to_target = obs[:, self.n_joints:]
            x = -torch.norm(to_target, dim=-1, keepdim=True)  # (batch, 1) per-env concentration

            batch_size = obs.shape[0]
            self._ensure_batch_state(batch_size, obs.device)

            # Learned, per-env hold duration.
            neg_conc_gradient = torch.relu(-1 * self.conc_gradient)
            tau_t = torch.clamp(
                torch.relu(neg_conc_gradient * (obs @ self.tau_layer).unsqueeze(-1)),
                min=2, max=50,
            )

            # Per-env: has this env's hold timer expired?
            due = self.counter_t >= tau_t

            # Where due: recompute gradient from new baseline and reset; else: keep holding.
            self.conc_gradient = torch.where(due, x - self.prev_x, self.conc_gradient)
            self.prev_x = torch.where(due, x, self.prev_x)
            self.prev_controls = torch.where(due, self.controls, self.prev_controls)
            self.counter_t = torch.where(due, torch.zeros_like(self.counter_t), self.counter_t + 1)

            self.tau_t = tau_t

        neg_conc_gradient = torch.relu(-1 * self.conc_gradient)
        time_constant = torch.relu(torch.exp(-1 * (tau_t - 1 - self.counter_t)))
        self.controls = torch.sigmoid(
            (neg_conc_gradient * time_constant * (obs @ excitatory(self.weight))) + self.prev_controls
        )

        left, right, speed = self.controls.split(1, dim=-1)
        return right, left, speed


# class MLPBased_PiouretteController(NaivePiouretteController):

#     def __init__(self, n_joints, state_fn=foraging_state, tau=20):
#         super().__init__(n_joints, state_fn, tau)
#         self.weight = nn.Parameter(torch.zeros((self.n_joints + 2, 3)))
#         self.conc_gradient = torch.tensor(-1E12)
#         self.prev_controls = torch.tensor([0., 0., 1.0])

#         self.tau_layer = nn.Parameter(torch.zeros(self.n_joints + 2,))
#         self.learned_tau = None

#     def _get_conc_gradient(self, observations):
#         obs = self.state_fn(observations, self.n_joints)
#         # print(obs.size())
#         joints = obs[:, :self.n_joints]
#         to_target = obs[:, self.n_joints:]

#         # to_target = self.state_fn(observations, self.n_joints)
#         # concentration proxy: negative distance, so higher = closer to food
#         x = -torch.norm(to_target, dim=-1)
#         if x.dim() > 0:
#             x = x[-1]

#         if (self.counter == 0) or (self.counter < int(self.tau)):
#             if self.counter == 0:
#                 self.concentration[0] = x
#                 self.prev_controls = self.controls.clone().detach()
#             self.counter += 1
#         else:
#             self.concentration[1] = x
#             self.conc_gradient = torch.tensor(self.concentration[1] - self.concentration[0])
#             self.counter = 0

#         return torch.cat([joints, to_target], dim=-1), self.conc_gradient

#     def _get_counter(self):
#         return torch.tensor(self.counter)

#     def forward(self, observations, n_joints=None):
#         with torch.no_grad():
#             obs, conc_gradient = self._get_conc_gradient(observations)
#             batch_size = obs.shape[0]
#             counter = self._get_counter()

#             # if counter == 0:
#             #     print(f"CONCENTRATION GRADIENT: {conc_gradient}")

#              # Reset any batch-shaped state that doesn't match the current call's batch size
#             # (Trainer alternates 4096-env training rollouts with single-env test episodes).
#             if self.controls.dim() == 1 or self.controls.shape[0] != batch_size:
#                 self.controls = torch.tensor([0., 0., 1.0]).expand(batch_size, 3).clone()
#             if self.prev_controls.dim() == 1 or self.prev_controls.shape[0] != batch_size:
#                 self.prev_controls = self.controls.clone().detach()

#         neg_conc_gradient = torch.relu(-1 * conc_gradient)
#         tau_val = torch.relu(neg_conc_gradient * (obs @ self.tau_layer))
#         tau_val = torch.clamp(tau_val, min=2, max=50)
#         time_constant = torch.relu(torch.exp(-1 * (tau_val - 1 - counter)))
#         self.controls = torch.sigmoid((neg_conc_gradient * time_constant * (obs @ self.weight)) + self.prev_controls)
#         # print(obs.size(), self.controls.size(), self.prev_controls.size())

#         with torch.no_grad():
#             self.learned_tau = tau_val

#         left, right, speed = self.controls.split(1, dim=-1)
#         return right, left, speed


class MLPBased_ReflexController(nn.Module):

    def __init__(self, n_joints, state_fn=foraging_state):
        super().__init__()
        self.n_joints = n_joints
        self.state_fn = state_fn
        self.weight = nn.Parameter(torch.zeros((self.n_joints + 2, 3)))

    def forward(self, observations, n_joints=None):
        with torch.no_grad():
            x = self.state_fn(observations, self.n_joints)
            # to_target = self.state_fn(observations, self.n_joints)
            # x = to_target[-1]

        y = torch.sigmoid(x @ self.weight)
        left, right, speed = y.split(1, dim=-1)
        return right, left, speed


class MLP_Reflex_Piourette_Controller(nn.Module):

    def __init__(self, n_joints, state_fn=foraging_state, tau=20):
        super().__init__()

        self.reflex_controller = MLPBased_ReflexController(n_joints, state_fn)
        self.piourette_controller = MLPBased_PiouretteController(n_joints, state_fn, tau)

        self.n_joints = n_joints
        self.state_fn =state_fn
        self.mixer_layer = nn.Linear(6, 3)

    def forward(self, observations, n_joints=None):
        r1, l1, s1 = self.reflex_controller(observations)
        r2, l2, s2 = self.piourette_controller(observations)

        x = torch.cat([r1, r2, l1, l2, s1, s2], dim=-1)
        right, left, speed = torch.sigmoid(self.mixer_layer(x)).split(1, dim=-1)

        return right, left, speed


def make_foraging_mlp(n_joints, hidden_size=16):
    """Learned counterpart of `make_foraging_reflex`: steer from the vector to the food."""
    return MLPController(n_joints, foraging_state, hidden_size=hidden_size)

def make_foraging_mlp_piourette(n_joints, tau=TAU):
    return MLPBased_PiouretteController(n_joints, state_fn=foraging_state, tau=tau)

def make_foraging_mlp_reflex(n_joints):
    return MLPBased_ReflexController(n_joints, state_fn=foraging_state)

def make_foraging_mlp_reflex_piourette(n_joints, tau=TAU):
    return MLP_Reflex_Piourette_Controller(n_joints, state_fn=foraging_state, tau=tau)

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
    'obstacle_avoidance': ('make_obstacle_avoidance_reflex', ('evasion',)),
    'mlp_foraging': ('make_foraging_mlp', ('foraging', 'swim_to_ball')),
    'mlp_obstacle_avoidance': ('make_obstacle_avoidance_mlp', ('evasion',)),
    'naive_piourette_foraging': ('make_foraging_naive_piourette', ('foraging', 'swim_to_ball')),
    'mlp_piourette_foraging': ('make_foraging_mlp_piourette', ('foraging', 'swim_to_ball')),
    'mlp_reflex_foraging': ('make_foraging_mlp_reflex', ('foraging', 'swim_to_ball')),
    'mlp_reflex_piourette_foraging': ('make_foraging_mlp_reflex_piourette', ('foraging', 'swim_to_ball')),
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
