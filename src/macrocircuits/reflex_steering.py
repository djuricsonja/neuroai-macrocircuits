"""Derive NCAP's turn signals directly from the already-
egocentric to_target/to_obstacle vectors, instead of learning the mapping with a
network.

A run names one of these with `controller=` (see CONTROLLERS). Both trainers go
through this module: `training.run_config` names the factory inside the agent code
string it eval's, while `es.run_es` calls `make_controller` for the real thing.
"""

import torch

from macrocircuits.envs import TASKS


def make_foraging_reflex(n_joints):
    """Builds a reflex that steers toward the nearest food.

    Assumes observation layout: joints, to_target, ... (see Swim.get_observation).
    """
    target_slice = slice(n_joints, n_joints + 2)  # [forward, lateral], head-egocentric

    def reflex(observations):
        lateral = observations[..., target_slice][..., 1, None]
        right = lateral.clamp(min=0, max=1)    # food's to my right -> turn right
        left = (-lateral).clamp(min=0, max=1)  # food's to my left -> turn left
        speed = torch.ones_like(right)
        return right, left, speed

    return reflex


def make_obstacle_avoidance_reflex(n_joints, reaction_distance=0.6):
    """Builds a reflex that steers away from the nearest obstacle, scaled by proximity.

    Same observation layout as before, but now also uses distance (the magnitude
    of the to_obstacle vector) to scale the response: a distant obstacle barely
    registers, a close one dominates. reaction_distance is a fixed threshold.
    """
    obstacle_slice = slice(n_joints, n_joints + 2)

    def reflex(observations):
        to_obstacle = observations[..., obstacle_slice]  # [forward, lateral]
        lateral = to_obstacle[..., 1, None]
        distance = torch.norm(to_obstacle, dim=-1, keepdim=True)

        # 1 when right on top of the obstacle, 0 once farther than reaction_distance.
        urgency = (1 - distance / reaction_distance).clamp(min=0, max=1)

        right = (-lateral).clamp(min=0, max=1) * urgency
        left = lateral.clamp(min=0, max=1) * urgency
        speed = torch.ones_like(right)
        return right, left, speed

    return reflex


# The reflexes a run may choose, each mapped onto its factory and the tasks whose
# observation layout it assumes (see envs.TASKS). The factory is kept as a *name* so
# training.run_config can spell it into the agent source string it eval's; make_controller
# resolves it to the real thing for callers that just want the reflex.
CONTROLLERS = {
    'foraging': ('make_foraging_reflex', ('foraging', 'swim_to_ball')),
    'obstacle_avoidance': ('make_obstacle_avoidance_reflex', ('evasion',)),
}


def check_controller(controller, network, task):
    """Raise unless `controller` can actually steer this (network, task) combination.

    Called by both trainers before anything is built, so a mismatch fails with an
    explanation rather than as a bare AssertionError inside SwimmerModule (turn control
    on with no signal to feed it) or as a silently useless reflex reading whichever
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
    """Build the named reflex for an `n_joints` body; None passes through as None."""
    if controller is None:
        return None
    if controller not in CONTROLLERS:
        raise ValueError(
            f'controller must be None or one of {sorted(CONTROLLERS)}, got {controller!r}'
        )
    return globals()[CONTROLLERS[controller][0]](n_joints)

