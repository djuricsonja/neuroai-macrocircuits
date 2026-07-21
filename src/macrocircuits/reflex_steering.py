"""Fixed, non-learned steering reflexes for NCAP.

Each derives the circuit's turn signals directly from the already-egocentric
to_target/to_obstacle vector a task puts in the observation, instead of learning that
mapping with a network -- the `MLPController` in `controllers` is the learned
alternative, and `controllers.CONTROLLERS` is the registry a run picks either from.

A reflex is a plain closure `reflex(observations) -> (right, left, speed)`, each
`(..., 1)` in `[0, 1]`; it holds no parameters, so a run using one still trains or
evolves only the circuit's own `bneuron_turn` weight.
"""

import torch


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

