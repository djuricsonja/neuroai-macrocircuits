import torch
import torch.nn as nn


def foraging_target(observations, n_joints):
    """Assumes task layout: joints, to_target, body_velocities
    (i.e. enable_foraging=True, enable_obstacles=False)."""
    target_slice = slice(n_joints, n_joints + 2)
    to_target = observations[..., target_slice]     # [forward, lateral], head-egocentric
    return to_target


def obstacle_avoidance_target(observations, n_joints):
    """Assumes task layout: joints, to_obstacle, body_velocities
    (i.e. enable_obstacles=True, enable_foraging=False)."""
    obstacle_slice = slice(n_joints, n_joints + 2)
    to_obstacle = observations[..., obstacle_slice]   # [forward, lateral], head-egocentric
    return to_obstacle


def forage_and_avoid_target(observations, n_joints):
    """Assumes task layout: joints, to_target, to_obstacle, body_velocities
    (i.e. enable_foraging=True, enable_obstacles=True)."""
    target_slice = slice(n_joints, n_joints + 2)
    obstacle_slice = slice(n_joints + 2, n_joints + 4)
    target = observations[..., target_slice]
    obstacle = observations[..., obstacle_slice]
    combined = torch.cat((target, obstacle), dim=-1)
    return combined


class MLP_controller(nn.Module):
    """Learns to map sensed target position to steering/speed commands,
    instead of hand-deriving the mapping. Assumes task layout:
    joints, to_target, body_velocities (enable_foraging=True)."""

    def __init__(self, n_joints, input_size=2, hidden_size=16):
        super().__init__()
        self.n_joints = n_joints
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # input: [forward, lateral] to target
            nn.Tanh(),
            nn.Linear(hidden_size, 3),   # output: right, left, speed (pre-activation)
        )

    def forward(self, observations, swimmer=None):
        target_slice = slice(self.n_joints, self.n_joints + 2)
        to_target = observations[..., target_slice]
        out = torch.sigmoid(self.net(to_target))   # squash to [0, 1], same range other controllers used
        right, left, speed = out.split(1, dim=-1)  # each stays shape (..., 1)
        return right, left, speed


def controllers_map(controller_name, n_joints=None, hidden_size=16):
    if controller_name is None:
        return None
    if n_joints is not None:
        if n_joints < 2:
            return None
    
    name = controller_name.upper()
    if name in ['FORAGE', 'FORAGING', 'AVOID', 'AVOIDANCE', 'OBSTACLE', 'EVASION']:
        return MLP_controller(n_joints=n_joints, input_size=2, hidden_size=hidden_size)
    elif name in ['FORAGE_AVOID', 'FORAGING_AVOIDANCE', 'FORAGING_EVASION']:
        return MLP_controller(n_joints=n_joints, input_size=4, hidden_size=hidden_size)
    else:
        return None