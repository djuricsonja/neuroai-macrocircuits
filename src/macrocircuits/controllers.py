import torch

def foraging_controller(observations, swimmer):
    n_joints = swimmer.n_joints
    target_slice = slice(n_joints, n_joints + 2)
    to_target = observations[..., target_slice]     # [forward, lateral], head-egocentric
    grad = to_target[..., 1:2]                        # keep trailing dim -> shape (..., 1)

    right = torch.clamp(grad, min=0)
    left = torch.clamp(-grad, min=0)
    speed = torch.ones_like(grad)
    return right, left, speed


def controllers_map(controller_name):
    if controller_name is None:
        return None
    elif controller_name.upper() in  ['FORAGE', 'FORAGING']:
        return foraging_controller
    else:
        return None