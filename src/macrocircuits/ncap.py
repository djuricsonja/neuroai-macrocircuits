"""The NCAP swimmer: a C.-elegans-inspired neural circuit architectural prior."""

import numpy as np
import torch
from torch import nn

from macrocircuits.constraints import (
    excitatory,
    excitatory_constant,
    excitatory_uniform,
    graded,
    inhibitory,
    inhibitory_constant,
    inhibitory_uniform,
    unsigned,
    unsigned_constant,
    unsigned_uniform,
)


# The learned target-steering projection, named <channel><target>:
#   channel -- r/l  lateral: target to the right / left of the head,
#              a/b  longitudinal: target ahead of / behind the head.
#   target  -- d/v  the dorsal / ventral head B-neuron it drives.
# Every one is sign-free and learned, so the circuit discovers which channel
# should excite which side; the wiring only supplies the four sensory channels.
_STEER_WEIGHTS = ('rd', 'rv', 'ld', 'lv', 'ad', 'av', 'bd', 'bv')


class SwimmerModule(nn.Module):
    """C.-elegans-inspired neural circuit architectural prior."""

    def __init__(
            self,
            n_joints: int,
            n_turn_joints: int = 1,
            oscillator_period: int = 60,
            use_weight_sharing: bool = True,
            use_weight_constraints: bool = True,
            use_weight_constant_init: bool = True,
            include_proprioception: bool = True,
            include_head_oscillators: bool = True,
            include_speed_control: bool = False,
            include_turn_control: bool = False,
            include_target_steering: bool = False,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_turn_joints = n_turn_joints
        self.oscillator_period = oscillator_period
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        self.include_speed_control = include_speed_control
        self.include_turn_control = include_turn_control
        self.include_target_steering = include_target_steering

        # Log activity
        self.connections_log = []

        # Timestep counter (for oscillations).
        self.timestep = 0

        # Weight sharing switch function.
        self.ws = lambda nonshared, shared: shared if use_weight_sharing else nonshared

        # Weight constraint and init functions.
        if use_weight_constraints:
            self.exc = excitatory
            self.inh = inhibitory
            if use_weight_constant_init:
                exc_param = excitatory_constant
                inh_param = inhibitory_constant
            else:
                exc_param = excitatory_uniform
                inh_param = inhibitory_uniform
        else:
            self.exc = unsigned
            self.inh = unsigned
            if use_weight_constant_init:
                exc_param = inh_param = unsigned_constant
            else:
                exc_param = inh_param = unsigned_uniform

        # Steering (version B) always uses sign-free weights so the turn
        # *direction* is discovered rather than wired: fixed magnitude, random
        # sign.
        #
        # The magnitude is set to 0.5 from measurement, not taste. An earlier
        # near-zero init (+/-0.1) did not train at all: it changed episode return
        # by only ~0.5 while the reward's own epoch-to-epoch noise spans several
        # units, so PPO's advantages were pure variance, the updates cancelled,
        # and after 150k steps the weights had not left their init band -- a
        # self-trapping init. Sweeping the scale over random-sign draws, the
        # spread of return *between* inits (the signal PPO climbs) goes
        # 0.75 (at 0.1) -> 0.92 (0.3) -> 2.09 (0.5) -> 1.47 (0.8, over-driven and
        # worse on average). 0.5 maximises both that spread and mean return.
        self.uns = unsigned
        steer_param = lambda: unsigned_constant(lower=-0.5, upper=0.5)

        # Learnable parameters.
        self.params = nn.ParameterDict()
        if use_weight_sharing:
            if self.include_proprioception:
                self.params['bneuron_prop'] = exc_param()
            if self.include_speed_control:
                self.params['bneuron_speed'] = inh_param()
            if self.include_turn_control:
                self.params['bneuron_turn'] = exc_param()
            if self.include_head_oscillators:
                self.params['bneuron_osc'] = exc_param()
            # Learned target-directed steering: eight sign-free sensorimotor
            # weights, shared across the head modules that steer. Four carry the
            # lateral channels (right/left -> dorsal/ventral head B-neuron), four
            # the longitudinal ones (ahead/behind -> dorsal/ventral) so a target
            # directly behind -- where lateral ~ 0 -- can still drive a turn.
            if self.include_target_steering:
                for _name in _STEER_WEIGHTS:
                    self.params[f'bneuron_steer_{_name}'] = steer_param()
            self.params['muscle_ipsi'] = exc_param()
            self.params['muscle_contra'] = inh_param()
        else:
            for i in range(self.n_joints):
                if self.include_proprioception and i > 0:
                    self.params[f'bneuron_d_prop_{i}'] = exc_param()
                    self.params[f'bneuron_v_prop_{i}'] = exc_param()

                if self.include_speed_control:
                    self.params[f'bneuron_d_speed_{i}'] = inh_param()
                    self.params[f'bneuron_v_speed_{i}'] = inh_param()

                if self.include_turn_control and i < self.n_turn_joints:
                    self.params[f'bneuron_d_turn_{i}'] = exc_param()
                    self.params[f'bneuron_v_turn_{i}'] = exc_param()

                if self.include_head_oscillators and i == 0:
                    self.params[f'bneuron_d_osc_{i}'] = exc_param()
                    self.params[f'bneuron_v_osc_{i}'] = exc_param()

                # Per-head-module steering weights (see the shared branch).
                if self.include_target_steering and i < self.n_turn_joints:
                    for _name in _STEER_WEIGHTS:
                        self.params[f'bneuron_steer_{_name}_{i}'] = steer_param()

                self.params[f'muscle_d_d_{i}'] = exc_param()
                self.params[f'muscle_d_v_{i}'] = inh_param()
                self.params[f'muscle_v_v_{i}'] = exc_param()
                self.params[f'muscle_v_d_{i}'] = inh_param()

    def reset(self):
        self.timestep = 0

    def log_activity(self, activity_type, neuron):
        """Logs an active connection between neurons."""
        self.connections_log.append((self.timestep, activity_type, neuron))

    def forward(
            self,
            joint_pos,
            right_control=None,
            left_control=None,
            speed_control=None,
            timesteps=None,
            target_vec=None,
            log_activity=True,
            log_file='log.txt'
    ):
        """Forward pass.

    Args:
      joint_pos (torch.Tensor): Joint positions in [-1, 1], shape (..., n_joints).
      right_control (torch.Tensor): Right turn control in [0, 1], shape (..., 1).
      left_control (torch.Tensor): Left turn control in [0, 1], shape (..., 1).
      speed_control (torch.Tensor): Speed control in [0, 1], 0 stopped, 1 fastest, shape (..., 1).
      timesteps (torch.Tensor): Timesteps in [0, max_env_steps], shape (..., 1).
      target_vec (torch.Tensor): Head-egocentric [forward, lateral] vector to the
        target, shape (..., 2). Only used when include_target_steering is set.

    Returns:
      (torch.Tensor): Joint torques in [-1, 1], shape (..., n_joints).
    """

        exc = self.exc
        inh = self.inh
        ws = self.ws

        # Record connections only when asked. The visualization path keeps the default
        # (log_activity=True); long training runs -- evolution strategies calls forward
        # millions of times -- pass log_activity=False so connections_log can't grow
        # without bound.
        log = self.log_activity if log_activity else (lambda *a, **kw: None)

        # Separate into dorsal and ventral sensor values in [0, 1], shape (..., n_joints).
        joint_pos_d = joint_pos.clamp(min=0, max=1)
        joint_pos_v = joint_pos.clamp(min=-1, max=0).neg()

        # Convert speed signal from acceleration into brake.
        if self.include_speed_control:
            assert speed_control is not None
            speed_control = 1 - speed_control.clamp(min=0, max=1)

        # Split the egocentric vector to the target into four rectified channels
        # (as proprioception is split dorsal/ventral), so the learned sign-free
        # steering weights below can act on either turn direction. Steering is
        # purely directional -- no distance term -- so it fades to zero as the
        # worm aligns, giving a self-correcting tropism.
        #
        # Axis convention, measured from the model rather than assumed: the nose
        # sits at head-local [0, -0.06, 0], so the body's long axis is y (forward
        # is -y, matching the swim reward's -head_vel[1]) and x is lateral.
        # Hence component 0 is left/right and component 1 is forward/backward.
        # The *sign* of each is not assumed -- the weights are learned.
        if self.include_target_steering:
            assert target_vec is not None, 'include_target_steering needs target_vec'
            lateral = target_vec[..., 0, None]
            longitudinal = target_vec[..., 1, None]
            steer_r = lateral.clamp(min=0, max=1)
            steer_l = (-lateral).clamp(min=0, max=1)
            # Longitudinal channels break the dead zone where the target is
            # directly behind: lateral ~ 0 there, so without these the circuit
            # would have no drive to turn around.
            steer_a = (-longitudinal).clamp(min=0, max=1)  # forward is -y
            steer_b = longitudinal.clamp(min=0, max=1)

        joint_torques = []  # [shape (..., 1)]
        for i in range(self.n_joints):
            bneuron_d = bneuron_v = torch.zeros_like(joint_pos[..., 0, None])  # shape (..., 1)

            # B-neurons recieve proprioceptive input from previous joint to propagate waves down the body.
            if self.include_proprioception and i > 0:
                bneuron_d = bneuron_d + joint_pos_d[
                    ..., i - 1, None] * exc(self.params[ws(f'bneuron_d_prop_{i}', 'bneuron_prop')])
                bneuron_v = bneuron_v + joint_pos_v[
                    ..., i - 1, None] * exc(self.params[ws(f'bneuron_v_prop_{i}', 'bneuron_prop')])
                log('exc', f'bneuron_d_prop_{i}')
                log('exc', f'bneuron_v_prop_{i}')

            # Speed control unit modulates all B-neurons.
            if self.include_speed_control:
                bneuron_d = bneuron_d + speed_control * inh(
                    self.params[ws(f'bneuron_d_speed_{i}', 'bneuron_speed')]
                )
                bneuron_v = bneuron_v + speed_control * inh(
                    self.params[ws(f'bneuron_v_speed_{i}', 'bneuron_speed')]
                )
                log('inh', f'bneuron_d_speed_{i}')
                log('inh', f'bneuron_v_speed_{i}')

            # Turn control units modulate head B-neurons.
            if self.include_turn_control and i < self.n_turn_joints:
                assert right_control is not None
                assert left_control is not None
                turn_control_d = right_control.clamp(min=0, max=1)  # shape (..., 1)
                turn_control_v = left_control.clamp(min=0, max=1)
                bneuron_d = bneuron_d + turn_control_d * exc(
                    self.params[ws(f'bneuron_d_turn_{i}', 'bneuron_turn')]
                )
                bneuron_v = bneuron_v + turn_control_v * exc(
                    self.params[ws(f'bneuron_v_turn_{i}', 'bneuron_turn')]
                )
                log('exc', f'bneuron_d_turn_{i}')
                log('exc', f'bneuron_v_turn_{i}')

            # Learned target-directed steering (version B): a sign-free
            # sensorimotor projection from the split lateral offset into the head
            # B-neurons. The four weights are learned, so the turn direction is
            # discovered, not wired; it is still a fixed, sparse NCAP motif (no
            # hidden layer, no dense connectivity), unlike an MLP controller.
            if self.include_target_steering and i < self.n_turn_joints:
                uns = self.uns

                def steer_w(name, i=i):
                    return uns(self.params[ws(f'bneuron_steer_{name}_{i}', f'bneuron_steer_{name}')])

                bneuron_d = (
                    bneuron_d + steer_r * steer_w('rd') + steer_l * steer_w('ld') +
                    steer_a * steer_w('ad') + steer_b * steer_w('bd')
                )
                bneuron_v = (
                    bneuron_v + steer_r * steer_w('rv') + steer_l * steer_w('lv') +
                    steer_a * steer_w('av') + steer_b * steer_w('bv')
                )
                log('steer', f'bneuron_d_steer_{i}')
                log('steer', f'bneuron_v_steer_{i}')

            # Oscillator units modulate first B-neurons.
            if self.include_head_oscillators and i == 0:
                if timesteps is not None:
                    phase = timesteps.round().remainder(self.oscillator_period)
                    mask = phase < self.oscillator_period // 2
                    oscillator_d = torch.zeros_like(timesteps)  # shape (..., 1)
                    oscillator_v = torch.zeros_like(timesteps)  # shape (..., 1)
                    oscillator_d[mask] = 1.
                    oscillator_v[~mask] = 1.
                else:
                    phase = self.timestep % self.oscillator_period  # in [0, oscillator_period)
                    if phase < self.oscillator_period // 2:
                        oscillator_d, oscillator_v = 1.0, 0.0
                    else:
                        oscillator_d, oscillator_v = 0.0, 1.0
                bneuron_d = bneuron_d + oscillator_d * exc(
                    self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')]
                )
                bneuron_v = bneuron_v + oscillator_v * exc(
                    self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')]
                )

                log('exc', f'bneuron_d_osc_{i}')
                log('exc', f'bneuron_v_osc_{i}')

            # B-neuron activation.
            bneuron_d = graded(bneuron_d)
            bneuron_v = graded(bneuron_v)

            # Muscles receive excitatory ipsilateral and inhibitory contralateral input.
            muscle_d = graded(
                bneuron_d * exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')]) +
                bneuron_v * inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            )
            muscle_v = graded(
                bneuron_v * exc(self.params[ws(f'muscle_v_v_{i}', 'muscle_ipsi')]) +
                bneuron_d * inh(self.params[ws(f'muscle_v_d_{i}', 'muscle_contra')])
            )

            # Joint torque from antagonistic contraction of dorsal and ventral muscles.
            joint_torque = muscle_d - muscle_v
            joint_torques.append(joint_torque)

        self.timestep += 1

        out = torch.cat(joint_torques, -1)  # shape (..., n_joints)
        return out


class SwimmerActor(nn.Module):
    def __init__(
            self,
            swimmer,
            controller=None,
            distribution=None,
            timestep_transform=(-1, 1, 0, 1000),
    ):
        super().__init__()
        self.swimmer = swimmer
        self.controller = controller
        self.distribution = distribution
        self.timestep_transform = timestep_transform

    def initialize(
            self,
            observation_space,
            action_space,
            observation_normalizer=None,
    ):
        self.action_size = action_space.shape[0]

    def forward(self, observations):
        joint_pos = observations[..., :self.action_size]
        timesteps = observations[..., -1, None]

        # Target-directed steering reads the head-egocentric [forward, lateral]
        # vector the foraging / single-target tasks place immediately after the
        # joints (indices action_size .. action_size+2). Only consulted when the
        # circuit was built to steer, so plain-swim runs are unaffected and no
        # index is silently misread on a task that has no target.
        if getattr(self.swimmer, 'include_target_steering', False):
            target_vec = observations[..., self.action_size:self.action_size + 2]
        else:
            target_vec = None

        # Normalize joint positions by max joint angle (in radians).
        joint_limit = 2 * np.pi / (self.action_size + 1)  # In dm_control, calculated with n_bodies.
        joint_pos = torch.clamp(joint_pos / joint_limit, min=-1, max=1)

        # Convert normalized time signal into timestep.
        if self.timestep_transform:
            low_in, high_in, low_out, high_out = self.timestep_transform
            timesteps = (timesteps - low_in) / (high_in - low_in) * (high_out - low_out) + low_out

        # Generate high-level control signals.
        if self.controller:
            right, left, speed = self.controller(observations)
        else:
            right, left, speed = None, None, None

        # Generate low-level action signals.
        actions = self.swimmer(
            joint_pos,
            timesteps=timesteps,
            target_vec=target_vec,
            right_control=right,
            left_control=left,
            speed_control=speed,
        )

        # Pass through distribution for stochastic policy.
        if self.distribution:
            actions = self.distribution(actions)

        return actions
