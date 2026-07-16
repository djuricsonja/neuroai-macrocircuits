"""Swimmer environments and the swim tasks the tutorial trains on.

Importing this module registers `swim` and `swim_12_links` with the dm_control
swimmer suite, which is what makes `suite.load('swimmer', 'swim')` resolve.
"""

import collections

import dm_control.suite.swimmer as swimmer
import numpy as np
from dm_control.rl import control
from dm_control.utils import rewards

from macrocircuits.video import display_video

_SWIM_SPEED = 0.1


class Swim(swimmer.Swimmer):
    """Task to swim forwards at the desired speed."""

    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target by setting alpha to 0.
        physics.named.model.mat_rgba['target', 'a'] = 0
        physics.named.model.mat_rgba['target_default', 'a'] = 0
        physics.named.model.mat_rgba['target_highlight', 'a'] = 0

    def get_observation(self, physics):
        """Returns an observation of joint angles and body velocities."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Returns a smooth reward that is 0 when stopped or moving backwards, and rises linearly to 1
        when moving forwards at the desired speed."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )


class SwimToBall(Swim):
    """Swim forwards AND earn extra reward for reaching the ball (target).

    Reuses `Swim`'s forward-velocity reward and re-enables the target ("ball") the
    stock dm_control swimmer already provides but that `Swim` hides: the ball is left
    visible, its head-local direction is exposed as a `to_target` observation, and a
    distance-to-ball bonus is added on top of the forward reward.
    """

    def __init__(self, desired_speed=_SWIM_SPEED, target_reward_weight=1.0, **kwargs):
        super().__init__(desired_speed=desired_speed, **kwargs)
        self._target_reward_weight = target_reward_weight

    def initialize_episode(self, physics):
        # Skip Swim.initialize_episode (which hides the target); call the grandparent
        # (dm_control Swimmer) directly so the ball is placed randomly AND stays visible.
        super(Swim, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns joints, the head-local vector to the ball, and body velocities.

        `to_target` is inserted between `joints` and `body_velocities` on purpose: the
        NCAP actor slices joints from the front and the time feature from the back, so
        keeping joints first and body_velocities last leaves both slices valid.
        """
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['to_target'] = physics.nose_to_target()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Returns the forward-swim reward plus a weighted bonus for nearing the ball."""
        forward_reward = super().get_reward(physics)
        # The ball's radius doubles as the success threshold (as in stock dm_control).
        target_size = physics.named.model.geom_size['target', 0]
        reach_reward = rewards.tolerance(
            physics.nose_to_target_dist(),
            bounds=(0, target_size),
            margin=5 * target_size,
            sigmoid='long_tail',
        )
        return forward_reward + self._target_reward_weight * reach_reward


@swimmer.SUITE.add()
def swim(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task for a n-link swimmer."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = Swim(desired_speed=desired_speed, random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@swimmer.SUITE.add()
def swim_12_links(
    n_links=12,
    desired_speed=_SWIM_SPEED,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task for a n-link swimmer."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = Swim(desired_speed=desired_speed, random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@swimmer.SUITE.add()
def swim_to_ball(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    target_reward_weight=1.0,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the SwimToBall task for a n-link swimmer."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = SwimToBall(
        desired_speed=desired_speed,
        target_reward_weight=target_reward_weight,
        random=random,
    )
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


def render(env):
    """Renders the current environment state to an image."""
    return env.physics.render(camera_id=0, width=640, height=480)


def test_dm_control(env):
    """Tests a DeepMind control suite environment by executing a series of random actions."""
    spec = env.action_spec()
    timestep = env.reset()
    frames = [render(env)]

    for _ in range(60):
        action = np.random.uniform(
            low=spec.minimum,
            high=spec.maximum,
            size=spec.shape,
        )
        timestep = env.step(action)
        frames.append(render(env))

    return display_video(frames)
