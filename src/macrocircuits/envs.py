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
