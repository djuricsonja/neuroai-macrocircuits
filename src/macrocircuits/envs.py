import collections

import dm_control.suite.swimmer as swimmer
import numpy as np
from dm_control.rl import control
from dm_control.utils import rewards
from lxml import etree

from macrocircuits.video import display_video
import mujoco

_SWIM_SPEED = 0.1


def _add_obstacles(
    model_string,
    n_obstacles,
    radius=0.05,
    min_distance_from_origin=0.2,
    max_distance_from_origin=0.6,
    min_obstacle_separation=0.5,
):
    """Adds n_obstacles static spherical obstacles to the swimmer MJCF, at
    random non-overlapping positions surrounding the origin (the worm's
    start point). Positions are fixed at construction time -- see note
    below if you also want per-episode re-randomization."""
    mjcf = etree.fromstring(model_string)
    worldbody = mjcf.find('./worldbody')

    placed = []
    for i in range(n_obstacles):
        for _ in range(50):  # rejection sampling attempts
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(min_distance_from_origin, max_distance_from_origin)
            xpos = dist * np.cos(angle)
            ypos = dist * np.sin(angle)
            if all(
                np.hypot(xpos - px, ypos - py) >= min_obstacle_separation
                for px, py in placed
            ):
                break  # found a non-overlapping spot
        placed.append((xpos, ypos))

        obstacle = etree.SubElement(worldbody, 'body', name=f'obstacle_{i}')
        obstacle.set('pos', f'{xpos} {ypos} 0.05')
        etree.SubElement(obstacle, 'geom', {
            'name': f'obstacle_{i}',
            'type': 'sphere',
            'size': str(radius),
            'rgba': '0.8 0.2 0.2 1',
            'contype': '1',
            'conaffinity': '1',
            'margin': '0.02'
        })
    return etree.tostring(mjcf, pretty_print=True)


def get_model_and_assets(n_joints, n_obstacles=0):
    model_string, assets = swimmer.get_model_and_assets(n_joints)
    if n_obstacles > 0:
        model_string = _add_obstacles(model_string, n_obstacles)
    return model_string, assets


class Physics(swimmer.Physics):
    def _body_geom_names(self):
        return [n for n in self.named.data.geom_xpos.axes.row.names if n.startswith('visual')]

    def nose_to_obstacles(self, n_obstacles):
        """Head-local (x, y) vector + true min distance from the whole body to each obstacle."""
        head_orientation = self.named.data.xmat['head'].reshape(3, 3)
        body_geoms = self._body_geom_names()
        vectors = []
        for i in range(n_obstacles):
            obs_pos = self.named.data.geom_xpos[f'obstacle_{i}']
            dists = [np.linalg.norm(self.named.data.geom_xpos[g] - obs_pos) for g in body_geoms]
            closest_geom = body_geoms[int(np.argmin(dists))]
            local_vec = (obs_pos - self.named.data.geom_xpos[closest_geom]).dot(head_orientation)[:2]
            vectors.append(local_vec)
        return np.array(vectors)

    def nearest_obstacle(self, n_obstacles):
        vectors = self.nose_to_obstacles(n_obstacles)
        dists = np.linalg.norm(vectors, axis=-1)
        idx = np.argmin(dists)
        return vectors[idx], dists[idx]


class Swim(swimmer.Swimmer):
    """Swim forwards, with independently toggleable foraging and obstacle avoidance."""

    def __init__(
        self,
        desired_speed=_SWIM_SPEED,
        enable_single_target=False,
        enable_foraging=False,
        enable_obstacles=False,
        n_obstacles=3,
        target_reward_weight=1.0,
        obstacle_penalty_weight=1.0,
        obstacle_safe_distance=0.4,
        obstacle_min_distance=0.5,
        food_size=0.02,
        food_flash_steps=5,       # how many control steps the flash lasts
        food_flash_color=(1.0, 1.0, 0.2, 1.0),  # bright yellow flash
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self._enable_single_target = enable_single_target
        self._enable_foraging = enable_foraging
        self._enable_obstacles = enable_obstacles
        self._n_obstacles = n_obstacles if enable_obstacles else 0
        self._target_reward_weight = target_reward_weight
        self._obstacle_penalty_weight = obstacle_penalty_weight
        self._obstacle_safe_distance = obstacle_safe_distance
        self._obstacle_min_distance = obstacle_min_distance
        self._food_size = food_size
        self._food_flash_steps = food_flash_steps
        self._food_flash_color = np.array(food_flash_color)
        self._food_flash_remaining = 0
        self._food_default_color = None  # captured on first reset


    def initialize_episode(self, physics):
        if self._enable_foraging or self._enable_single_target:
            # Skip Swim's target-hiding step; call the grandparent (stock Swimmer)
            # directly so the target is randomly placed AND stays visible.
            super(Swim, self).initialize_episode(physics)
            physics.forward()

            if self._enable_foraging:
                physics.named.model.geom_size['target', 0] = self._food_size

                if self._food_default_color is None:
                    self._food_default_color = physics.named.model.mat_rgba['target'].copy()
                physics.named.model.mat_rgba['target'] = self._food_default_color
                self._food_flash_remaining = 0

                # Place food a short distance directly ahead of the worm's current heading.
                forage_distance = 0.15  # ~1-2 forward moves given ~0.1 unit segment spacing
                head_pos = physics.named.data.xpos['head']
                head_orientation = physics.named.data.xmat['head'].reshape(3, 3)
                forward_local = np.array([0, -1, 0])  # head-local forward, per reward's -y convention
                forward_world = head_orientation.dot(forward_local)
                target_xy = head_pos[:2] + forage_distance * forward_world[:2]

                physics.named.model.geom_pos['target', 'x'] = target_xy[0]
                physics.named.model.geom_pos['target', 'y'] = target_xy[1]
                physics.forward()  # propagate the new target position
        else:
            super().initialize_episode(physics)
            physics.named.model.mat_rgba['target', 'a'] = 0
            physics.named.model.mat_rgba['target_default', 'a'] = 0
            physics.named.model.mat_rgba['target_highlight', 'a'] = 0

    def all_body_positions(self, physics):
        """World (x, y, z) position of every body in the model, including the head."""
        names = physics.named.data.xpos.axes.row.names
        return names, physics.named.data.xpos[:]
    
    def worm_positions(physics, n_joints, head_name):
        """World (x, y, z) position of just the worm's own segments, excluding
        world/target/obstacle bodies. head_name must be confirmed from
        all_body_positions' printed names first (e.g. 'head')."""
        names = [head_name] + [f'segment_{i}' for i in range(n_joints)]
        return np.array([physics.named.data.xpos[n] for n in names])
    
    def _respawn_food_near_worm(self, physics):
        """Places the food a short distance in front of or beside the worm's
        *current* position/heading -- never behind it."""
        head_pos = physics.named.data.xpos['head']
        head_orientation = physics.named.data.xmat['head'].reshape(3, 3)
        local_angle = self.random.uniform(-np.pi / 2, np.pi / 2)  # front-hemisphere only
        forward_local = np.array([np.sin(local_angle), -np.cos(local_angle), 0])  # -y is forward
        distance = self.random.uniform(0.15, 0.3)
        forward_world = head_orientation.dot(forward_local)
        target_xy = head_pos[:2] + distance * forward_world[:2]
        physics.named.model.geom_pos['target', 'x'] = target_xy[0]
        physics.named.model.geom_pos['target', 'y'] = target_xy[1]
        physics.forward()

    def get_observation(self, physics):
        """joints, [to_target], [to_obstacle], body_velocities -- in that fixed order,
        so slices stay predictable regardless of which flags are on."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        if self._enable_foraging or self._enable_single_target:
            obs['to_target'] = physics.nose_to_target()
        if self._enable_obstacles:
            vector, _ = physics.nearest_obstacle(self._n_obstacles)
            obs['to_obstacle'] = vector
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        reward = rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )

        if self._enable_single_target:
            target_size = physics.named.model.geom_size['target', 0]
            reward += rewards.tolerance(
                physics.nose_to_target_dist(),
                bounds=(0, target_size),
                margin=5 * target_size,
                sigmoid='long_tail',
            )

        if self._enable_foraging:
            target_size = physics.named.model.geom_size['target', 0]
            dist = physics.nose_to_target_dist()
            reward += self._target_reward_weight * rewards.tolerance(
                dist,
                bounds=(0, target_size),
                margin=5 * target_size,
                sigmoid='long_tail',
            )

            if self._food_flash_remaining > 0:
                self._food_flash_remaining -= 1
                if self._food_flash_remaining == 0:
                    self._respawn_food_near_worm(physics)
                    physics.named.model.mat_rgba['target'] = self._food_default_color
            elif dist < target_size:
                physics.named.model.mat_rgba['target'] = self._food_flash_color
                self._food_flash_remaining = self._food_flash_steps

        if self._enable_obstacles:
            _, dist = physics.nearest_obstacle(self._n_obstacles)
            safety = rewards.tolerance(
                dist,
                bounds=(self._obstacle_safe_distance, float('inf')),
                margin=self._obstacle_safe_distance,
                value_at_margin=0.,
                sigmoid='linear',
            )
            reward -= self._obstacle_penalty_weight * (1 - safety)

        return reward


@swimmer.SUITE.add()
def swim(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    enable_single_target = False,
    enable_foraging=False,
    enable_obstacles=False,
    n_obstacles=3,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance."""
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    physics.model.opt.disableflags = 0
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        random=random,
    )
    return control.Environment(
        physics, task, time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP, **environment_kwargs,
    )


@swimmer.SUITE.add()
def swim_to_ball(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    enable_single_target = True,
    enable_foraging=False,
    enable_obstacles=False,
    n_obstacles=3,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance."""
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    physics.model.opt.disableflags = 0
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        random=random,
    )
    return control.Environment(
        physics, task, time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP, **environment_kwargs,
    )


@swimmer.SUITE.add()
def foraging(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    enable_single_target = False,
    enable_foraging=True,
    enable_obstacles=False,
    n_obstacles=3,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance."""
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    physics.model.opt.disableflags = 1
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        random=random,
    )
    return control.Environment(
        physics, task, time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP, **environment_kwargs,
    )


@swimmer.SUITE.add()
def evasion(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    enable_single_target = False,
    enable_foraging=False,
    enable_obstacles=True,
    n_obstacles=3,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance."""
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    physics.model.opt.disableflags = 0
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        random=random,
    )
    return control.Environment(
        physics, task, time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP, **environment_kwargs,
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