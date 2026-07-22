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


# class Physics(swimmer.Physics):
#     """Adds obstacle-awareness on top of the stock swimmer Physics."""

#     def nose_to_obstacles(self, n_obstacles):
#         """Head-local (x, y) vectors from nose to each obstacle, shape (n_obstacles, 2)."""
#         head_orientation = self.named.data.xmat['head'].reshape(3, 3)
#         nose_pos = self.named.data.geom_xpos['nose']
#         vectors = [
#             (self.named.data.geom_xpos[f'obstacle_{i}'] - nose_pos).dot(head_orientation)[:2]
#             for i in range(n_obstacles)
#         ]
#         return np.array(vectors)

#     def nearest_obstacle(self, n_obstacles):
#         """Head-local vector and distance to the closest obstacle."""
#         vectors = self.nose_to_obstacles(n_obstacles)
#         dists = np.linalg.norm(vectors, axis=-1)
#         idx = np.argmin(dists)
#         return vectors[idx], dists[idx]


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
        target_distance=None,
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
        self._target_distance = target_distance

    def _place_target(self, physics, around=None):
        """Reposition the target; `around` is the (x, y) to place it relative to.

        With `target_distance` set, the target goes at exactly that distance and a
        uniformly random bearing -- around the origin at episode start, around the
        worm's nose when a pellet respawns.

        Fixing the distance makes every episode a comparable navigation problem.
        dm_control's stock spawn samples a +/-2.0 box and, 20% of the time, a
        +/-0.3 one, so steering is irrelevant when the target lands almost on top
        of the worm and near-hopeless when it lands far away; its benefit is
        diluted across episodes. Measured on swim_to_ball, pinning the distance
        raised the steering-on vs -off effect from 41 to 141 return (SNR 0.47 ->
        1.41) -- almost entirely by growing the effect, not by cutting variance.

        The bearing stays uniformly random and the worm's heading is randomised by
        dm_control, so the egocentric angle is still fully random: the worm has to
        read the target vector and cannot settle on a fixed world heading. This is
        a training curriculum -- evaluate on the stock spawn (target_distance=None)
        to check that the learned steering generalises.
        """
        if self._target_distance is not None:
            angle = self.random.uniform(0, 2 * np.pi)
            origin_x, origin_y = (0.0, 0.0) if around is None else (around[0], around[1])
            xpos = origin_x + self._target_distance * np.cos(angle)
            ypos = origin_y + self._target_distance * np.sin(angle)
        elif around is None:
            return  # keep dm_control's own episode-start placement
        else:
            xpos, ypos = self.random.uniform(-1.5, 1.5, size=2)
        physics.named.model.geom_pos['target', 'x'] = xpos
        physics.named.model.geom_pos['target', 'y'] = ypos
        physics.named.model.light_pos['target_light', 'x'] = xpos
        physics.named.model.light_pos['target_light', 'y'] = ypos

    def initialize_episode(self, physics):
        # disabled = bool(physics.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # print('contact globally disabled:', disabled)

        if self._enable_foraging or self._enable_single_target:
            # Skip Swim's target-hiding step; call the grandparent (stock Swimmer)
            # directly so the target is randomly placed AND stays visible.
            super(Swim, self).initialize_episode(physics)
            if self._enable_foraging:
                physics.named.model.geom_size['target', 0] = self._food_size
            self._place_target(physics)  # no-op unless target_distance is set
        else:
            super().initialize_episode(physics)
            physics.named.model.mat_rgba['target', 'a'] = 0
            physics.named.model.mat_rgba['target_default', 'a'] = 0
            physics.named.model.mat_rgba['target_highlight', 'a'] = 0
        # Initialize last-target distance for approach reward tracking.
        if self._enable_foraging or self._enable_single_target:
            # forward() so geom_xpos reflects both the randomised pose and any
            # target we just moved, before the distance is read off it.
            physics.forward()
            self._last_target_dist = float(physics.nose_to_target_dist())
        else:
            self._last_target_dist = None
        # print(f"All Body Psotions: {self.all_body_positions(physics)}")
        # print(f"Observation: {self.get_observation(physics)}")

    def all_body_positions(self, physics):
        """World (x, y, z) position of every body in the model, including the head."""
        names = physics.named.data.xpos.axes.row.names
        return names, physics.named.data.xpos[:]
    
    # def worm_positions(physics, n_joints, head_name):
    #     """World (x, y, z) position of just the worm's own segments, excluding
    #     world/target/obstacle bodies. head_name must be confirmed from
    #     all_body_positions' printed names first (e.g. 'head')."""
    #     names = [head_name] + [f'segment_{i}' for i in range(n_joints)]
    #     return np.array([physics.named.data.xpos[n] for n in names])

    def get_observation(self, physics):
        """joints, [to_target], [to_obstacle], body_velocities -- in that fixed order,
        so slices stay predictable regardless of which flags are on."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        if self._enable_foraging or self._enable_single_target:
            obs['to_target'] = physics.nose_to_target()
            # scalar distance to target (head-centric). Keep as a 1-d array so
            # downstream code can slice predictably: [forward, lateral], dist
            try:
                dist = physics.nose_to_target_dist()
            except Exception:
                # fall back if API differs
                vec = physics.nose_to_target()
                dist = np.linalg.norm(vec)
            obs['to_target_dist'] = np.array([dist])
        if self._enable_obstacles:
            vector, _ = physics.nearest_obstacle(self._n_obstacles)
            obs['to_obstacle'] = vector
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        # When foraging or single-target tasks are enabled, we disable
        # the forward-speed reward so the agent focuses on food-related
        # objectives instead of purely maximizing speed.
        if not (self._enable_foraging or self._enable_single_target):
            forward_velocity = -physics.named.data.sensordata['head_vel'][1]
            reward = rewards.tolerance(
                forward_velocity,
                bounds=(self._desired_speed, float('inf')),
                margin=self._desired_speed,
                value_at_margin=0.,
                sigmoid='linear',
            )
        else:
            reward = 0.0

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
            # Dense approach reward: the *signed* change in distance, so it is a
            # proper potential-based shaping term that telescopes to the total
            # distance closed over the episode. Clamping this at 0 (as it was)
            # rewards every decrease but penalises no increase, so a worm that
            # merely oscillates toward and away accumulates reward with zero net
            # progress -- it pays for wiggling rather than for navigating.
            if getattr(self, '_last_target_dist', None) is not None:
                approach = self._last_target_dist - float(dist)
                reward += self._target_reward_weight * approach
            self._last_target_dist = float(dist)

            # Existing proximity-shaped reward (kept for stability).
            reward += self._target_reward_weight * rewards.tolerance(
                dist,
                bounds=(0, target_size),
                margin=5 * target_size,
                sigmoid='long_tail',
            )

            # Arrival bonus and respawn when food reached.
            if dist < target_size:  # worm reached the food -- respawn it
                reward += self._target_reward_weight * 1.0
                self._place_target(physics, around=physics.named.data.geom_xpos['nose'][:2])
                physics.forward()
                # Re-baseline against the *new* target. Without this the next
                # step's signed approach term is (old ~0 distance) - (new far
                # distance): a large penalty for the very act of eating. This was
                # masked while the term was clamped at 0, but not once it is signed.
                self._last_target_dist = float(physics.nose_to_target_dist())

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
    target_distance=None,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance.

    target_distance: pin the target to this distance at a random bearing instead
    of dm_control's random box spawn -- see Swim._place_target. None keeps the
    stock behaviour, which is what a generalisation check should evaluate on.
    """
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        target_distance=target_distance,
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
    target_distance=None,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance.

    target_distance: pin each pellet to this distance at a random bearing (from
    the worm's nose on respawn) instead of dm_control's random box spawn -- see
    Swim._place_target. None keeps the stock behaviour.
    """
    model_string, assets = get_model_and_assets(
        n_links, n_obstacles=n_obstacles if enable_obstacles else 0
    )
    physics = Physics.from_xml_string(model_string, assets=assets)
    task = Swim(
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        target_distance=target_distance,
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
    # physics.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
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
def swim_12_links(
    n_links=12,
    desired_speed=_SWIM_SPEED,
    enable_single_target=False,
    enable_foraging=False,
    enable_obstacles=False,
    n_obstacles=3,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the plain Swim task on a longer, 12-link body."""
    return swim(
        n_links=n_links,
        desired_speed=desired_speed,
        enable_single_target=enable_single_target,
        enable_foraging=enable_foraging,
        enable_obstacles=enable_obstacles,
        n_obstacles=n_obstacles,
        time_limit=time_limit,
        random=random,
        environment_kwargs=environment_kwargs,
    )


# ==================================================================================================
# Choosing a task.
#
# All of the above are the same Swim task with different flags, so a run picks one by
# name. training.run_config / es.run_es turn that choice into the dm_control task name
# tonic loads ('swimmer-<name>'), via env_task/task_env_kwargs below.

# task name -> the extra observation it inserts after 'joints' (None if it adds none).
# The reflexes in macrocircuits.reflex_steering slice exactly that vector, so which one
# a task provides is what decides which reflex fits it (see training._CONTROLLERS).
TASKS = {
    'swim': None,           # swim forward as fast as possible -- the original task
    'swim_to_ball': 'to_target',   # one visible, fixed target to reach
    'foraging': 'to_target',       # food pellets that respawn elsewhere once eaten
    'evasion': 'to_obstacle',      # swim forward while keeping clear of static obstacles
}

# Swimmer body length (rigid links) -> the plain-swimming task registered for it. Only
# 'swim' has a per-length registration; the other tasks take n_links as a task kwarg.
SWIM_TASKS = {6: 'swim', 12: 'swim_12_links'}


def env_task(task='swim', n_links=6):
    """Registered dm_control task name for a (task, body length) choice.

    'swim' has a separate registration per body length ('swim'/'swim_12_links'); every
    other task is registered once and takes its length through task_env_kwargs().
    """
    if task not in TASKS:
        raise ValueError(f'task must be one of {sorted(TASKS)}, got {task!r}')
    if n_links not in SWIM_TASKS:
        raise ValueError(f'n_links must be one of {sorted(SWIM_TASKS)}, got {n_links!r}')
    return SWIM_TASKS[n_links] if task == 'swim' else task


def task_env_kwargs(task='swim', n_links=6, task_kwargs=None):
    """Task kwargs to load `task` with -- the run's own, plus n_links where it is needed.

    Only the non-'swim' tasks need n_links passed explicitly (env_task already encodes
    it for 'swim'), and only when it differs from their default, so a plain run's
    kwargs stay empty and its stored config string is unchanged.
    """
    kwargs = dict(task_kwargs or {})
    if task != 'swim' and n_links != 6:
        kwargs.setdefault('n_links', n_links)
    return kwargs


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