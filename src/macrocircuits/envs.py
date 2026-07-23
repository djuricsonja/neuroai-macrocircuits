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
        speed_reward_weight=1.0,
        target_reward_weight=1.0,
        progress_reward_weight=0.0,
        alignment_reward_weight=0.0,
        alignment_gated_progress_weight=0.0,
        velocity_alignment_reward_weight=0.0,
        velocity_alignment_ema_reward_weight=0.0,
        velocity_alignment_ema_alpha=0.02,
        eaten_bonus=0.0,
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
        self._speed_reward_weight = speed_reward_weight
        self._target_reward_weight = target_reward_weight
        self._progress_reward_weight = progress_reward_weight
        self._alignment_reward_weight = alignment_reward_weight
        self._alignment_gated_progress_weight = alignment_gated_progress_weight
        self._velocity_alignment_reward_weight = velocity_alignment_reward_weight
        self._velocity_alignment_ema_reward_weight = velocity_alignment_ema_reward_weight
        self._velocity_alignment_ema_alpha = velocity_alignment_ema_alpha
        self._velocity_alignment_ema = None
        self._eaten_bonus = eaten_bonus
        self._obstacle_penalty_weight = obstacle_penalty_weight
        self._obstacle_safe_distance = obstacle_safe_distance
        self._obstacle_min_distance = obstacle_min_distance
        self._food_size = food_size
        self._target_distance = target_distance
        self._prev_target_dist = None

    def _place_target(self, physics, around=None):
        """Reposition the target; `around` is the (x, y) to place it relative to.

        With `target_distance` set, the target goes at exactly that distance and a
        uniformly random bearing -- around the origin at episode start, around the
        worm's nose when a pellet respawns. Fixing the distance makes every episode
        a comparable navigation problem: dm_control's stock spawn samples a wide box,
        so steering is irrelevant when the target lands almost on top of the worm and
        near-hopeless when it lands far away (ported from Luka's
        foraging_with_neural_reuse branch). None keeps the stock behaviour, which is
        what a generalisation check should evaluate on.
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

        self._prev_target_dist = None  # fresh episode -- no prior-step distance yet
        self._velocity_alignment_ema = None  # fresh episode -- no smoothing history yet
        # Ground-truth eat counter for eval (Phase 0 shared protocol). Not part of the
        # observation/reward contract -- a respawn moves the target before an eval loop
        # reading physics *after* env.step() returns can see the pre-respawn distance, so
        # "true eat" (dist < food_size) can't be reconstructed from outside afterwards.
        self._episode_n_eaten = 0

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
        if self._enable_obstacles:
            vector, _ = physics.nearest_obstacle(self._n_obstacles)
            obs['to_obstacle'] = vector
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        reward = self._speed_reward_weight * rewards.tolerance(
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

            # Progress reward: distance closed since the previous step, positive when
            # getting closer. Unlike the tolerance term above (near-zero unless
            # already close, margin is only 5x the tiny food size), this gives dense,
            # continuous feedback throughout an approach -- classic potential-based
            # reward shaping (Ng, Harada & Russell 1999), which doesn't change the
            # optimal policy, just how easy it is to find via gradient-based
            # training. Skipped on the very first step of an episode (no prior
            # distance yet) and right after a respawn (see below), so a fresh/new
            # target's sudden distance jump is never scored as a huge, spurious
            # negative "regression".
            progress = None
            if self._prev_target_dist is not None:
                progress = self._prev_target_dist - dist
            if self._progress_reward_weight and progress is not None:
                reward += self._progress_reward_weight * progress

            # Alignment reward: reward facing the food directly (egocentric forward
            # component of the unit vector to it), not just being near it -- distance
            # reward alone never explicitly rewards good heading, it only rewards
            # outcomes that a good heading tends to produce eventually.
            alignment = None
            if dist > 1e-6:
                to_target = physics.nose_to_target()
                alignment = to_target[0] / dist  # cos(angle to target), egocentric
            if self._alignment_reward_weight and alignment is not None:
                reward += self._alignment_reward_weight * alignment

            # Alignment-gated progress: instead of adding progress and alignment as two
            # independent terms (tried, and stacked *worse* than either alone -- see
            # foraging_reflex_debugging_saga memory), scale progress by how aligned the
            # worm currently is (clamped to [0, inf), so a badly-misaligned step earns
            # ~0 credit for it instead of full credit). Targets the specific failure
            # mode the additive combination didn't: distance closed by incidental drift
            # while facing the wrong way (real earlier in this project -- the "close
            # starts succeed via lucky wiggle" confound) shouldn't get the same credit
            # as distance closed while actually pointed at the food.
            if self._alignment_gated_progress_weight and progress is not None and alignment is not None:
                reward += self._alignment_gated_progress_weight * progress * max(alignment, 0.0)

            # Velocity-alignment reward: reward the head's own *velocity direction*
            # pointing at the food, rather than its nose orientation (what
            # `alignment` above measures). These are not the same thing -- the
            # debugging saga confirmed the head/nose sweeps 30-60 degrees every
            # stroke from gait wobble alone even when net heading is fine (an
            # undulating swimmer's own gait, not the reflex), while what actually
            # determines success is net *displacement* direction, confirmed
            # separately and repeatedly. Tested alone (weight=0.5): 33% physics-only
            # success, a real improvement over the 23% baseline but below plain
            # nose-alignment's 47% -- likely because velocity, being a derivative-like
            # quantity, is *more* exposed to gait-stroke noise than pose is, the same
            # reason the original D-term attempt with omega failed (Step 9), not less
            # as first assumed.
            velocity_alignment = None
            if (self._velocity_alignment_reward_weight or self._velocity_alignment_ema_reward_weight) and dist > 1e-6:
                head_vel = physics.body_velocities()[:2]  # head's own local (vx, vy)
                speed = np.linalg.norm(head_vel)
                if speed > 1e-6:
                    velocity_alignment = np.dot(to_target[:2], head_vel) / (dist * speed)
            if self._velocity_alignment_reward_weight and velocity_alignment is not None:
                reward += self._velocity_alignment_reward_weight * velocity_alignment

            # EMA-smoothed velocity-alignment reward: same signal as above, but
            # averaged over recent steps instead of used raw, to filter out the
            # gait-stroke-frequency noise directly rather than reward an
            # instantaneous, wobble-corrupted sample of it. Safe to keep state for
            # here (unlike inside the policy/reflex): this runs once per real step
            # during rollout collection, before anything reaches PPO's shuffled
            # minibatches, so step-order-dependent state is not a problem the way it
            # would be for a stateful reflex (see LearnableAdaptiveForagingReflex's
            # docstring, or the phase-gate discussion in reflex_steering.py).
            # alpha ~= 1/oscillator_period keeps the EMA's effective averaging window
            # comparable to one full gait cycle (default oscillator_period=60), so the
            # cycle's own alternating push/pull on velocity direction has a chance to
            # cancel out rather than dominate the signal.
            if self._velocity_alignment_ema_reward_weight and velocity_alignment is not None:
                if self._velocity_alignment_ema is None:
                    self._velocity_alignment_ema = velocity_alignment
                else:
                    alpha = self._velocity_alignment_ema_alpha
                    self._velocity_alignment_ema = (
                        alpha * velocity_alignment + (1 - alpha) * self._velocity_alignment_ema
                    )
                reward += self._velocity_alignment_ema_reward_weight * self._velocity_alignment_ema

            self._prev_target_dist = dist

            if dist < target_size:  # worm reached the food -- respawn it
                reward += self._eaten_bonus
                self._episode_n_eaten += 1
                # With target_distance set, respawn near the nose (keeps the next
                # pellet local instead of dropping it across the arena); with it
                # unset, falls back to the stock uniform-box respawn as before.
                self._place_target(physics, around=physics.named.data.geom_xpos['nose'][:2])
                physics.forward()
                # Re-measure against the newly-spawned target so next step's progress
                # reward reflects real motion relative to it, not the jump caused by
                # this respawn (which would otherwise look like the worm suddenly
                # moved miles away from a "new" prior distance of ~0).
                self._prev_target_dist = physics.nose_to_target_dist()

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
    speed_reward_weight=0.0,
    progress_reward_weight=0.0,
    alignment_reward_weight=0.0,
    alignment_gated_progress_weight=0.0,
    velocity_alignment_reward_weight=0.0,
    velocity_alignment_ema_reward_weight=0.0,
    velocity_alignment_ema_alpha=0.02,
    eaten_bonus=0.0,
    target_distance=None,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task, with optional foraging and obstacle avoidance.

    speed_reward_weight defaults to 0 here (unlike the base Swim/swim task, where it
    defaults to 1): this isolates the food-seeking reward so it's possible to check
    whether a controller is actually approaching food, without swim-speed reward
    masking the answer. Pass speed_reward_weight=1.0 to restore the normal combined
    reward.

    progress_reward_weight, alignment_reward_weight, alignment_gated_progress_weight,
    velocity_alignment_reward_weight, velocity_alignment_ema_reward_weight, eaten_bonus
    all default to 0 (opt-in, same reasoning as speed_reward_weight -- don't change
    behavior for existing configs). See Swim.get_reward for what each one actually
    computes.

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
        speed_reward_weight=speed_reward_weight,
        progress_reward_weight=progress_reward_weight,
        alignment_reward_weight=alignment_reward_weight,
        alignment_gated_progress_weight=alignment_gated_progress_weight,
        velocity_alignment_reward_weight=velocity_alignment_reward_weight,
        velocity_alignment_ema_reward_weight=velocity_alignment_ema_reward_weight,
        velocity_alignment_ema_alpha=velocity_alignment_ema_alpha,
        eaten_bonus=eaten_bonus,
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