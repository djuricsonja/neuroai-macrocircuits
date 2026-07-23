"""Shared physics-only evaluation protocol (FORAGING_FORWARD_PLAN.md, Phase 0).

Every metric here is read directly from dm_control ground truth (physics.nose_to_target(),
physics.nose_to_target_dist(), a respawn counter on the task) rather than from any
controller's own internal observation slicing. That's deliberate: reflex_steering.py's
forward/lateral variable labels have been confirmed (by direct simulator test) to be
swapped relative to the true local axes, producing a consistent 90-degree error in the
reflex's own internal steering angle. This module never reads that labeling, so its
success metrics stay correct regardless of what any individual controller gets right or
wrong about its own axes.

Training reward is never used for ranking -- it has repeatedly been misleading in this
environment (see FORAGING_REFLEX_PROGRESS.md). Only these physics metrics decide anything.
"""

import collections
import math
import os

import numpy as np
import tonic
import tonic.torch
import yaml

from macrocircuits import training as _training_mod

NEAR_FOOD_DIST = 0.15  # "near-food" success threshold (episode ever got this close)
TRUE_EAT_DIST = 0.02   # matches Swim's default food_size; ground-truthed via the eat counter

DIST_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, float('inf'))]
ANGLE_BINS = [(-180, -90), (-90, 0), (0, 90), (90, 180)]


def _bin_label(value, bins):
    for lo, hi in bins:
        if lo <= value < hi:
            hi_label = 'inf' if hi == float('inf') else f'{hi:g}'
            return f'[{lo:g},{hi_label})'
    return 'other'


def true_bearing_deg(to_target):
    """Egocentric bearing to food in degrees, 0 = dead ahead, from the RAW nose_to_target().

    Ground truth, verified directly against the simulator (see verify_axis_convention3/4.py
    in the debugging history): to_target[0] is the local LATERAL axis, to_target[1] is local
    longitudinal with true forward = -to_target[1]. This is intentionally independent of
    reflex_steering.py's own (confirmed swapped) forward/lateral variable naming -- do not
    "fix" this to match that file without re-deriving it from the simulator first.
    """
    forward = -to_target[1]
    lateral = to_target[0]
    return float(np.degrees(np.arctan2(lateral, forward)))


def _env_obj(environment):
    """Finds the dm_control control.Environment inside a tonic-wrapped environment --
    the object exposing both .physics and .task -- by walking its .env chain of gym
    wrappers down to whichever one exposes them, directly or via .environment.

    Mirrors training._physics_of's traversal, but returns the object itself (not just
    .physics) since eval also needs .task for the ground-truth eat counter.
    """
    obj = environment.environments[0]
    while obj is not None:
        if hasattr(obj, 'physics') and hasattr(obj, 'task'):
            return obj
        inner = getattr(obj, 'environment', None)
        if inner is not None and hasattr(inner, 'physics') and hasattr(inner, 'task'):
            return inner
        obj = getattr(obj, 'env', None)
    raise AttributeError('Could not find a dm_control Environment (.physics + .task) inside this environment')


def _rollout_episode(agent, environment, dm_env):
    physics, task = dm_env.physics, dm_env.task
    obs = environment.start()
    to_target0 = physics.nose_to_target().copy()
    start_dist = float(np.linalg.norm(to_target0))
    start_angle = true_bearing_deg(to_target0)
    min_dist = start_dist
    score = 0.0
    step = 0
    while True:
        actions = agent.test_step(obs, step)
        obs, infos = environment.step(actions)
        step += 1
        dist = float(physics.nose_to_target_dist())
        min_dist = min(min_dist, dist)
        score += float(infos['rewards'][0])
        if infos['resets'][0]:
            break
    n_eaten = task._episode_n_eaten
    return dict(
        start_dist=start_dist,
        start_angle=start_angle,
        dist_bin=_bin_label(start_dist, DIST_BINS),
        angle_bin=_bin_label(start_angle, ANGLE_BINS),
        min_dist=min_dist,
        near_food=min_dist < NEAR_FOOD_DIST,
        n_eaten=n_eaten,
        true_eat=n_eaten > 0,
        score=score,
        length=step,
    )


def _build_agent_and_env(run, checkpoint, eval_task_kwargs):
    """Loads a trained checkpoint's agent, plus a fresh single test environment built
    from `run`'s own config but with task_kwargs overridden by `eval_task_kwargs` --
    e.g. training under target_distance=0.8 but evaluating at target_distance=None
    (stock spawn) to check generalisation, per Phase 0/1 of the forward plan.
    """
    path = _training_mod.run_path(**run)
    agent_str, _, _, _ = _training_mod.run_config(**run)

    base_kwargs = dict(run.get('task_kwargs') or {})
    base_kwargs.update(eval_task_kwargs or {})
    eval_run = {**run, 'task_kwargs': base_kwargs}
    _, env_str, _, _ = _training_mod.run_config(**eval_run)

    namespace = dict(vars(_training_mod))
    environment = tonic.environments.distribute(lambda: eval(env_str, namespace))
    agent = eval(agent_str, namespace)
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=0,
    )
    if run['method'] != 'es':
        checkpoint_path = os.path.join(path, 'checkpoints', checkpoint)
        agent.load(checkpoint_path)
    return agent, environment


def evaluate_arm(
    run,
    checkpoint='step_20000',
    eval_task_kwargs=None,
    seeds=(0, 1, 2, 3, 4),
    episodes_per_seed=100,
):
    """Runs the shared Phase 0 protocol for one trained arm (a resolved `run` dict).

    Returns a list of per-episode dict rows (one per seed x episode), each carrying
    start_dist/start_angle/dist_bin/angle_bin/min_dist/near_food/n_eaten/true_eat/
    score/length -- ready to aggregate with summarize() or dump straight to CSV.

    Re-instantiates the agent+environment fresh per seed (rather than one long roll of
    episodes) so that seeds are independently reproducible: task_kwargs=dict(random=seed)
    seeds the dm_control task's own RandomState once at construction, and every reset
    after that draws further samples from that same stream -- so `episodes_per_seed`
    resets under one seed already gives a diverse, reproducible batch of episodes.
    """
    rows = []
    for seed in seeds:
        seed_task_kwargs = dict(eval_task_kwargs or {})
        seed_task_kwargs['random'] = seed
        agent, environment = _build_agent_and_env(run, checkpoint, seed_task_kwargs)
        dm_env = _env_obj(environment)
        for _ in range(episodes_per_seed):
            row = _rollout_episode(agent, environment, dm_env)
            row['seed'] = seed
            rows.append(row)
    return rows


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else float('nan')
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float('nan')
    return mean, 1.96 * se, n


def summarize(rows, group_by=None):
    """Aggregates rows into mean +/- 95% CI for near_food and true_eat rates.

    group_by=None gives one overall row; group_by='dist_bin' or 'angle_bin' gives a
    stratified table (Phase 0 step 2), one row per bin, in the bin's natural order.
    """
    if group_by is None:
        groups = {'overall': rows}
        order = ['overall']
    else:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[group_by]].append(row)
        bins = DIST_BINS if group_by == 'dist_bin' else ANGLE_BINS
        order = [_bin_label((lo + hi) / 2 if hi != float('inf') else lo + 1, bins) for lo, hi in bins]
        order = [label for label in order if label in groups]

    summary = []
    for key in order:
        group_rows = groups[key]
        near_mean, near_ci, n = _mean_ci([r['near_food'] for r in group_rows])
        eat_mean, eat_ci, _ = _mean_ci([r['true_eat'] for r in group_rows])
        summary.append(dict(
            group=key, n=n,
            near_food_rate=near_mean, near_food_ci=near_ci,
            true_eat_rate=eat_mean, true_eat_ci=eat_ci,
        ))
    return summary


def print_summary(label, rows, group_by=None):
    print(f'--- {label} ({group_by or "overall"}) ---')
    for row in summarize(rows, group_by=group_by):
        print(
            f"  {row['group']:<14} n={row['n']:<4} "
            f"near_food={row['near_food_rate']*100:5.1f}% +/- {row['near_food_ci']*100:4.1f}  "
            f"true_eat={row['true_eat_rate']*100:5.1f}% +/- {row['true_eat_ci']*100:4.1f}"
        )
