"""Final comprehensive training run for the NCAP foraging/obstacle-avoidance reflexes.

Matches the NCAP paper's own RL hyperparameters (Appendix A.4): 5e6 timesteps per run,
8-core parallelism, 10 random seeds per condition (the paper uses 10 seeds for its
reported error bars / bootstrap confidence intervals, e.g. Figure 4).

This is NOT meant to run on a laptop. At this project's own measured single-environment
throughput (~50-85 steps/sec), 5e6 steps would take 17-30 hours *per run*; parallel=8
cuts that substantially but this is still a genuinely large amount of compute --
4 arms x 10 seeds = 40 runs total. Built to run unattended (e.g. overnight or longer)
on a multi-core / GPU machine, not to be watched.

Safe to interrupt and resume: is_trained() skips any run already completed with
matching parameters (checkpoint + config.yaml already on disk), so re-running this
script picks up wherever it left off rather than retraining everything from scratch.

PARALLEL=8 requires Linux (or another fork()-based OS). Verified two real bugs in the
training pipeline while setting this up:
  1. tonic's own Trainer never called environment.initialize(seed) before running --
     harmless for parallel=1 (Sequential doesn't need that call to work), but Parallel
     silently never spawned its worker processes without it. Fixed in this project's
     own training.train() (not in Ressources/tonic/, which is gitignored and re-cloned
     fresh by ensure_tonic() -- a fix there would never reach anyone else who runs
     this).
  2. On Windows specifically, multiprocessing has to pickle each worker's target
     function to spawn it, and tonic's worker (Parallel.initialize's local `proc`
     function) is a nested closure, which Python cannot pickle on Windows -- confirmed
     directly (AttributeError: Can't get local object 'Parallel.initialize.<locals>.proc').
     This is a Windows-only limitation (Linux forks instead of pickling, so the same
     code works there), not fixed here. If parallel=8 fails with that exact error,
     you're on Windows -- either run this under WSL/a Linux box, or drop PARALLEL to 1
     as a fallback (much slower: single-environment runs on this project's own
     hardware measured ~50-85 steps/sec, so 5e6 steps would take 17-30 hours *each*).

Run from anywhere: `python train_final.py` (this script chdir's itself into Ressources/,
matching every other training script in this project, since run_path()'s returned paths
are relative to that directory).
"""

import os
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, SRC)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Ressources'))

from macrocircuits import ensure_tonic
ensure_tonic()
import tonic.torch
from macrocircuits.training import is_trained, resolve_runs, run_config, run_path, train

STEPS = int(5e6)
SAVE_STEPS = int(5e5)
PARALLEL = 8  # matches the paper's 8-core RL training setup
SEEDS = list(range(10))  # matches the paper's 10 random seeds

# Team default reward (FORAGING_FORWARD_PLAN.md, Phase 2 decision): Luka's signed
# progress, not alignment -- alignment_reward_weight's own formula was found to have
# the same forward/lateral axis-swap bug as the old reflex (it rewards facing
# *sideways* at the food, not at it -- see FORAGING_REFLEX_PROGRESS.md, "what is
# alignment reward") and was never re-derived to fix that, so it's not used here.
# progress_reward_weight=20.0 is the value already settled in this project's own
# Phase 7 sweep (43% physics-only success on the old, buggy reflex) -- reused as the
# best available default rather than re-sweeping blind against the corrected reflexes.
PROGRESS_WEIGHT = 20.0

DEFAULTS = dict(method='ppo', n_links=6, action_noise=0.3, steps=STEPS, save_steps=SAVE_STEPS)

# Arms to train. This is my own default proposal, not yet confirmed -- floor (no
# steering, the reference), steer_to_food and avoid_obstacle in isolation (each
# reflex's own solo task), and forage_and_avoid combined. No MLP: this project's
# focus is the NCAP-architecture story (see FORAGING_FORWARD_PLAN.md's goal -- "a
# simple innate architecture is enough... not a large learned controller"). Adjust
# this list directly once the final arm selection is confirmed; everything below it
# (the seed loop, training loop) does not need to change.
ARM_SPECS = [
    dict(
        label='ncap_floor', task='foraging', controller=None,
        task_kwargs=dict(speed_reward_weight=0.0, progress_reward_weight=PROGRESS_WEIGHT),
    ),
    dict(
        label='ncap_steer_to_food', task='foraging', controller='steer_to_food',
        task_kwargs=dict(speed_reward_weight=0.0, progress_reward_weight=PROGRESS_WEIGHT),
    ),
    dict(
        # evasion()'s factory doesn't forward progress/speed reward kwargs (only
        # foraging() does) -- this arm uses evasion's own defaults (obstacle penalty
        # + the stock forward-swim reward), there being no food on this task at all.
        label='ncap_avoid_obstacle', task='evasion', controller='avoid_obstacle',
        task_kwargs=dict(n_obstacles=3),
    ),
    dict(
        label='ncap_forage_and_avoid', task='foraging', controller='forage_and_avoid',
        task_kwargs=dict(
            speed_reward_weight=0.0, progress_reward_weight=PROGRESS_WEIGHT,
            enable_obstacles=True, n_obstacles=3,
        ),
    ),
]

RUNS = []
for spec in ARM_SPECS:
    for seed in SEEDS:
        RUNS.append(dict(
            network='ncap', task=spec['task'], controller=spec['controller'],
            task_kwargs=spec['task_kwargs'], seed=seed,
            label=f'{spec["label"]}_seed{seed}',
        ))
RUNS = resolve_runs(RUNS, defaults=DEFAULTS)

print(
    f'{len(RUNS)} total runs ({len(ARM_SPECS)} arms x {len(SEEDS)} seeds), '
    f'{STEPS:,} steps each, parallel={PARALLEL}',
    flush=True,
)
for run in RUNS:
    print(f'  {run["label"]:<32} {run["task"]:<9} seed={run["seed"]}', flush=True)

start_time = time.time()
for i, run in enumerate(RUNS):
    path = run_path(**run)
    agent, environment, name, trainer = run_config(**run)
    if is_trained(path, agent, environment, trainer, seed=run['seed']):
        print(f'[{i + 1}/{len(RUNS)}] Skipping {name}: already trained at {path}', flush=True)
        continue
    print(
        f'[{i + 1}/{len(RUNS)}] Training {name} '
        f'({run["steps"]:,} steps, seed={run["seed"]}) ===',
        flush=True,
    )
    run_start = time.time()
    train(
        'import tonic.torch', agent, environment, name=name, trainer=trainer,
        parallel=PARALLEL, seed=run['seed'],
    )
    print(
        f'[{i + 1}/{len(RUNS)}] Done: {name} ({time.time() - run_start:.0f}s)',
        flush=True,
    )

print(f'=== All runs complete ({time.time() - start_time:.0f}s total) ===', flush=True)
