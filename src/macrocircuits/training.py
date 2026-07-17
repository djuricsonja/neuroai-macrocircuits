"""Training agents with tonic, and replaying trained checkpoints as video.

Importing this module requires tonic, so call `ensure_tonic()` first.
"""

import argparse
import inspect
import os

import numpy as np
import tonic
import tonic.torch
import yaml

from macrocircuits.video import display_video

# Imported so that the code strings passed to train() and stored in config.yaml --
# e.g. 'tonic.torch.agents.PPO(model=ppo_mlp_model(...))' -- resolve when eval'd.
# run_config() only builds PPO/A2C/TRPO strings, but the off-policy factories are
# imported too so a hand-written DDPG/D4PG agent string passed to train() resolves.
from macrocircuits.models import (
    d4pg_mlp_model,
    d4pg_swimmer_model,
    ddpg_mlp_model,
    ddpg_swimmer_model,
    ppo_mlp_model,
    ppo_swimmer_model,
)


def _eval_namespace():
    """Builds the namespace the code strings below are evaluated in.

    train() and play_model() take Python source as strings and eval it. In the
    original notebook those strings resolved against the notebook's own globals;
    now that these helpers live in a module, eval would only see this file's
    globals. Merging the caller's globals on top restores the old behaviour, so a
    model defined in a notebook cell still resolves -- and shadows the one here.
    """
    namespace = dict(globals())
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    if caller is not None:
        namespace.update(caller.f_globals)
    return namespace


# RL method -> (tonic agent class, actor-updater class). All three drive the same
# stochastic ActorCritic the model factories build, so they are interchangeable here.
# The actor updater is named so run_config() can turn on gradient clipping for it;
# TRPO's trust-region update already bounds each step, so it has none (None).
_RL_AGENTS = {
    'ppo': ('PPO', 'ClippedRatio'),
    'a2c': ('A2C', 'StochasticPolicyGradient'),
    'trpo': ('TRPO', None),
}

# Swimmer body length (rigid links) -> the dm_control task macrocircuits.envs
# registers on import. Only these two lengths exist.
_SWIM_TASKS = {6: 'swim', 12: 'swim_12_links'}


# Every key a run dict may set, and the value used when it does not set it. A run is
# one training run: a network plus the algorithm, body, sizes and budget to train it
# with. resolve_runs() validates run dicts against these keys, and run_config() takes
# a resolved one as **kwargs -- so the two must stay in sync.
_RUN_DEFAULTS = {
    'rl_method': 'ppo',
    'n_links': 6,
    'actor_sizes': (256, 256),
    'critic_sizes': (256, 256),
    'action_noise': 0.1,
    'gradient_clip': 0.5,
    'steps': int(1e5),
    'save_steps': int(5e4),
    'label': None,
}


def _run_name(network, rl_method, label=None):
    """Directory name for a run: its label, or '<network>_<rl_method>' if unlabelled."""
    return label or f'{network}_{rl_method}'


def resolve_runs(runs, defaults=None):
    """Fill in each run dict and check that no two runs would train into the same directory.

    Lets a notebook declare any number of runs while spelling out only what each one
    varies:

        resolve_runs(
            [dict(network='ncap'), dict(network='mlp', label='mlp_wide')],
            defaults=dict(rl_method='trpo'),
        )

    Values are taken from the run dict first, then `defaults`, then _RUN_DEFAULTS.

    A run's directory is derived from its label (see _run_name), which defaults to
    '<network>_<rl_method>'. Two runs that differ only in, say, critic_sizes would
    therefore share a directory and silently overwrite each other's checkpoints and
    logs, so that case raises and asks for a distinct label instead.

    Returns a list of complete run dicts, each ready to splat into run_config(**run)
    or run_path(**run).
    """
    resolved = []
    seen = {}
    for index, run in enumerate(runs):
        unknown = set(run) - set(_RUN_DEFAULTS) - {'network'}
        if unknown:
            raise ValueError(
                f'runs[{index}] has unknown key(s) {sorted(unknown)}; '
                f'valid keys are {sorted(set(_RUN_DEFAULTS) | {"network"})}'
            )
        config = {**_RUN_DEFAULTS, **(defaults or {}), **run}
        if not config.get('network'):
            raise ValueError(f"runs[{index}] is missing the required 'network' key")
        config['label'] = _run_name(config['network'], config['rl_method'], config['label'])

        # Runs of different body lengths live under different task directories, so a
        # label only has to be unique among runs sharing an n_links.
        key = (config['n_links'], config['label'])
        if key in seen:
            raise ValueError(
                f"runs[{index}] and runs[{seen[key]}] would both train into "
                f"'{run_path(**config)}' and overwrite each other. Give at least one of "
                f"them a distinct label=... (it also names the run in the plot legend)."
            )
        seen[key] = index
        resolved.append(config)
    return resolved


def run_config(
    network,
    rl_method='ppo',
    n_links=6,
    actor_sizes=(256, 256),
    critic_sizes=(256, 256),
    action_noise=0.1,
    gradient_clip=0.5,
    steps=int(1e5),
    save_steps=int(5e4),
    label=None,
):
    """Assemble the code strings train() needs for one (network, algorithm, body) choice.

    train() takes its agent and environment as Python *source strings* and eval's
    them (see _eval_namespace). This turns the notebook's plain parameters into the
    matching factory calls so the notebook itself stays declarative.

    Parameters:
    - network:   'mlp'  -- generic fully-connected baseline, or
                 'ncap' -- the C. elegans-derived circuit prior. NCAP's actor reads
                 the time feature the environment appends, so it is enabled for it.
    - rl_method: on-policy tonic agent driving the stochastic policy: 'ppo', 'a2c', 'trpo'.
    - n_links:   swimmer length. 6 -> 5 joints ('swim'), 12 -> 11 joints ('swim_12_links').
    - actor_sizes/critic_sizes: MLP torso widths. NCAP's actor is the fixed circuit,
                 so actor_sizes is used by the MLP baseline only; critic_sizes applies to both.
    - action_noise: exploration std of the NCAP policy head.
    - gradient_clip: max gradient norm per update (0 disables). The swimmer policy uses
                 a small fixed action std, so without clipping a single large PPO/A2C
                 step can blow the importance ratio up to inf and drive the weights to
                 NaN; capping the step keeps the ratio bounded. Applied to the critic
                 always, and to the actor except for TRPO (already trust-region bounded).
    - steps/save_steps: total env steps to train for, and the checkpoint interval.
    - label:     directory name for this run, and its label in the plot legend. Defaults
                 to '<network>_<rl_method>'; required to tell apart runs that share both
                 (see resolve_runs).

    Returns (agent, environment, name, trainer), four strings ready to pass to train().
    """
    if rl_method not in _RL_AGENTS:
        raise ValueError(f"rl_method must be one of {sorted(_RL_AGENTS)}, got {rl_method!r}")
    if n_links not in _SWIM_TASKS:
        raise ValueError(f"n_links must be one of {sorted(_SWIM_TASKS)}, got {n_links!r}")

    agent_cls, actor_updater_cls = _RL_AGENTS[rl_method]
    task = _SWIM_TASKS[n_links]
    n_joints = n_links - 1

    if network == 'mlp':
        model = f'ppo_mlp_model(actor_sizes={actor_sizes}, critic_sizes={critic_sizes})'
        time_feature = ''
    elif network == 'ncap':
        model = (
            f'ppo_swimmer_model(n_joints={n_joints}, '
            f'critic_sizes={critic_sizes}, action_noise={action_noise})'
        )
        # NCAP's SwimmerActor reads observations[..., -1] as the timestep, which tonic
        # only appends when time_feature=True -- so it is required for this network.
        time_feature = ', time_feature=True'
    else:
        raise ValueError(f"network must be 'mlp' or 'ncap', got {network!r}")

    # Assemble the agent, enabling gradient clipping on its updaters (see the docstring).
    agent_args = [f'model={model}']
    if gradient_clip and actor_updater_cls:
        agent_args.append(
            f'actor_updater=tonic.torch.updaters.{actor_updater_cls}(gradient_clip={gradient_clip})'
        )
    if gradient_clip:
        agent_args.append(
            f'critic_updater=tonic.torch.updaters.VRegression(gradient_clip={gradient_clip})'
        )
    agent = f'tonic.torch.agents.{agent_cls}(' + ', '.join(agent_args) + ')'
    environment = f'tonic.environments.ControlSuite("swimmer-{task}"{time_feature})'
    trainer = f'tonic.Trainer(steps={int(steps)}, save_steps={int(save_steps)})'
    return agent, environment, _run_name(network, rl_method, label), trainer


def run_path(network, rl_method='ppo', n_links=6, label=None, **_run_config_kwargs):
    """Data directory train() writes for this run -- pass to plot_performance / play_model.

    Ignores the remaining run_config keys (sizes, budget, ...) so that a resolved run
    dict can be splatted straight in: run_path(**run).
    """
    if n_links not in _SWIM_TASKS:
        raise ValueError(f"n_links must be one of {sorted(_SWIM_TASKS)}, got {n_links!r}")
    task = _SWIM_TASKS[n_links]
    return f'data/local/experiments/tonic/swimmer-{task}/{_run_name(network, rl_method, label)}'


def train(
    header,
    agent,
    environment,
    name='test',
    trainer='tonic.Trainer()',
    before_training=None,
    after_training=None,
    parallel=1,
    sequential=1,
    seed=0,
):
    """
    Some additional parameters:

    - before_training: Python code to execute immediately before the training loop commences, suitable for setup actions needed after initialization but prior to training.
    - after_training: Python code to run once the training loop concludes, ideal for teardown or analytical purposes.
    - parallel: The count of environments to execute in parallel. Limited to 1 in a Colab notebook, but if additional resources are available, this number can be increased to expedite training.
    - sequential: The number of sequential steps the environment runs before sending observations back to the agent. This setting is useful for temporal batching. It can be disregarded for this tutorial's purposes.
    - seed: The experiment's random seed, guaranteeing the reproducibility of the training process.

    """
    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    namespace = _eval_namespace()

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header, namespace)

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.distribute(
        lambda: eval(_environment, namespace), parallel, sequential
    )
    test_environment = tonic.environments.distribute(lambda: eval(_environment, namespace))

    # Build the agent.
    agent = eval(agent, namespace)
    agent.initialize(
        observation_space=test_environment.observation_space,
        action_space=test_environment.action_space,
        seed=seed,
    )

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f'-{parallel}x{sequential}'

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join('data', 'local', 'experiments', 'tonic', environment_name, name)
    tonic.logger.initialize(path, script_path=None, config=args)

    # Build the trainer.
    trainer = eval(trainer, namespace)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
    )

    # Run some code before training.
    if before_training:
        exec(before_training, namespace)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training, namespace)


def play_model(path, checkpoint='last', environment='default', seed=None, header=None):
    """
    Plays a model within an environment and renders the gameplay to a video.

    Parameters:
    - path (str): Path to the directory containing the model and checkpoints.
    - checkpoint (str): Specifies which checkpoint to use ('last', 'first', or a specific ID). 'none' indicates no checkpoint.
    - environment (str): The environment to use. 'default' uses the environment specified in the configuration file.
    - seed (int): Optional seed for reproducibility.
    - header (str): Optional Python code to execute before initializing the model, such as importing libraries.
    """

    namespace = _eval_namespace()

    if checkpoint == 'none':
        # Use no checkpoint, the agent is freshly created.
        checkpoint_path = None
        tonic.logger.log('Not loading any weights')
    else:
        checkpoint_path = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            tonic.logger.error(f'{checkpoint_path} is not a directory')
            checkpoint_path = None

        # List all the checkpoints.
        checkpoint_ids = []
        for file in os.listdir(checkpoint_path):
            if file[:5] == 'step_':
                checkpoint_id = file.split('.')[0]
                checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
            if checkpoint == 'last':
                # Use the last checkpoint.
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
            elif checkpoint == 'first':
                # Use the first checkpoint.
                checkpoint_id = min(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
            else:
                # Use the specified checkpoint.
                checkpoint_id = int(checkpoint)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
                else:
                    tonic.logger.error(f'Checkpoint {checkpoint_id} not found in {checkpoint_path}')
                    checkpoint_path = None
        else:
            tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
            checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(path, 'config.yaml')
    with open(arguments_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)

    # Run the header first, e.g. to load an ML framework.
    try:
        if config.header:
            exec(config.header, namespace)
        if header:
            exec(header, namespace)
    except:
        pass

    # Build the agent.
    agent = eval(config.agent, namespace)

    # Build the environment.
    if environment == 'default':
        environment = tonic.environments.distribute(lambda: eval(config.environment, namespace))
    else:
        environment = tonic.environments.distribute(lambda: eval(environment, namespace))
    if seed is not None:
        environment.seed(seed)

    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    steps = 0
    test_observations = environment.start()
    frames = [environment.render('rgb_array', camera_id=0, width=640, height=480)[0]]
    score, length = 0, 0

    while True:
        # Select an action.
        actions = agent.test_step(test_observations, steps)
        assert not np.isnan(actions.sum())

        # Take a step in the environment.
        test_observations, infos = environment.step(actions)
        frames.append(environment.render('rgb_array', camera_id=0, width=640, height=480)[0])
        agent.test_update(**infos, steps=steps)

        score += infos['rewards'][0]
        length += 1

        if infos['resets'][0]:
            break

    video_path = os.path.join(path, 'video.mp4')
    print('Reward for the run: ', score)
    return display_video(frames, video_path)
