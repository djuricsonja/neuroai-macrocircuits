"""Training agents with tonic, and replaying trained checkpoints as video.

Importing this module requires tonic, so call `ensure_tonic()` first.
"""

import argparse
import collections
import inspect
import os

import numpy as np
import tonic
import tonic.torch
import yaml

from macrocircuits.envs import TASKS as _TASKS, env_task, task_env_kwargs
from macrocircuits.es import es_run_path
from macrocircuits.video import display_video

# Imported for the same reason as the model factories below: run_config() can name one
# of these inside the agent string, so it has to resolve when that string is eval'd.
from macrocircuits.reflex_steering import (
    make_foraging_reflex,
    make_obstacle_avoidance_reflex,
)

# Imported so that the code strings passed to train() and stored in config.yaml --
# e.g. 'tonic.torch.agents.PPO(model=ppo_mlp_model(...))' -- resolve when eval'd.
# Every factory run_config() can name is here; see _RL_AGENTS.
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


# How run_config() assembles each RL method's tonic agent. The off-policy pair differs
# from the on-policy three in more than the agent class, so every part is tabulated
# here rather than hardcoded below:
#
#   agent      -- the tonic.torch.agents class.
#   actor      -- actor updater, named so gradient clipping can be turned on for it.
#                 None means the method already bounds its own step (TRPO's trust
#                 region), so it takes no clipping.
#   critic     -- critic updater. The value objective is method-specific: regression
#                 onto returns for the on-policy three, Q-learning for DDPG, and a
#                 distributional Q for D4PG.
#   models     -- prefix of the factory pair in models.py ('<prefix>_mlp_model' and
#                 '<prefix>_swimmer_model'). Selects the actor head and critic the
#                 agent expects: a stochastic ActorCritic for the on-policy three, an
#                 ActorCriticWithTargets with a deterministic head for DDPG/D4PG.
#   off_policy -- the actor is deterministic, so the agent explores with an injected
#                 noise strategy and `action_noise` sets *its* scale rather than the
#                 std of a stochastic policy head.
_RLMethod = collections.namedtuple('_RLMethod', 'agent actor critic models off_policy')

_RL_AGENTS = {
    'ppo': _RLMethod('PPO', 'ClippedRatio', 'VRegression', 'ppo', False),
    'a2c': _RLMethod('A2C', 'StochasticPolicyGradient', 'VRegression', 'ppo', False),
    'trpo': _RLMethod('TRPO', None, 'VRegression', 'ppo', False),
    'ddpg': _RLMethod(
        'DDPG', 'DeterministicPolicyGradient', 'DeterministicQLearning', 'ddpg', True
    ),
    'd4pg': _RLMethod(
        'D4PG', 'DistributionalDeterministicPolicyGradient',
        'DistributionalDeterministicQLearning', 'd4pg', True
    ),
}

# Every training method a run may choose: the tonic RL agents above, plus 'es' --
# Evolution Strategies, which is not RL at all (no critic, no gradients, no tonic) and
# so is trained by macrocircuits.es.run_es rather than train(). See es_config().
_METHODS = (*_RL_AGENTS, 'es')

# Non-learned steering reflexes (macrocircuits.reflex_steering) a run may plug into
# NCAP's turn inputs, instead of learning that mapping with a network. Each maps the
# run's `controller` name onto the factory run_config() names in the agent string, and
# the tasks whose observation layout that reflex slices (see envs.TASKS).
_CONTROLLERS = {
    'foraging': ('make_foraging_reflex', ('foraging', 'swim_to_ball')),
    'obstacle_avoidance': ('make_obstacle_avoidance_reflex', ('evasion',)),
}


# Every key a run dict may set, and the value used when it does not set it. A run is
# one training run: a network plus the algorithm, body, sizes and budget to train it
# with. resolve_runs() validates run dicts against these keys; run_config() (RL) and
# es.es_config() (ES) each take a resolved one as **kwargs and ignore the keys that
# belong to the other -- so all three must stay in sync.
_RUN_DEFAULTS = {
    # -- any method --
    'method': 'ppo',
    'n_links': 6,
    'seed': 0,
    'task': 'swim',  # which environment to train in; see envs.TASKS.
    'task_kwargs': None,  # dm_control task options, e.g. dict(n_obstacles=10).
    'swimmer_kwargs': None,  # NCAP circuit options, e.g. dict(oscillator_period=60).
    'label': None,
    # -- tonic RL methods only ('ppo', 'a2c', 'trpo', 'ddpg', 'd4pg') --
    'controller': None,  # non-learned steering reflex for NCAP; see _CONTROLLERS.
    'actor_sizes': (256, 256),
    'critic_sizes': (256, 256),
    'action_noise': 0.1,
    'gradient_clip': 0.5,
    'steps': int(1e5),
    'epoch_steps': None,  # env steps between test+log points; None -> auto (see run_config).
    'save_steps': int(5e4),
    # -- 'es' only --
    'generations': 100,
    'population_size': 64,
    'sigma': 0.02,
    'lr': 0.02,
    'weight_decay': 0.0,
    'n_evals': 1,
    'hidden_sizes': (64, 64),
}

# Keys each family ignores. resolve_runs() rejects a run that sets one its method will
# never read, so a knob that would silently do nothing fails loudly instead.
_RL_ONLY_KEYS = frozenset({
    'controller', 'actor_sizes', 'critic_sizes', 'action_noise', 'gradient_clip',
    'steps', 'epoch_steps', 'save_steps',
})
_ES_ONLY_KEYS = frozenset({
    'generations', 'population_size', 'sigma', 'lr', 'weight_decay', 'n_evals', 'hidden_sizes'
})


def _run_name(network, method, label=None):
    """Directory name for a run: its label, or '<network>_<method>' if unlabelled."""
    return label or f'{network}_{method}'


def resolve_runs(runs, defaults=None):
    """Fill in each run dict and check that no two runs would train into the same directory.

    Lets a notebook declare any number of runs while spelling out only what each one
    varies:

        resolve_runs(
            [dict(network='ncap'), dict(network='mlp', label='mlp_wide')],
            defaults=dict(method='trpo'),
        )

    Values are taken from the run dict first, then `defaults`, then _RUN_DEFAULTS.

    A run's directory is derived from its label (see _run_name), which defaults to
    '<network>_<method>'. Two runs that differ only in, say, critic_sizes would
    therefore share a directory and silently overwrite each other's checkpoints and
    logs, so that case raises and asks for a distinct label instead.

    A run is also rejected for setting a key its method ignores -- `sigma` on a PPO
    run, `gradient_clip` on an ES one -- since that knob would otherwise do nothing.
    Only the run's own keys are checked, not `defaults`, so shared defaults can carry
    keys that some methods use and others do not.

    Returns a list of complete run dicts, each ready to splat into run_config(**run),
    es.es_config(**run) or run_path(**run).
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

        method = config['method']
        if method not in _METHODS:
            raise ValueError(
                f'runs[{index}] has method={method!r}; must be one of {sorted(_METHODS)}'
            )

        # Reject knobs the chosen method never reads (see _RL_ONLY_KEYS/_ES_ONLY_KEYS).
        ignored = set(run) & (_RL_ONLY_KEYS if method == 'es' else _ES_ONLY_KEYS)
        if ignored:
            # An ES run was rejected for setting RL-only keys, and vice versa.
            family = 'the tonic RL methods' if method == 'es' else 'Evolution Strategies'
            raise ValueError(
                f'runs[{index}] sets {sorted(ignored)}, which method={method!r} ignores '
                f'(those keys only apply to {family}).'
            )

        if config['task'] not in _TASKS:
            raise ValueError(
                f"runs[{index}] has task={config['task']!r}; must be one of {sorted(_TASKS)}"
            )

        config['label'] = _run_name(config['network'], method, config['label'])

        # Runs on different tasks (or body lengths) already live under different
        # directories, so a label only has to be unique among runs that would share one.
        key = run_path(**config)
        if key in seen:
            raise ValueError(
                f"runs[{index}] and runs[{seen[key]}] would both train into "
                f"'{key}' and overwrite each other. Give at least one of "
                f"them a distinct label=... (it also names the run in the plot legend)."
            )
        seen[key] = index
        resolved.append(config)
    return resolved


def run_config(
    network,
    method='ppo',
    n_links=6,
    task='swim',
    task_kwargs=None,
    controller=None,
    actor_sizes=(256, 256),
    critic_sizes=(256, 256),
    action_noise=0.1,
    gradient_clip=0.5,
    steps=int(1e5),
    epoch_steps=None,
    save_steps=int(5e4),
    swimmer_kwargs=None,
    label=None,
    **_es_kwargs,
):
    """Assemble the code strings train() needs for one (network, algorithm, body) choice.

    train() takes its agent and environment as Python *source strings* and eval's
    them (see _eval_namespace). This turns the notebook's plain parameters into the
    matching factory calls so the notebook itself stays declarative.

    Parameters:
    - network:   'mlp'  -- generic fully-connected baseline, or
                 'ncap' -- the C. elegans-derived circuit prior. NCAP's actor reads
                 the time feature the environment appends, so it is enabled for it.
    - method:    tonic RL agent to train with (see _RL_AGENTS):
                 on-policy, stochastic policy   -- 'ppo', 'a2c', 'trpo'
                 off-policy, deterministic policy + replay buffer -- 'ddpg', 'd4pg'
                 The two families need different actor heads and critics, so the model
                 factory is chosen to match. 'es' is not an RL method and belongs to
                 es.run_es; passing it here raises.
    - n_links:   swimmer length. 6 -> 5 joints ('swim'), 12 -> 11 joints ('swim_12_links').
    - task:      which environment to train in (see envs.TASKS): 'swim' (forward
                 swimming, the original task), 'swim_to_ball' (reach one visible
                 target), 'foraging' (chase respawning food) or 'evasion' (swim forward
                 while avoiding static obstacles). The latter three add a to_target /
                 to_obstacle vector to the observation, which `controller` reads.
    - task_kwargs: options forwarded to the dm_control task, e.g. dict(n_obstacles=10)
                 on 'evasion' or dict(time_limit=60).
    - controller: non-learned steering reflex plugged into NCAP's turn inputs
                 (network='ncap' only), instead of learning that mapping:
                 'obstacle_avoidance' on 'evasion', 'foraging' on the target tasks.
                 None leaves the circuit unsteered, as in the paper. See _CONTROLLERS
                 and macrocircuits.reflex_steering.
    - actor_sizes/critic_sizes: MLP torso widths. NCAP's actor is the fixed circuit,
                 so actor_sizes is used by the MLP baseline only; critic_sizes applies to both.
    - action_noise: exploration std. For the on-policy methods it is the std of the NCAP
                 policy head; for DDPG/D4PG, whose actor is deterministic, it is the
                 scale of the agent's NormalActionNoise exploration instead.
    - gradient_clip: max gradient norm per update (0 disables). The swimmer policy uses
                 a small fixed action std, so without clipping a single large PPO/A2C
                 step can blow the importance ratio up to inf and drive the weights to
                 NaN; capping the step keeps the ratio bounded. Applied to the critic
                 always, and to the actor except for TRPO (already trust-region bounded).
    - steps/save_steps: total env steps to train for, and the checkpoint interval.
    - epoch_steps: env steps between the trainer's test+log points. tonic writes a
                 log.csv row (and runs a test episode) only at these epoch boundaries,
                 so a run shorter than one epoch produces no log.csv at all. Left None
                 it auto-picks min(20000, steps // 5): tonic's 20,000-step cadence on
                 long runs, but ~5 log points on the short demo runs so they still log.
    - swimmer_kwargs: options forwarded to NCAP's SwimmerModule (network='ncap' only) --
                 e.g. dict(oscillator_period=60, use_weight_sharing=False). The
                 use_weight_* flags are the ablations the paper reports.
    - label:     directory name for this run, and its label in the plot legend. Defaults
                 to '<network>_<method>'; required to tell apart runs that share both
                 (see resolve_runs).
    - **_es_kwargs: ignored, so a resolved run dict carrying ES-only keys can be
                 splatted straight in: run_config(**run).

    Returns (agent, environment, name, trainer), four strings ready to pass to train().
    """
    if method == 'es':
        raise ValueError(
            "method='es' is not a tonic RL agent and train() cannot run it; "
            'use macrocircuits.es.run_es(**es_config(**run)) instead.'
        )
    if method not in _RL_AGENTS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")

    rl = _RL_AGENTS[method]
    # Raises on an unknown task or body length, so do it before anything is assembled.
    env_name = env_task(task, n_links)
    env_kwargs = task_env_kwargs(task, n_links, task_kwargs)
    n_joints = n_links - 1

    if controller is not None:
        if controller not in _CONTROLLERS:
            raise ValueError(
                f'controller must be None or one of {sorted(_CONTROLLERS)}, got {controller!r}'
            )
        if network != 'ncap':
            raise ValueError(
                f'controller={controller!r} steers NCAP\'s turn inputs, which the '
                f"{network!r} baseline does not have; drop it or use network='ncap'."
            )
        reflex, reflex_tasks = _CONTROLLERS[controller]
        if task not in reflex_tasks:
            raise ValueError(
                f'controller={controller!r} reads the {_TASKS[reflex_tasks[0]]!r} vector '
                f'that {sorted(reflex_tasks)} add to the observation, but this run is on '
                f'task={task!r}.'
            )

    # Extra SwimmerModule options, spelled out inside the factory call (NCAP only).
    swimmer_args = ''.join(f', {key}={value!r}' for key, value in (swimmer_kwargs or {}).items())

    if network == 'mlp':
        model = f'{rl.models}_mlp_model(actor_sizes={actor_sizes}, critic_sizes={critic_sizes})'
        time_feature = ''
    elif network == 'ncap':
        model_args = [f'n_joints={n_joints}', f'critic_sizes={critic_sizes}']
        if not rl.off_policy:
            # Only the stochastic head takes a fixed action std. DDPG/D4PG wrap the
            # deterministic actor in an exploration strategy instead (see below).
            model_args.append(f'action_noise={action_noise}')
        if controller is not None:
            # The reflex turns the task's egocentric to_target/to_obstacle vector into
            # NCAP's turn signals; the circuit's own bneuron_turn weight, still learned,
            # decides how strongly to act on them.
            model_args.append(f'controller={_CONTROLLERS[controller][0]}({n_joints})')
        model = f'{rl.models}_swimmer_model(' + ', '.join(model_args) + swimmer_args + ')'
        # NCAP's SwimmerActor reads observations[..., -1] as the timestep, which tonic
        # only appends when time_feature=True -- so it is required for this network.
        time_feature = ', time_feature=True'
    else:
        raise ValueError(f"network must be 'mlp' or 'ncap', got {network!r}")

    # Assemble the agent, enabling gradient clipping on its updaters (see the docstring).
    agent_args = [f'model={model}']
    if gradient_clip and rl.actor:
        agent_args.append(
            f'actor_updater=tonic.torch.updaters.{rl.actor}(gradient_clip={gradient_clip})'
        )
    if gradient_clip:
        agent_args.append(
            f'critic_updater=tonic.torch.updaters.{rl.critic}(gradient_clip={gradient_clip})'
        )
    if rl.off_policy:
        # A deterministic actor explores only through injected noise; action_noise sets
        # its scale. Left to tonic's default, D4PG also keeps its 5-step return buffer.
        agent_args.append(f'exploration=tonic.explorations.NormalActionNoise(scale={action_noise})')

    agent = f'tonic.torch.agents.{rl.agent}(' + ', '.join(agent_args) + ')'
    # Only spelled out when non-empty, so a plain 'swim' run's environment string (and
    # so the config.yaml is_trained() compares against) is unchanged.
    task_args = f', task_kwargs={env_kwargs!r}' if env_kwargs else ''
    environment = f'tonic.environments.ControlSuite("swimmer-{env_name}"{time_feature}{task_args})'
    # tonic dumps a log.csv row (and runs a test episode) only at each epoch boundary,
    # so a run shorter than one epoch writes no log.csv. Cap the epoch at the run length
    # so even the short demo runs cross a boundary; None auto-targets ~5 points while
    # keeping tonic's 20,000-step cadence on long runs (steps >= 1e5 are unchanged).
    if epoch_steps is None:
        epoch_steps = min(20000, max(1, int(steps) // 5))
    trainer = (f'tonic.Trainer(steps={int(steps)}, epoch_steps={int(epoch_steps)}, '
               f'save_steps={int(save_steps)})')
    return agent, environment, _run_name(network, method, label), trainer


def run_path(network, method='ppo', n_links=6, task='swim', label=None, **_run_config_kwargs):
    """Data directory this run writes to -- pass to plot_performance / play_model.

    Runs are grouped by the environment they train in, so the same label on 'swim' and
    on 'evasion' stays two separate runs. Ignores the remaining run keys (sizes,
    budget, ...) so that a resolved run dict can be splatted straight in: run_path(**run).

    ES runs are written by es.run_es under a separate `es/` tree, so they are handed to
    es_run_path; either way the returned directory holds the log.csv plot_performance
    reads, which is what lets RL and ES curves go on one plot.
    """
    name = _run_name(network, method, label)
    if method == 'es':
        return es_run_path(network, n_links=n_links, name=name, task=task)
    return f'data/local/experiments/tonic/swimmer-{env_task(task, n_links)}/{name}'


def is_trained(path, agent, environment, trainer, seed=0):
    """True if `path` already holds a checkpointed run with these exact settings.

    Compares the agent/environment/trainer code strings (and seed) that `train()`
    would record in config.yaml against what is already saved there, so a changed
    parameter -- steps, sizes, action_noise, swimmer_kwargs, ... -- is retrained even
    though the run's label (and so its directory, from run_path) is unchanged. Also
    requires at least one checkpoint, so a run interrupted before its first save is
    retrained rather than treated as done.

    Pass it the exact (agent, environment, trainer) strings run_config() returns for
    the run being considered; call before train() and skip the call if it returns True.
    """
    config_path = os.path.join(path, 'config.yaml')
    checkpoints_path = os.path.join(path, 'checkpoints')
    if not os.path.isfile(config_path) or not os.path.isdir(checkpoints_path):
        return False
    if not any(name.startswith('step_') for name in os.listdir(checkpoints_path)):
        return False
    with open(config_path, 'r') as config_file:
        saved = yaml.load(config_file, Loader=yaml.FullLoader) or {}
    return (
        saved.get('agent') == agent
        and saved.get('environment') == environment
        and saved.get('trainer') == trainer
        and saved.get('seed') == seed
    )


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
