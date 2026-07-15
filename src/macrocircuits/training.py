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
from macrocircuits.models import d4pg_swimmer_model, ppo_mlp_model, ppo_swimmer_model


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
