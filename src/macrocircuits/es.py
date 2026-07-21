"""Training NCAP (and an MLP baseline) with Evolution Strategies.

This is a single-process port of the upstream `estorch` classic Evolution Strategy
(https://github.com/nikhilxb/ncap-swimmer, which itself wraps
https://github.com/goktug97/estorch). The upstream trainer parallelises rollouts
with MPI (`mpi4py`) and drives a `garage`-based policy; neither is available here, so
this module keeps only the algorithm and runs the population sequentially. It is
deliberately tonic-free -- it needs nothing beyond torch, numpy and dm_control -- so
it can be imported without cloning tonic, unlike `training`/`models`.

The optimisation is the classic ES of Salimans et al. 2017
(https://arxiv.org/abs/1703.03864): antithetic Gaussian perturbations of the policy
weights are scored by episode return, the returns are rank-transformed, and their
weighted sum estimates a gradient that an ordinary torch optimizer ascends. NCAP has
only a handful of weight-shared parameters, which is exactly the regime where ES
shines and where the paper reports its swimmer results.

Typical use from a notebook cell::

    from macrocircuits.es import run_es
    es = run_es('ncap', n_links=6, population_size=64, sigma=0.02, n_steps=100)
    # es.best_reward, es.best_policy_dict, and a plot-compatible log.csv on disk.
"""

import copy
import csv
import os
from functools import lru_cache

import numpy as np
import torch
from torch import nn

import yaml

# Importing this registers `swim` / `swim_12_links` / `swim_to_ball` / `foraging` /
# `evasion` with the dm_control swimmer suite, so `suite.load('swimmer', 'swim')`
# resolves below. env_task/task_env_kwargs map a run's (task, n_links) choice onto the
# registered task name and its kwargs -- the same helpers `training.run_config` uses.
from macrocircuits.envs import env_task, task_env_kwargs
from macrocircuits.ncap import SwimmerModule
from macrocircuits.reflex_steering import check_controller, make_controller


# ==================================================================================================
# Rank transformation (ported verbatim from estorch).


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers


def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards):
    """Maps returns onto evenly spaced centers in [-0.5, 0.5] by rank.

    Rank (rather than raw return) makes the gradient estimate invariant to reward
    scale and robust to outliers, which is standard for ES.
    """
    ranks = _compute_ranks(rewards)
    values = _center_function(len(rewards))
    return values[ranks]


# ==================================================================================================
# Policies. An ES policy is a plain nn.Module mapping a flat observation tensor to an
# action tensor in [-1, 1], with an optional reset() called at each episode start.


class NCAPSwimmerPolicy(nn.Module):
    """The NCAP circuit as a deterministic, resettable ES policy.

    Slices joint angles from the front of the observation (as the dm_control swimmer
    lays them out) and normalises them the same way `ncap.SwimmerActor` does. Unlike
    the RL actor it does not read a timestep feature from the observation; the head
    oscillator instead advances the module's own step counter, which `reset()` zeroes
    at the start of each rollout.

    `controller` is an optional steering reflex (see `reflex_steering`), which reads
    the task's egocentric to_target / to_obstacle vector out of the same observation
    and drives the circuit's turn inputs with it -- the ES counterpart of what
    `ncap.SwimmerActor` does for the RL path. It adds no parameters: the reflex is
    fixed, and only the circuit's own `bneuron_turn` weight is evolved. The vector it
    reads is in raw task units on both paths -- tonic normalizes inside its observation
    encoders, which `SwimmerActor` does not use -- so a reflex tuned on one (e.g.
    `make_obstacle_avoidance_reflex`'s `reaction_distance`) behaves the same on the other.
    """

    def __init__(self, n_joints, controller=None, **swimmer_kwargs):
        super().__init__()
        self.n_joints = n_joints
        self.controller = controller
        self.swimmer = SwimmerModule(
            n_joints=n_joints, include_turn_control=(controller is not None), **swimmer_kwargs
        )
        self._joint_limit = 2 * np.pi / (n_joints + 1)  # As in dm_control (uses n_bodies).

    def reset(self):
        self.swimmer.reset()

    def forward(self, observations):
        joint_pos = observations[..., :self.n_joints]
        joint_pos = torch.clamp(joint_pos / self._joint_limit, min=-1.0, max=1.0)
        right, left, speed = (
            self.controller(observations) if self.controller else (None, None, None)
        )
        # timesteps=None -> use the module's internal counter for the oscillator;
        # log_activity=False -> don't accumulate connection records across the run.
        return self.swimmer(
            joint_pos,
            timesteps=None,
            right_control=right,
            left_control=left,
            speed_control=speed,
            log_activity=False,
        )


class MLPSwimmerPolicy(nn.Module):
    """A generic fully-connected policy, the ES counterpart of the RL MLP baseline.

    A tanh-MLP mapping the full flattened observation to actions, with a final tanh
    bounding actions to [-1, 1]. `reset()` is a no-op; it exists so the rollout loop
    can treat every policy uniformly.
    """

    def __init__(self, observation_size, action_size, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        sizes = [observation_size, *hidden_sizes]
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(in_size, out_size), activation()]
        layers += [nn.Linear(sizes[-1], action_size), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def reset(self):
        pass

    def forward(self, observations):
        return self.net(observations)


# ==================================================================================================
# Rollout agent.


def _flatten_observation(observation):
    """Concatenate a dm_control observation OrderedDict into one float32 vector.

    dm_control lays the swimmer observation out with 'joints' first, which is what
    lets NCAPSwimmerPolicy slice joint angles off the front.
    """
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in observation.values()])


class SwimmerESAgent:
    """Scores a policy by rolling it out on a dm_control swimmer task.

    ES only needs a scalar fitness per candidate, so this wraps one environment and
    returns the mean episode return over `n_evals` deterministic rollouts. It also
    tallies `total_steps`, the cumulative environment interactions, so the trainer can
    report ES's (large) sample cost honestly.
    """

    def __init__(self, task='swim', n_evals=1, seed=0, task_kwargs=None):
        from dm_control import suite

        task_kwargs = dict(task_kwargs or {})
        task_kwargs.setdefault('random', seed)
        self.task = task
        self.n_evals = n_evals
        self.env = suite.load('swimmer', task, task_kwargs=task_kwargs)
        self.action_size = self.env.action_spec().shape[0]
        self.observation_size = _flatten_observation(self.env.reset().observation).shape[0]
        self.total_steps = 0

    def rollout(self, policy):
        """Return the mean episode return of `policy` over `n_evals` episodes."""
        total_reward = 0.0
        for _ in range(self.n_evals):
            timestep = self.env.reset()
            if hasattr(policy, 'reset'):
                policy.reset()
            while not timestep.last():
                observation = torch.as_tensor(
                    _flatten_observation(timestep.observation), dtype=torch.float32
                )
                action = policy(observation).detach().cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                timestep = self.env.step(action)
                self.total_steps += 1
                if timestep.reward is not None:
                    total_reward += float(timestep.reward)
        return total_reward / self.n_evals


# ==================================================================================================
# Evolution Strategy optimizer (single process).


class EvolutionStrategy:
    """Classic Evolution Strategy, sequential (no MPI).

    Optimises `policy.parameters()` to maximise `agent.rollout(policy)`. Each step
    samples an antithetic population theta +/- sigma * epsilon, scores every member,
    rank-transforms the returns, forms the ES gradient estimate, writes it (negated,
    since optimizers descend) into `.grad`, clamps it for stability as upstream does,
    and takes one optimizer step.

    Args:
      policy: an nn.Module instance to optimise (e.g. NCAPSwimmerPolicy).
      agent: an object with `rollout(policy) -> float`.
      optimizer: a constructed torch optimizer over `policy.parameters()`. If None, an
                 Adam with `lr`/`weight_decay` is created.
      population_size: number of perturbed candidates per step (must be even; the
                 population is mirrored, so half that many noise vectors are drawn).
      sigma: standard deviation of the weight perturbations.
      device: torch device; CPU is recommended (a full-policy clone is evaluated).
      seed: optional seed for torch/numpy sampling.
    """

    def __init__(
        self,
        policy,
        agent,
        optimizer=None,
        population_size=64,
        sigma=0.02,
        lr=0.02,
        weight_decay=0.0,
        device='cpu',
        seed=None,
    ):
        if population_size % 2 != 0:
            raise ValueError(f'population_size must be even, got {population_size}')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.agent = agent
        self.population_size = population_size
        self.sigma = sigma
        self.optimizer = optimizer or torch.optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        # A clone loaded with each candidate's weights, so scoring never touches the
        # master policy's parameters.
        self.target = copy.deepcopy(self.policy).to(self.device)
        self.n_parameters = sum(p.numel() for p in self.policy.parameters())

        self.best_reward = -float('inf')
        self.best_policy_dict = copy.deepcopy(self.policy.state_dict())
        self.history = []

    def _sample_population(self):
        params = nn.utils.parameters_to_vector(self.policy.parameters()).detach().cpu()
        noise = torch.distributions.normal.Normal(0.0, self.sigma).sample(
            [self.population_size // 2, params.shape[0]]
        )
        population = torch.cat((params + noise, params - noise))
        epsilon = torch.cat((noise, -noise))
        return population, epsilon

    def _population_returns(self, population):
        returns = np.empty(len(population), dtype=np.float32)
        for i, candidate in enumerate(population):
            nn.utils.vector_to_parameters(candidate.to(self.device), self.target.parameters())
            returns[i] = self.agent.rollout(self.target)
        return returns

    def _estimate_gradient(self, returns, epsilon):
        ranked = torch.from_numpy(rank_transformation(returns)).unsqueeze(0).float()
        return (torch.mm(ranked, epsilon) / (self.population_size * self.sigma)).squeeze()

    def step(self):
        """Run one ES generation; return (post-update episode return, population returns)."""
        with torch.no_grad():
            population, epsilon = self._sample_population()
            returns = self._population_returns(population)
            grad = self._estimate_gradient(returns, epsilon)

            index = 0
            for parameter in self.policy.parameters():
                size = parameter.numel()
                # Negate: the estimate is an ascent direction, optimizers descend.
                parameter.grad = (-grad[index:index + size]).view(parameter.shape).to(self.device)
                # Bound the update, exactly as upstream does, to keep training stable.
                parameter.grad.clamp_(-1.0, 1.0)
                index += size
            self.optimizer.step()

            episode_reward = self.agent.rollout(self.policy)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_policy_dict = copy.deepcopy(self.policy.state_dict())
        return episode_reward, returns

    def train(self, n_steps, callback=None, verbose=True):
        """Run `n_steps` generations, recording per-generation stats in `self.history`."""
        for step in range(n_steps):
            episode_reward, returns = self.step()
            record = {
                'step': step,
                'episode_reward': float(episode_reward),
                'max_population_reward': float(returns.max()),
                'best_reward': float(self.best_reward),
                'total_env_steps': int(getattr(self.agent, 'total_steps', 0)),
            }
            self.history.append(record)
            if verbose:
                print(
                    f'ES step {step:4d} | episode {episode_reward:8.2f} | '
                    f'pop max {returns.max():8.2f} | best {self.best_reward:8.2f} | '
                    f'env steps {record["total_env_steps"]:,}'
                )
            if callback is not None:
                callback(self, record)
        return self.history

    def load_best(self):
        """Load the best-scoring weights found during training back into `self.policy`."""
        if self.best_policy_dict is not None:
            self.policy.load_state_dict(self.best_policy_dict)
        return self.policy


# ==================================================================================================
# High-level entry points (mirror training.run_config / run_path).


def es_run_path(network, n_links=6, name=None, data_dir=None, task='swim'):
    """Directory an ES run writes to -- pass to plot_performance / for checkpoints.

    Parallels `training.run_path` but under an `es/` tree rather than `tonic/`, so RL
    and ES curves can be handed to `plot_performance` together. Joined with '/' rather
    than os.sep to match run_path, which the notebook prints these next to; forward
    slashes are valid paths on Windows too.
    """
    task = env_task(task, n_links)
    name = name or f'{network}_es'
    base = data_dir or 'data/local/experiments/es'
    return '/'.join((base, f'swimmer-{task}', name))


def es_config(
    network,
    n_links=6,
    task='swim',
    task_kwargs=None,
    controller=None,
    generations=100,
    population_size=64,
    sigma=0.02,
    lr=0.02,
    weight_decay=0.0,
    n_evals=1,
    hidden_sizes=(64, 64),
    seed=0,
    swimmer_kwargs=None,
    label=None,
    **_rl_kwargs,
):
    """Map a resolved run dict onto run_es()'s keyword arguments.

    The ES counterpart of `training.run_config`: a run declared in the notebook is
    splatted in (`run_es(**es_config(**run))`) and the RL-only keys it also carries --
    critic_sizes, gradient_clip, steps, ... -- are ignored here, exactly as run_config
    ignores the ES-only ones.

    Two names differ from run_es()'s own: `generations` is its `n_steps` (ES counts
    generations, not environment steps, so reusing the RL `steps` key would mislead),
    and `label` is its `name`, so the run lands in the directory run_path() predicts.
    """
    return dict(
        network=network,
        n_links=n_links,
        task=task,
        task_kwargs=task_kwargs,
        controller=controller,
        n_steps=generations,
        population_size=population_size,
        sigma=sigma,
        lr=lr,
        weight_decay=weight_decay,
        n_evals=n_evals,
        hidden_sizes=hidden_sizes,
        seed=seed,
        swimmer_kwargs=swimmer_kwargs,
        name=label,
    )


def is_es_trained(
    path,
    network,
    n_links=6,
    task='swim',
    task_kwargs=None,
    controller=None,
    population_size=64,
    sigma=0.02,
    lr=0.02,
    n_steps=100,
    n_evals=1,
    seed=0,
    hidden_sizes=(64, 64),
    weight_decay=0.0,
    swimmer_kwargs=None,
    **_run_es_kwargs,
):
    """True if `path` already holds an ES run with these exact hyperparameters.

    The ES counterpart of `training.is_trained`: compares the hyperparameters
    `run_es` would save in config.yaml (see `_save_es_run`) against what's already
    there, so a changed population_size, sigma, hidden_sizes, ... is retrained even
    though the run's label (and so its directory) is unchanged. Also requires the
    best-policy checkpoint to exist. **_run_es_kwargs absorbs the extra keys
    `es_config` returns (name, data_dir) that aren't saved hyperparameters, so this
    can be called as `is_es_trained(path, **es_config(**run))`.
    """
    config_path = os.path.join(path, 'config.yaml')
    checkpoint_path = os.path.join(path, 'checkpoints', 'best.pt')
    if not os.path.isfile(config_path) or not os.path.isfile(checkpoint_path):
        return False
    with open(config_path, 'r') as config_file:
        saved = yaml.safe_load(config_file) or {}
    return (
        saved.get('network') == network
        and saved.get('n_links') == n_links
        and saved.get('task') == env_task(task, n_links)
        # Defaulted to {} rather than None so a run saved before task_kwargs existed
        # still matches a plain run and is not needlessly retrained.
        and saved.get('task_kwargs', {}) == task_env_kwargs(task, n_links, task_kwargs)
        and saved.get('controller') == controller
        and saved.get('population_size') == population_size
        and saved.get('sigma') == sigma
        and saved.get('lr') == lr
        and saved.get('n_steps') == n_steps
        and saved.get('n_evals') == n_evals
        and saved.get('seed') == seed
        and saved.get('hidden_sizes') == list(hidden_sizes)
        and saved.get('weight_decay') == weight_decay
        and saved.get('swimmer_kwargs') == dict(swimmer_kwargs or {})
    )


def make_es_policy(network, agent, hidden_sizes=(64, 64), swimmer_kwargs=None, controller=None):
    """Build the ES policy for `network` ('ncap' or 'mlp'), sized from `agent`.

    `controller` names a steering reflex to plug into NCAP's turn inputs; the MLP
    baseline has none, and `check_controller` (called by run_es) rejects that pairing
    before it gets here.
    """
    if network == 'ncap':
        return NCAPSwimmerPolicy(
            n_joints=agent.action_size,
            controller=make_controller(controller, agent.action_size),
            **(swimmer_kwargs or {}),
        )
    if network == 'mlp':
        return MLPSwimmerPolicy(agent.observation_size, agent.action_size, hidden_sizes=hidden_sizes)
    raise ValueError(f"network must be 'mlp' or 'ncap', got {network!r}")


def _save_es_run(path, es, config):
    """Write a tonic-compatible log.csv, a config.yaml, and the best checkpoint.

    log.csv carries the two columns `plotting.plot_performance` reads:
    `test/episode_score/mean` (the running best return) against
    `test/episode_length/mean` (environment steps spent that generation, so the
    plot's cumulative-steps x-axis reflects ES's true sample cost).
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'config.yaml'), 'w') as config_file:
        yaml.safe_dump(config, config_file)

    fieldnames = [
        'test/episode_score/mean',
        'test/episode_length/mean',
        'step',
        'episode_reward',
        'max_population_reward',
        'best_reward',
        'total_env_steps',
    ]
    with open(os.path.join(path, 'log.csv'), 'w', newline='') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        previous_steps = 0
        for record in es.history:
            step_delta = record['total_env_steps'] - previous_steps
            previous_steps = record['total_env_steps']
            writer.writerow({
                'test/episode_score/mean': record['best_reward'],
                'test/episode_length/mean': step_delta,
                **record,
            })

    checkpoints = os.path.join(path, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)
    torch.save(es.best_policy_dict, os.path.join(checkpoints, 'best.pt'))


def run_es(
    network='ncap',
    n_links=6,
    task='swim',
    controller=None,
    population_size=64,
    sigma=0.02,
    lr=0.02,
    n_steps=100,
    n_evals=1,
    seed=0,
    hidden_sizes=(64, 64),
    weight_decay=0.0,
    device='cpu',
    name=None,
    data_dir=None,
    swimmer_kwargs=None,
    task_kwargs=None,
    verbose=True,
    save=True,
):
    """Train NCAP or an MLP baseline on the swimmer with Evolution Strategies.

    Builds the rollout agent and policy, runs `n_steps` ES generations, and (by
    default) writes a plot-compatible run directory (see `es_run_path`). Returns the
    fitted `EvolutionStrategy`; its `best_reward`, `best_policy_dict`, and `history`
    hold the results.

    Parameters mirror `training.run_config` where they overlap (`network`, `n_links`,
    `task`, `controller`); the rest are ES hyperparameters. `n_evals` averages several
    rollouts per candidate to reduce fitness noise; `swimmer_kwargs` is forwarded to the
    NCAP `SwimmerModule` (e.g. `oscillator_period`), `task_kwargs` to the dm_control task
    (e.g. `n_obstacles` on 'evasion', or `time_limit`).

    `controller` plugs a steering reflex into NCAP on the target/obstacle tasks, exactly
    as on the RL path -- see `NCAPSwimmerPolicy`. Without one the circuit swims
    unsteered there, so only its reward changes and not its behaviour.
    """
    check_controller(controller, network, task)
    env_name = env_task(task, n_links)
    env_kwargs = task_env_kwargs(task, n_links, task_kwargs)

    agent = SwimmerESAgent(task=env_name, n_evals=n_evals, seed=seed, task_kwargs=env_kwargs)
    policy = make_es_policy(
        network,
        agent,
        hidden_sizes=hidden_sizes,
        swimmer_kwargs=swimmer_kwargs,
        controller=controller,
    )
    es = EvolutionStrategy(
        policy,
        agent,
        population_size=population_size,
        sigma=sigma,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )
    es.train(n_steps, verbose=verbose)

    if save:
        config = {
            'network': network,
            'n_links': n_links,
            'population_size': population_size,
            'sigma': sigma,
            'lr': lr,
            'n_steps': n_steps,
            'n_evals': n_evals,
            'seed': seed,
            'hidden_sizes': list(hidden_sizes),
            'weight_decay': weight_decay,
            'task': env_name,
            'best_reward': es.best_reward,
            # Saved so play_es_model() can rebuild the exact policy and environment.
            'task_kwargs': env_kwargs,
            'controller': controller,
            'swimmer_kwargs': dict(swimmer_kwargs or {}),
        }
        _save_es_run(
            es_run_path(network, n_links, name=name, data_dir=data_dir, task=task), es, config
        )
    return es


def play_es_model(path, camera_id=0, width=640, height=480, fps=60):
    """Replay an ES run's best policy and render it -- the ES twin of `training.play_model`.

    An ES checkpoint is a bare policy state_dict, not a tonic agent checkpoint, so
    tonic's player cannot load it. This rebuilds the policy from the run's config.yaml,
    rolls out one deterministic episode, and returns the rendered video.

    Args:
      path: an ES run directory, as returned by `es_run_path` / `training.run_path`.

    Returns:
      An IPython HTML video element, like `training.play_model`.
    """
    from macrocircuits.video import display_video

    with open(os.path.join(path, 'config.yaml')) as config_file:
        config = yaml.safe_load(config_file)

    agent = SwimmerESAgent(
        task=config['task'],
        n_evals=1,
        seed=config.get('seed', 0),
        # Rebuilt with the task options it trained on, so e.g. an 'evasion' replay has
        # the same number of obstacles (and so the same observation size).
        task_kwargs=config.get('task_kwargs'),
    )
    policy = make_es_policy(
        config['network'],
        agent,
        hidden_sizes=tuple(config.get('hidden_sizes') or (64, 64)),
        swimmer_kwargs=config.get('swimmer_kwargs'),
        controller=config.get('controller'),
    )
    checkpoint = os.path.join(path, 'checkpoints', 'best.pt')
    policy.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    policy.eval()

    # The rollout is SwimmerESAgent.rollout's, with a rendered frame per step; it is
    # repeated here rather than folded into that method to keep the scoring path -- run
    # once per candidate per generation -- free of rendering.
    frames = []
    total_reward = 0.0
    timestep = agent.env.reset()
    policy.reset()
    with torch.no_grad():
        while not timestep.last():
            frames.append(agent.env.physics.render(camera_id=camera_id, width=width, height=height))
            observation = torch.as_tensor(
                _flatten_observation(timestep.observation), dtype=torch.float32
            )
            action = np.clip(policy(observation).cpu().numpy(), -1.0, 1.0)
            timestep = agent.env.step(action)
            if timestep.reward is not None:
                total_reward += float(timestep.reward)

    print(f'Loaded {checkpoint} -- episode score {total_reward:.2f}')
    return display_video(frames, fps=fps)


__all__ = [
    'EvolutionStrategy',
    'MLPSwimmerPolicy',
    'NCAPSwimmerPolicy',
    'SwimmerESAgent',
    'es_config',
    'es_run_path',
    'is_es_trained',
    'make_es_policy',
    'play_es_model',
    'rank_transformation',
    'run_es',
]
