"""Actor-critic model factories, for both the MLP baseline and NCAP.

Importing this module requires tonic, so call `ensure_tonic()` first.
"""

import torch
from torch import nn
from tonic.torch import models, normalizers

from macrocircuits.ncap import SwimmerActor, SwimmerModule


def ppo_mlp_model(
    actor_sizes=(64, 64),
    actor_activation=torch.nn.Tanh,
    critic_sizes=(64, 64),
    critic_activation=torch.nn.Tanh,
):
    """
    Constructs an ActorCritic model with specified architectures for the actor and critic networks.

    Parameters:
    - actor_sizes (tuple): Sizes of the layers in the actor MLP.
    - actor_activation (torch activation): Activation function used in the actor MLP.
    - critic_sizes (tuple): Sizes of the layers in the critic MLP.
    - critic_activation (torch activation): Activation function used in the critic MLP.

    Returns:
    - models.ActorCritic: An ActorCritic model comprising an actor and a critic with MLP torsos,
      equipped with a Gaussian policy head for the actor and a value head for the critic,
      along with observation normalization.
    """

    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_sizes, actor_activation),
            head=models.DetachedScaleGaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def ppo_swimmer_model(
    n_joints=5,
    action_noise=0.1,
    critic_sizes=(64, 64),
    critic_activation=nn.Tanh,
    **swimmer_kwargs,
):
    return models.ActorCritic(
        actor=SwimmerActor(
            swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs),
            controller=swimmer_kwargs['controller'],
            distribution=lambda x: torch.distributions.normal.Normal(x, action_noise),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def d4pg_swimmer_model(
    n_joints=5,
    critic_sizes=(256, 256),
    critic_activation=nn.ReLU,
    **swimmer_kwargs,
):
    return models.ActorCriticWithTargets(
        actor=SwimmerActor(
            swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs), 
            controller=swimmer_kwargs['controller'],
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            # These values are for the control suite with 0.99 discount.
            head=models.DistributionalValueHead(-150., 150., 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )
