"""Minimal training script: PPO + NCAP on the foraging task (no controller).

Usage: python scripts/train_ppo_ncap.py

It clones `tonic` if missing, then starts a short demo run. Adjust `STEPS`
and `PARALLEL` to scale up to longer experiments.
"""

from macrocircuits.tonic_setup import ensure_tonic
ensure_tonic()

from macrocircuits.training import run_config, train

# User-configurable knobs
NETWORK = 'ncap'
METHOD = 'ppo'
N_LINKS = 6
TASK = 'foraging'
CONTROLLER = None  # NCAP-only
STEPS = int(5e4)
PARALLEL = 1


def main():
    agent, environment, name, trainer = run_config(
        network=NETWORK,
        method=METHOD,
        n_links=N_LINKS,
        task=TASK,
        controller=CONTROLLER,
        steps=STEPS,
        swimmer_kwargs={'use_weight_sharing': True},
    )

    print('Starting training:', name)
    train(header=None, agent=agent, environment=environment, name=name, trainer=trainer, parallel=PARALLEL)


if __name__ == '__main__':
    main()
