"""Supporting code for the Macrocircuits tutorial notebooks.

The notebooks in `Ressources/` orchestrate; the implementation lives here:

- `video`       -- writing rollout frames to mp4 and showing them inline
- `envs`        -- the swimmer environments and swim tasks
- `tonic_setup` -- cloning the tonic RL library on demand
- `training`    -- training agents (tonic RL) and replaying checkpoints
- `es`          -- training NCAP / an MLP baseline with Evolution Strategies
- `models`      -- actor-critic factories for the MLP baseline and NCAP
- `constraints` -- weight/activation constraints and initializers
- `ncap`        -- the C.-elegans-inspired swimmer circuit
- `plotting`    -- learning curves and architecture diagrams

Importing this package registers the `swim`, `swim_12_links`, `swim_to_ball`,
`foraging` and `evasion` tasks with the dm_control swimmer suite, so
`suite.load('swimmer', 'swim')` resolves.

`training` and `models` are deliberately not re-exported here: they import
tonic, which `ensure_tonic()` has to clone first. `es` is tonic-free (pure torch +
dm_control), so its entry points are re-exported below.
"""

from macrocircuits.envs import Swim, render, test_dm_control  # SwimToBall folded into Swim (see enable_single_target)
from macrocircuits.es import (
    EvolutionStrategy,
    es_config,
    es_run_path,
    play_es_model,
    run_es,
)
from macrocircuits.tonic_setup import ensure_tonic
from macrocircuits.video import display_video, write_video

__all__ = [
    'EvolutionStrategy',
    'Swim',
    # 'SwimToBall',  # folded into Swim's enable_single_target/enable_foraging/enable_obstacles flags
    'display_video',
    'ensure_tonic',
    'es_config',
    'es_run_path',
    'play_es_model',
    'render',
    'run_es',
    'test_dm_control',
    'write_video',
]
