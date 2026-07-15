"""Supporting code for the Macrocircuits tutorial notebooks.

The notebooks in `Ressources/` orchestrate; the implementation lives here:

- `video`       -- writing rollout frames to mp4 and showing them inline
- `envs`        -- the swimmer environments and swim tasks
- `tonic_setup` -- cloning the tonic RL library on demand
- `training`    -- training agents and replaying checkpoints
- `models`      -- actor-critic factories for the MLP baseline and NCAP
- `constraints` -- weight/activation constraints and initializers
- `ncap`        -- the C.-elegans-inspired swimmer circuit
- `plotting`    -- learning curves and architecture diagrams

Importing this package registers the `swim` and `swim_12_links` tasks with the
dm_control swimmer suite, so `suite.load('swimmer', 'swim')` resolves.

`training` and `models` are deliberately not re-exported here: they import
tonic, which `ensure_tonic()` has to clone first.
"""

from macrocircuits.envs import Swim, render, test_dm_control
from macrocircuits.tonic_setup import ensure_tonic
from macrocircuits.video import display_video, write_video

__all__ = [
    'Swim',
    'display_video',
    'ensure_tonic',
    'render',
    'test_dm_control',
    'write_video',
]
