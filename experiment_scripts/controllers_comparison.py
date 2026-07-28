
# --> Append src path
import sys
from pathlib import Path
SRC = str(Path.cwd().parent / 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)



# --> Import packages
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt
from IPython.display import display

from dm_control import suite
from macrocircuits import ensure_tonic, test_dm_control
ensure_tonic()

from macrocircuits.training import (  # See src/macrocircuits/training.py.
    is_trained,
    play_model,
    resolve_runs,
    run_config,
    run_path,
    train,
)

from macrocircuits.es import (
    es_config,
    is_es_trained,
    play_es_model,
    run_es,
)
from macrocircuits.plotting import (  # See src/macrocircuits/plotting.py.
    plot_performance,
)



# --> Define TASK and experiment parameters
TASK = 'foraging'
STEPS = int(2E6)

# --> Define RUNS configuration
print(f"Defining runs for {TASK.upper()} task")
LABELS_CONTROLLERS = {
    # 'ncap_ppo (mlp_reflex controller)': 'mlp_reflex_foraging',
    # 'ncap_ppo (mlp_piourette controller)': 'mlp_piourette_foraging',
    # 'ncap_ppo (learned_steering controller)': 'learned_steering',
    'ncap_ppo (learned_steering minus warm_start controller)': 'learned_steering_no_warm_start',
    # 'ncap_ppo (mlp_reflex+piourette controller)': 'mlp_reflex_piourette_foraging',
    # 'ncap_ppo (mlp controller)': 'mlp_foraging'
}
RUNS = [ dict(network='ncap', task=TASK, controller=v, steps=STEPS, action_noise=0.3, label=k) for k, v in LABELS_CONTROLLERS.items()]



# --> Run Environment per RUNS configuration
for run in tqdm(RUNS, total=len(RUNS), desc=f"Running {TASK.upper()} task"):
    np.random.seed(42)
    torch.manual_seed(42)
    agent, environment, name, trainer = run_config(**run)
    train('import tonic.torch', agent, environment, name=name, trainer=trainer, seed=42)



# -> Play and save run video per RUNS configuration
for label in tqdm(LABELS_CONTROLLERS.keys(), total=len(RUNS), desc=f"Saving video, {TASK.upper()} task"):
    play_model(f'data/local/experiments/tonic/swimmer-{TASK}/{label}')
