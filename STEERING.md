# Steering & Foraging — Implementation Guide

Branch: `hardcoded-turn-reflex` (off `main`).

How the NCAP swimmer is made to **turn on command** and **navigate to food**, from a
hardcoded reflex up to a small learned controller. Everything here is additive over
`main`; none of the abandoned `improve-foraging-reflex` machinery (the P+D reflex, the
5-term reward stack, the learnable-gain / adaptive / phase-gate reflexes) is reused.

---

## TL;DR

- **The whole thing turned on one bug.** The egocentric `to_target` vector is
  `[lateral, longitudinal]`, not `[forward, lateral]` — the older steering code had the
  two axes **swapped**, so it was steering on the wrong component and could never work.
  Measured directly against the simulator (see [The convention](#the-convention)).
- With the axes correct, a **trivial hardcoded reflex** (`steer_to_food`) reaches
  **~90%** physics-only success on foraging — versus the old investigation's best of 47%.
  No learning.
- A small **learned** controller (`learned_steering`, Option 5) learns *only the steering
  decision* on top of a fixed turn primitive. Warm-started at the hardcoded solution, PPO
  lands around the same ~90% — it **matches** the hardcoded reflex within seed noise
  (individual seeds range 83–93%). Once the geometry is right and the turn primitive is
  given, learning neither clearly beats nor breaks it.

| Controller | Learned? | Foraging success (physics-only, 30 eps) |
|---|---|---|
| none (circuit swims straight), untrained | — | ~10–23% |
| **none, PPO-trained 1e5 steps** | circuit only | **23%** — identical to untrained |
| *old saga best (swapped axes)* | — | *47%* |
| MLP / learned-from-random-init (never found the sign) | yes | 7% |
| **`steer_to_food`** (hardcoded, correct convention) | no | **~90%** |
| `learned_steering`, warm-started, before RL | (BC init) | ~80–90% |
| `learned_steering`, PPO-trained | yes | **83–93%** (3-seed mean 86.7%; a single seed hit 93%) |

> Absolute numbers shift ~±10% with the evaluation seed (which fresh episodes are drawn),
> so compare within one eval seed. The `forage_train_poc.ipynb` run (eval seed 0) reads
> no-steering 23% / warm-start 83% / PPO-trained **93%**; the 3-seed sweep (eval seed
> 4242) reads warm-start ~90% / PPO-trained 86.7%. Both are consistent with "learned ≈
> hardcoded."
>
> The trained no-steering row is the sharpest version of the point. With `controller=None`
> the circuit is built with `include_turn_control=False`, so there is no turn input for
> PPO to learn to use — 1e5 steps of training on a dense food-seeking reward moves it from
> 23% to 23%. The 23% itself is not navigation: it is the rate at which the default
> undulating gait happens to graze food that spawned close by. **The gap to 93% is the
> controller's, not training's.**

---

## The convention

Measured by placing food at a known bearing relative to the worm's actual heading and
reading the observation (`scratchpad/check_convention.py`):

| food placed | `to_target[0]` | `to_target[1]` |
|---|---|---|
| dead ahead | 0.0 | **−0.5** |
| to the worm's **left** | **+0.5** | 0.0 |
| to the worm's **right** | **−0.5** | 0.0 |

So for any steering code:

```python
lateral = to_target[0]        # +ve  => food is to the worm's LEFT
forward = -to_target[1]       # +ve  => food is AHEAD
angle   = atan2(lateral, forward)   # 0 = dead ahead, +ve = food to the left
```

And, from the turn-primitive characterisation, **driving NCAP's `left` input turns the
worm to its own left** (`right` → its right). This is recorded in the project memory note
`swimmer-egocentric-axis-convention`.

---

## Controllers added

All are selected by name via a run's `controller=` (see
`macrocircuits.controllers.CONTROLLERS`) and reach NCAP the same way as the existing
reflex/MLP controllers.

### 1. Turn primitive — `make_turn_reflex` (`reflex_steering.py`)

Sensor-free: holds NCAP's turn signal to one side every step. A direct actuator test, and
the building block everything else stands on.

```python
make_turn_reflex(n_joints, direction='left'|'right', strength=1.0, speed=1.0)
# registry names: 'turn_left', 'turn_right'  (strength 1.0)
```

- `strength` regimes (calibrated in `turn_poc.ipynb`): **~0.5** = wide arc, **~0.75** =
  tight in-place pivot, **≥0.8** = the head oscillator is overpowered and the spin
  destabilises (can even flip direction), **1.0** = curls up in place.

### 2. Hardcoded navigator — `make_steer_to_food_reflex` (`reflex_steering.py`)

Turns toward the food (correct convention) and swims. A fixed reflex, **no learning**,
**~90%** on foraging.

```python
make_steer_to_food_reflex(n_joints, strength=0.75, gain=3.0)
# registry name: 'steer_to_food'
```

### 3. Learned decision — `LearnedSteering` / `make_learned_steering` (`controllers.py`)

Option 5: a tiny MLP maps the *unit* egocentric direction to the food → one turn command,
fed to NCAP's left/right inputs at a fixed strength. Learns only the *decision*, not the
mechanics.

```python
make_learned_steering(n_joints, hidden_size=8, turn_strength=0.75, warm_start=True)
# registry name: 'learned_steering'   (~33 learnable params)
```

Two deliberate design choices, each fixing a concrete failure:
- **`tanh`-bounded, never a hard `[0,1]` clamp.** A clamp saturates ~95% of steps and
  starves the parameters of gradient (a wall this project already hit); `tanh` keeps a
  live gradient every step, and `turn_strength ≤ 0.75 < 1` means no upper clamp is needed.
- **Warm start.** From random init, PPO never even discovered the correct turn *sign*
  (~7%). So the net is behaviour-cloned to the correct-sign hardcoded steerer first
  (`_behaviour_clone`), and PPO only *refines* it — the "good init + learning" philosophy
  the NCAP paper uses for the circuit weights.

---

## Reward added (`envs.py`)

Fresh, minimal, on the `foraging` task only. Swim-speed is off by default (it rewards
"swim fast" regardless of the food — the original confound), leaving a dense progress
signal plus a success bonus.

| `foraging(...)` / `task_kwargs` key | default | meaning |
|---|---|---|
| `speed_reward_weight` | **0.0** | stock swim-speed reward (off here to isolate food-seeking) |
| `progress_reward_weight` | 0.0 | dense per-step reward for distance **closed** toward food (potential-based) |
| `alignment_reward_weight` | 0.0 | reward for **facing** the food — `cos(bearing)`, +1 dead ahead, −1 dead behind |
| `alignment_gated_progress_weight` | 0.0 | progress **scaled by** alignment, so distance closed while pointed the wrong way earns ~0 |
| `eat_bonus` | 0.0 | one-off reward each time the food is reached |

Progress is skipped on the first step and right after a respawn, so a target's sudden
distance jump is never scored as a huge spurious loss. Plain `swim` is unchanged
(`speed_reward_weight` defaults to 1.0 there).

The last two came from the `improve-foraging-reflex` branch, where they were the largest
single win found (a fixed reflex went 23% → 47% on `alignment_reward_weight=0.5` alone).
**Those numbers do not transfer as-is:** that branch computed alignment as
`to_target[0]/dist`, which on the measured axes is the *lateral* component — so it was
rewarding swimming **abeam** of the food, not at it. The version here uses
`-to_target[1]/dist`, verified by placing food at known bearings (ahead → +1.00,
behind → −1.00, abeam → 0.00). Treat the weights as untuned on the corrected term.

That branch also reported, across at least four separate checks, that two independently
good changes reliably stack *worse* than either alone (progress + alignment + eat = 37%
vs. 47% for alignment by itself). Worth knowing before combining these.

---

## Notebooks (`Ressources/`)

All run top-to-bottom; the first two need **no training**.

| Notebook | What it shows |
|---|---|
| `turn_poc.ipynb` | The worm turning **left** and **right** in place (turn primitive), videos + head-path plots. |
| `forage_poc.ipynb` | The worm **navigating to food** with `steer_to_food` (~90%), video + trajectory + success bars. No training. |
| `forage_train_poc.ipynb` | **Trains two runs** with PPO (~1e5 steps each) — NCAP alone (`controller=None`) and NCAP + `learned_steering` — on an identical reward, then learning curves + a 4-way success comparison (each run against its own untrained start) + a video and head-path plot per run. |

---

## How to use

**Run a controller (train a config):**
```python
from macrocircuits.training import resolve_runs, run_config, run_path, is_trained, train

RUN = resolve_runs([dict(
    network='ncap', method='ppo', task='foraging',
    controller='learned_steering',                       # or 'steer_to_food' (no training needed)
    task_kwargs=dict(progress_reward_weight=10.0, eat_bonus=5.0),
    steps=int(1e5), label='learned_steering_run',
)])[0]
agent, env, name, trainer = run_config(**RUN)
train(header='import tonic.torch', agent=agent, environment=env, name=name, trainer=trainer)
```

**Evaluate physics-only success** (the trustworthy metric — aggregate training reward is
misleading in this env): see `success_rate()` in `forage_poc.ipynb`, or
`scratchpad/train_eval_learned_steering.py`. It counts the fraction of fresh episodes
whose head comes within 0.15 of the food, reading distance from the pre-reset transition
observation (never from a just-reset env).

---

## Design notes / gotchas

- **Measure the convention, don't trust the code.** The axis swap sat undetected through
  the entire old investigation. `scratchpad/check_convention.py` settles it in seconds.
- **Aggregate reward lies here.** Always score with physics-only success across several
  seeds — single runs are noisy enough to invert rankings.
- **The env emits no `done` at the time limit.** The raw tonic gym env relies on the
  distributed wrapper (`tonic.environments.distribute`) for episode boundaries; loop over
  that (via `infos['resets']`), not the raw env's `done`.

---

## Files changed vs `main`

```
src/macrocircuits/reflex_steering.py  +make_turn_reflex, make_turn_left/right_reflex,
                                       make_steer_to_food_reflex
src/macrocircuits/controllers.py      +LearnedSteering, make_learned_steering,
                                       registry: turn_left/turn_right/steer_to_food/learned_steering
src/macrocircuits/envs.py             +speed_reward_weight, progress_reward_weight, eat_bonus
src/macrocircuits/training.py         +imports for the above
Ressources/turn_poc.ipynb             turn-primitive POC (new)
Ressources/forage_poc.ipynb           navigation POC, no training (new)
Ressources/forage_train_poc.ipynb     navigation + PPO training (new)
```
