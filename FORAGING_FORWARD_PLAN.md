# Foraging Forward Plan

Actionable plan for improving NCAP foraging (Sonja + Luka).  
Out of scope: Emmanuel’s pirouette / run-and-tumble controller work.

Related reading:
- `FORAGING_REFLEX_PROGRESS.md` — what was already tried on the PD reflex
- Branch `foraging_with_neural_reuse` — Luka’s spawn curriculum, signed progress, circuit steering motif
- Paper: [Neural Circuit Architectural Priors for Embodied Control](https://arxiv.org/abs/2201.05242) (swimming prior; foraging is our extension)

---

## Goal

Show that a **simple innate architecture** is enough for foraging once swimming is acquired — either:

- a tiny fixed sensory reflex on top of the swim circuit, or
- a sparse sensory motif inside the circuit itself

— not a large learned controller. Judge success with **physics metrics**, not training reward.

---

## Current baseline (do not lose this)

Best verified config so far (Sonja):

```python
dict(
    network='ncap',
    task='foraging',
    controller='foraging',
    task_kwargs=dict(speed_reward_weight=0.0, alignment_reward_weight=0.5),
    swimmer_kwargs=dict(include_speed_control=True),
)
# angle_gain=16.0, rate_gain=112.0
```

- **~47%** physics-only success (dist &lt; 0.15 at some point, 30 episodes)
- Still unsolved (~53% fail)
- Aggregate `log.csv` reward is **misleading** in this env — never rank configs by it alone

---

## Shared principles

1. **Physics metrics only** for decisions (near-food + true eat).
2. **One change at a time** until eval noise is under control; independently good changes often fail to stack.
3. **Train on controlled spawn; evaluate on stock spawn.**
4. Prefer mechanisms that keep the “simple innate” story intact.
5. Leave pirouette / klinotaxis controllers to Emmanuel.

---

## Phase 0 — Lock a shared evaluation protocol

Do this once. Everything else depends on it.

### Steps

1. Define two success metrics on fresh rollouts (no training reward):
   - **Near-food:** `min_dist < 0.15` (current diagnostic)
   - **True eat:** `min_dist < food_size` (currently `0.02`)
2. Stratify episodes by **start-distance bins** (and optionally start-angle bins).
3. Report mean ± CI over **≥5 seeds × ≥100 episodes** (30-ep / 1-seed comparisons are too noisy; SE ≈ 9% at 47%).
4. Always include:
   - floor: `controller=None`
   - ceiling: `mlp_foraging`
   - current best: `gain16x` + alignment reward
5. Separate plots/tables for:
   - **train distribution** (fixed target distance)
   - **held-out stock spawn** (dm_control default box)

### Done when

One shared eval script/notebook cell both people use, producing the same tables for every new run.

---

## Phase 1 — Fix the task geometry (highest expected impact)

Start-distance correlation with outcome was ≈ −0.935. Stock spawn makes many episodes trivial or near-impossible.

### Steps (from Luka’s branch — use fully)

1. Add **fixed `target_distance`** placement: random bearing, fixed range.
2. Train with a short curriculum, e.g. `0.5 → 0.8 → 1.2`.
3. Keep **stock spawn** as the real generalization test (`target_distance=None`).
4. Debug first on **`swim_to_ball`** (one target, no respawn confound), then **`foraging`**.
5. For foraging multi-pellet episodes, use **respawn near the nose** so the next pellet stays local.

### Decision rule

| Result on fixed-distance train | Interpretation | Next |
|---|---|---|
| Large jump (e.g. 47% → 70%+) | Spawn difficulty was the main bottleneck | Simplify controller story; move to Phase 2 bake-off |
| Little change | Reflex / circuit coupling still broken | Phase 2 + Phase 3 interventions |

### Done when

Same controller shows clearly higher success under fixed distance than stock, with stratified tables explaining where wins come from.

---

## Phase 2 — One shared reward (no cocktails)

### Default team reward

Use Luka’s design as the shared default:

- **Signed progress:** `Δd = prev_dist - dist` (can be negative; do **not** clamp at 0)
- Keep a small **proximity / tolerance** term for stability
- Re-baseline `_prev_target_dist` after respawn (avoid fake penalties for eating)
- Keep `speed_reward_weight=0` for foraging

### Reference only (do not stack yet)

- Sonja’s **alignment-only** reward (`alignment_reward_weight=0.5`) — current best at 47%
- `eaten_bonus` alone
- Alignment-gated progress (peaked ~43%)

### Explicitly park

- Additive combinations of progress + alignment + gated-progress + eaten_bonus  
  (repeatedly worse than the best single term)

### Done when

One reward config is declared team default; all bake-offs use it unless an ablation explicitly varies reward.

---

## Phase 3 — Controller bake-off (main scientific comparison)

Run under **identical** Phase 0 eval + Phase 1 spawn + Phase 2 reward.

| Arm | Setup | Claim |
|---|---|---|
| **A. Reflex** | Sonja’s fixed PD `gain16x` → `right/left/speed`; train mostly `bneuron_turn` | Innate swim circuit + tiny fixed sensory reflex is enough |
| **B. Circuit steering** | Luka’s `include_target_steering` (~8 sparse sign-free head weights); `controller=None` | Innate swim circuit + sparse sensory motif is enough |
| **C. Ceiling** | `mlp_foraging` | Remaining headroom |
| **D. Floor** | `controller=None` (no steering motif) | Chance / swim-only baseline |

Order:

1. `swim_to_ball` + fixed distance  
2. `swim_to_ball` + stock spawn eval  
3. `foraging` + fixed distance (+ local respawn)  
4. `foraging` + stock spawn eval  

### Rules

- Do **not** combine A+B in the first bake-off (muddies the claim).
- Rank by Phase 0 physics metrics, multi-seed.
- Optional later: hybrid only if both A and B clearly plateau.

### Done when

A clear winner (or clear tie) on stock-spawn physics success, with MLP ceiling reported.

---

## Phase 4 — If A and B both plateau

Apply **one** intervention at a time on the better of A/B:

1. **Phase-gated turns** — Sonja’s finding: `right`/`left` are washed out on half the oscillator cycle; only fire on the live half.
2. **`n_turn_joints = 2` (or 3)** — steer with more than the head module.
3. **Soften saturation** — correction is saturated ~95% of steps at high gain; slightly lower gains or graded unsaturated commands so `bneuron_turn` gets a real learning signal.
4. **Longer / staged curriculum** — short distance → longer → stock spawn.
5. **Swim init transfer** — start from a competent swim checkpoint; then train only turn / steer params.
6. **Physics-success checkpoint picker** — among saved checkpoints, pick by physics success rather than episode return (kills “reward lies” at selection time).

### Still parked unless bake-off forces a revisit

- Learnable `angle_gain` / `rate_gain` (inconclusive; high noise)
- Adaptive P/D-agreement gain modulation (~43%, within noise of progress-alone)
- Bang-bang / hard deadband speed gating (shown to kill thrust)
- Cubed alignment speed gates
- Reward term stacking

---

## Phase 5 — Optional training-protocol upgrades

Not required to start, useful once the bake-off is clean:

1. Pretrain / init from good swim weights.
2. Freeze swim weights; train only steering-related params.
3. Longer runs only after Phase 0 shows a real gap (Phase 2 of the progress log already showed length alone doesn’t fix bimodality).
4. ES later if desired (paper’s swimming method); if used, consider scoring candidates by physics success rather than shaped return.
5. Multi-seed is mandatory before claiming any new SOTA over 47%.

---

## Possible improvements checklist

Use as a backlog. Prefer top items first.

### Task / environment

- [ ] Fixed `target_distance` training curriculum
- [ ] Stock-spawn held-out eval
- [ ] Local food respawn around nose
- [ ] `swim_to_ball` as debug task before foraging
- [ ] True-eat metric (`food_size`) alongside near-food (`0.15`)
- [ ] Stratified success tables by start distance / angle
- [ ] Optional: widen proximity margin of the stock tolerance term (today margin ≈ `5 * food_size` is tiny)

### Reward

- [ ] Signed progress as team default
- [ ] Alignment-only as reference baseline
- [ ] Reward-component normalization / scaling for the critic
- [ ] Avoid additive stacking until single terms are stable multi-seed

### Controllers / architecture

- [ ] Bake-off: PD reflex vs `include_target_steering` vs MLP vs none
- [ ] Phase-gated turn commands
- [ ] `n_turn_joints > 1`
- [ ] Soften high-gain saturation so learning isn’t dead
- [ ] Optional hybrid (reflex + circuit steering) only after both solo arms plateau

### Training / eval hygiene

- [ ] ≥5 seeds × ≥100 eval episodes
- [ ] Physics-success-based checkpoint selection
- [ ] Swim → forage transfer init
- [ ] Shared eval script used by both Sonja and Luka

### Explicitly out of scope (this plan)

- [ ] Emmanuel’s pirouette / run-and-tumble controllers
- [ ] Further micro-sweeps of adaptive-gain / learnable-gain without multi-seed evidence
- [ ] Trusting `log.csv` mean/max alone

---

## Suggested immediate sequence

1. Port Luka’s spawn curriculum + signed progress into the working branch (or merge branches).
2. Implement Phase 0 shared eval.
3. Re-measure floor / `gain16x+align` / MLP under fixed-distance train + stock eval.
4. Run Phase 3 bake-off on `swim_to_ball`, then foraging.
5. Only if needed: Phase 4 interventions, one at a time.
6. Write up whichever simple innate inductive bias wins, with MLP as ceiling.

---

## Definition of “solved enough”

On **stock spawn**, with a **simple** controller (Arm A or B), multi-seed physics success is:

- clearly above `controller=None`, and
- competitive with `mlp_foraging`,

without pirouette and without a pile of stacked reward hacks.

---

## Notes from prior dead ends (do not re-learn the hard way)

- Training reward can look great while distance never falls (swim-speed confound).
- Close starts can “succeed” via wiggle without navigation — stratify by distance.
- Measurement bugs (post-reset pose, non-rotating cameras, reused `Camera`) have faked conclusions before — verify world-frame diagnostics carefully.
- Combining independently good changes has failed repeatedly; treat stacking as a new experiment, not a free win.
- 16× gains helped directional precision; 32× broke it — don’t assume monotonicity.
