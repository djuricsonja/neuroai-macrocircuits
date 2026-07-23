# Foraging Reflex — Progress Log

What's been tried on `make_foraging_reflex` (branch `improve-foraging-reflex`), in the order it happened, including the dead ends. If you just want the current state, read the **TL;DR** and **Current best config** sections and skip the rest.

## TL;DR

- **Current best config**: the fixed `gain16x` reflex (`angle_gain=16.0`, `rate_gain=112.0`) plus `alignment_reward_weight=0.5` on the reward. **47% physics-only success rate** on the original single-seed measurement — **but a multi-seed re-check ties it with the no-steering floor (26.7% vs 25.0%, n=60, see the "Adopting the shared team plan" section)**. Treat 47% as provisional, not settled.
- **A real, structural bug was found**: `make_foraging_reflex`'s `forward`/`lateral` axis labels are swapped relative to the true physical axes (confirmed directly against the simulator), producing a consistent 90-degree error in its internal steering angle. Predates this whole project. Deliberately not fixed yet — planned as its own isolated ablation once there's a stable baseline (see "Structural bug found" section).
- **Fixed-distance spawn training (Phase 1 of the shared team plan) shows the reflex does do something real**: 60% in-distribution / 40% stock-spawn-generalization vs. floor's 10%/25% under the identical manipulation — a clean, well-controlled result, obtained *despite* the axis bug still being unfixed.
- **Two older levers, each independently settled at 43%**: a multiplicative "alignment-gated progress" reward, and an adaptive-gain reflex that reacts to its own P/D-term agreement. Combining them did not help (see below).
- **Three structural reflex tweaks (phase-aware gain, distance-scaled gain, learnable adapt_strength) all failed to beat baseline** — see Phase 8. A velocity-alignment reward reached 33% and EMA-smoothing it was a dead end — see Phase 9. Across the whole project, only *reward-side* changes (and now, the spawn-distance curriculum) have ever produced a repeatable win; every purely structural reflex tweak has failed.
- **The single most important methodology point**: raw training reward (`log.csv` mean/max) is actively misleading in this environment and cannot be trusted on its own. Physics-only success rate — the fraction of fresh episodes where the worm's distance to food drops below 0.15 units at some point, measured directly from simulator state — is the only metric ever used to rank configs. This distinction mattered enormously and is the reason several early "great results" turned out to be nothing at all (see Phase 0 and Phase 3).
- **Now following a shared team plan** (`FORAGING_FORWARD_PLAN.md`, Sonja + Luka) rather than ad-hoc exploration — see the dedicated section below for what's changed as a result.
- **Open decision, pending team input**: the plan's chosen team-default reward (signed progress) generalizes to stock spawn worse than alignment does under the same fixed-distance setup (28.3% vs 40.0%, see Phase 2) — not yet resolved which one carries into the Phase 3 bake-off.

## How the reflex works, briefly

`make_foraging_reflex` turns the egocentric vector to the food into `right`/`left`/`speed` signals for NCAP's circuit. It's a P+D controller: proportional term on the angle to the food, derivative term on how fast that angle is changing (a "line-of-sight rate," missile-guidance style), both feeding into a saturating turn command, plus a distance- and alignment-gated speed. See the extensive docstring in `src/macrocircuits/reflex_steering.py` for the exact formula and its full derivation history — this document is the narrative version of that history plus everything from the newest round of work.

---

## Phase 0 — Foundational bugs (getting it to steer at all)

| # | What was tried | Result / problem found |
|---|---|---|
| 1 | Added a gain to amplify a too-weak raw lateral-offset signal | Steering was still structurally broken underneath |
| 2 | Switched to `angle = atan2(lateral, forward)` instead of raw lateral | Reflex was blind to "food is behind me" — raw lateral can't tell that apart from a small in-front offset |
| 3 | **Found a real sign bug**: `lateral` is *negative* when food is physically to the worm's right — the opposite of what the original code comment assumed, and never actually checked against the simulator. Fixed: `atan2(-lateral, forward)` | This was a foundational bug likely present since before this branch started |
| 4 | Lowered `angle_gain` to fix oscillation | A saturating turn command with no taper near alignment just kept overshooting past facing the food |
| 5 | **The swim-speed confound**: a great-looking reward score (~730, very stable) turned out to be almost entirely from swimming fast, with zero real food-seeking underneath (distance never decreased, actually got worse) | This is the moment that established: **never trust the aggregate reward alone in this environment, ever again** |

## Phase 1 — Building the P+D controller

| # | What was tried | Result / problem found |
|---|---|---|
| 6 | Isolated the food-seeking reward (`speed_reward_weight=0`) to remove the swim-speed confound | Confirmed food-seeking really was ~0 before this point |
| 7 | Traced distance/angle over a long rollout | Worm was circling the food at a constant radius — a classic "pursuit curve" (full speed + continuous bearing correction never converges on a stationary target) |
| 8 | Gated speed by alignment (`speed = near * cos(angle)`) | Orbit shrank but didn't close — a P-only controller can't tell "40° off and holding" from "40° off and sweeping past" |
| 9 | Added a full D term including the head's own yaw rate (`omega`) | Beautiful short-term convergence, then got stuck facing the food without ever closing distance — `omega` turned out to be dominated by the worm's own gait wobble, not real target information |
| 10 | Dropped `omega`, kept only the velocity-based part of the rate term | **Best result at the time** — one to two orders of magnitude above every earlier version |
| 11 | Still some residual orbiting in specific episodes | A memoryless reflex reacting only to the current instant is close to luck when gain/loss of ground alternates every 10-20 steps |
| 12 | **Major structural finding**: a clean, deconfounded test showed `right_control`/`left_control` have exactly **zero effect** during half of NCAP's own oscillator cycle (the circuit's own cross-inhibition washes it out) | Not a tuning problem — a hard structural fact about the circuit, confirmed on multiple random inits and the trained checkpoint alike |
| 13 | Phase-gated turn control to only fire during the "live" half of the cycle | Mechanistically sound, but didn't clearly beat the plain (ungated) PD controller in testing — shelved, not proven wrong |
| 14 | Found and fixed two camera bugs that had been silently corrupting how video evidence was judged all along (tracking cameras don't rotate with heading; a reused `Camera` object silently freezes mid-video) | Unrelated to the reflex itself, but affected trust in every earlier visual check |
| 15 | Considered switching to Evolution Strategies (NCAP's own paper uses ES) | Abandoned — full run was 1hr+ vs. ~4 min for a PPO quick check, and ES only saves a single final checkpoint |
| 16 | One-variable-at-a-time sweep: gain boost, extra turn joint, lower action noise, shorter oscillator period, a 10° deadband | None clearly beat the step-10 result; several looked great mid-training then collapsed by the final checkpoint |

## Phase 2 — The "mean/max lies" correction

| # | What was tried | Result / problem found |
|---|---|---|
| 17 | **Major correction**: checked `min`/`std`, not just `mean`/`max`, across only 5 test episodes per epoch | Every single config's `min` sat near zero, every epoch — the whole night's mean-based ranking had been measuring how big/frequent lucky outliers were, not typical behavior. Nothing had actually "worked reliably." |
| 18 | Ran a 5x-longer (1e5-step) confirmation with finer checkpoints | Same bimodal (mostly-fail, rarely-huge-success) pattern at every single checkpoint — not a training-duration problem |

## Phase 3 — Two more major corrections, then the real bottleneck

| # | What was tried | Result / problem found |
|---|---|---|
| 19 | Logged starting angle/distance vs. outcome across 40 fresh episodes | Starting **distance**, not angle, was overwhelmingly the driver (corr = -0.935) |
| 20 | **Second major correction**: watched a rendered replay of a "successful" close-start episode | The worm barely moved — its default undulating wiggle alone was enough to graze the reward zone when food started close, without any real navigation. Confirmed via world-frame position tracking: net displacement ~0 in every episode. |
| 21-23 | Chased "bang-bang"/deadband speed gating (force zero speed while misaligned) across several widths and variants | Consistently worse — confirmed mechanism: **any sustained extreme turn signal (fully off or fully on) kills real thrust**; only the reflex's natural, continuously-varying signal produces genuine forward progress |
| 24 | Reverted to plain continuous P+D, raised gain 3x | No better — net displacement identical to every earlier gain level |
| 25 | **Trained a plain swim task with no foraging at all, as a control** | Found the *exact same* closed-loop, near-zero-net-displacement signature. **The circling was never the foraging reflex's fault — it's the base swimming gait itself**, which the reward had never pressured away from a curve since it only measures instantaneous velocity, never net position. |
| 26 | (a "constant trim" fix proposed here was retracted, see #27) | |
| 27 | **Third major correction**: the entire "worm loops back to start" narrative (steps 20-26) turned out to be a **measurement bug** — the diagnostic script captured position *after* an episode-ending step, which by then reflects the *next* episode's reset position | Real locomotion was substantially more directed than every one of the last 7 steps had concluded |

## Phase 4 — Direction accuracy: the real bottleneck, and the gain sweep

| # | What was tried | Result / problem found |
|---|---|---|
| 28 | Sharpened the alignment speed-gate (`cos(angle)³`) | No clear improvement |
| 29 | **Confirmed cleanly**: angle between the worm's actual net travel direction and the *true* direction to food predicts success almost monotonically (< 20° usually succeeds, > 30° almost always fails) | Reframed the whole remaining problem as **directional precision**, not locomotion capability |
| 30 | Tried suppressing speed harder during the not-yet-converged early phase (twice, different strengths) | Both worse — same "sustained low speed kills thrust" mechanism as steps 21-23 |
| 31-34 | **Swept gain 4x → 8x → 16x → 32x the original values** | Each step tightened direction accuracy — until 32x broke the trend. **16x/112x ("gain16x") established as the sweet spot.** This is the current fixed-reflex baseline: **23% physics-only success.** |
| 35 | Re-ran the starting-condition sweep on `gain16x` | Distance still dominates, but angle now matters too (unlike the original checkpoint). "Successful foraging radius" grew ~2.5x (0.3 → 0.77) from the original reflex. |

## Phase 5 — Reward shaping (23% → 47%)

Added three new opt-in reward terms to `envs.py` (`progress_reward_weight`, `alignment_reward_weight`, `eaten_bonus`), all tested in isolation on top of the fixed `gain16x` reflex:

| Config | Physics-only success | Note |
|---|---|---|
| baseline (no reward changes) | 23% | |
| eaten_bonus alone | 30% | sparse — only fires *after* success, gives no guidance during the actual approach |
| gain20x | 30% | |
| use_weight_sharing=False | 37% | independent circuit-capacity change, unrelated to reward |
| all 3 reward terms combined (additive) | 37% | **worse than alignment or progress alone** |
| progress reward alone | 43% | |
| **alignment reward alone** | **47%** | **best result of the whole project so far** |
| gain24x | 17% | had the *best-looking aggregate reward of the entire investigation* (mean 122.6) — yet the worst physics-only result of this sub-sweep. Another concrete case of aggregate reward actively lying. |
| alignment + use_weight_sharing=False combined | 20% | worse than *either* alone |

**Pattern confirmed repeatedly (at least 3 separate times by this point): improvements found independently do not stack.** Combining two independently-good changes reliably made things *worse*, not better. Not fully explained — candidate causes are genuine interference, a harder joint-optimization problem, or plain training noise — but real and consistent enough to take seriously.

---

## Phase 6 (most recent session) — Learnable steering gains ("item 1"): inconclusive

**Idea**: make `angle_gain`/`rate_gain` learnable `nn.Parameter`s (warm-started at 16.0/112.0) instead of frozen hand-tuned constants, so PPO fine-tunes them jointly with the circuit's own weight.

- Registered cleanly as a new controller (`'foraging_learnable'`), same mechanism `MLPController` already uses — no training-loop changes needed.
- **Found and fixed a real bug**: the turn command's hard clamp saturates on ~95% of steps in a real rollout, so gradient to the two new parameters vanished almost everywhere. Confirmed: they moved <0.5% over 20k training steps. Fixed with a straight-through gradient estimator (forward pass unchanged, backward pass gets a small gradient even while saturated).
- Even after the fix, results across every variant tried (different `leak` values, a small genetic-algorithm search over 5 gain/leak genomes) ranged 13-37% with no reliable pattern. **A repeat of the literal same genome** (an accidental RNG-reset bug in the GA's mutation step) gave 33% one run and 13% the next — proof the whole spread is training noise at this evaluation budget, not real signal.
- **Verdict**: the mechanism is correctly implemented, but there is no evidence yet that it beats the fixed baseline. Properly settling this would need multi-seed averaging (not done — cost tradeoff).
- Side note: this investigation's own manual gain/weight sweeps (all of Phase 0-5 above) are, in retrospect, exactly a genetic algorithm done by hand — a Neuromatch lecture on the Baldwin effect prompted building an actual small one instead of continuing to guess by hand.

## Phase 7 (most recent session) — Item 2 & item 4: two new levers

**Item 2 — alignment-gated progress reward** (multiplicative combination instead of additive): `reward += weight * progress * max(alignment, 0)`. Directly targets the Phase-5 finding that additively stacking progress + alignment made things worse — now progress is only rewarded in proportion to how aligned the worm currently is.
- weight=10: 13% (much worse)
- **weight=20: 43%** (peak — ties progress-alone, clearly beats the naive additive combination's 37%)
- weight=40: 33% (overshot)
- Clean, fairly narrow inverted-U — 20.0 (its original, untuned value) happened to already sit right at the peak.

**Item 4 — adaptive gain via instantaneous P/D-term agreement** (no reward change, no new learnable parameters): when the proportional and derivative terms currently *agree* (heading error still compounding), push harder; when they *disagree* (error already closing fast on its own), ease off to avoid overshoot. A true multi-step "remember recent oscillation" version was considered but is incompatible with how PPO trains here (shuffled minibatches break anything that depends on step order) — this is a memory-free, single-instant approximation of that idea instead.
- adapt_strength=0.3: 37% (matches `use_weight_sharing=False`'s independent win)
- **adapt_strength=0.5: 43%** (peak)
- adapt_strength=0.7: 23% (worse, back near baseline)
- Same inverted-U shape as the original angle_gain/rate_gain sweep back in Phase 4.

**Both tuning sweeps are now complete and land on the same number: 43%, tied with plain progress-reward-alone, still short of alignment-reward-alone's 47%.**

**Combined (item 2 + item 4 together)**: the first-ever test of pairing a reward-side change with a steering-law-side change (every earlier combination in Phase 5 paired two changes of the *same* kind). Result: 37% — identical to item 4 alone, no better, no worse. No interference, but no synergy either.

**Current plan**: since the combination didn't produce a new best result, focus is on tuning item 2 and item 4 independently rather than trying further new mechanisms.

## Phase 8 — Three more structural tweaks ("give the worm more information"): all failed to beat baseline

All isolated tests (fixed reflex, no reward changes, same conditions as the 23% `gain16x` baseline):

| Idea | What it was | Result |
|---|---|---|
| Phase-aware gain modulation | Boost/damp the correction continuously based on whether it currently has real leverage over NCAP's own oscillator cycle (Step 12 found this leverage is *exactly zero* during half the cycle — a softer, graduated successor to the old hard phase gate) | **23% — exactly baseline** |
| Distance-scaled gain | Boost gain while food is far (faster convergence before the "runway" runs out), ease off once close | **23% — exactly baseline** |
| Learnable `adapt_strength` | Item 4's own modulation strength as an `nn.Parameter`, with the straight-through gradient fix applied *proactively* this time. The parameter genuinely moved (0.5 → 0.471 over 20k steps, ~6%, vs. item 1's <0.5%) — confirms the fix works | **30%** — still worse than just hard-coding 0.5 (43%) |

**Bigger-picture pattern, now confirmed across two full sessions: only reward-side changes have ever produced a repeatable win over baseline** (alignment 47%, progress/item2/item4 ~43%). **Every purely structural reflex tweak has failed** — original phase gate, bang-bang, deadband, sharpened alignment (Phase 0-4), and now all three of the above. 47% is not being accepted as a final answer — work has shifted back to the reward side.

## Phase 9 (in progress) — Back to the reward side, with a twist

A useful realization while planning this phase: the "no memory in the policy" constraint that blocked a true recurrent version of item 4 does **not** apply to the reward function. `Swim.get_reward()` already keeps state across steps (`self._prev_target_dist`) and runs once per real step during rollout collection — *before* anything enters PPO's shuffled minibatches. So reward-side memory (e.g. smoothing a noisy signal over recent steps) is safe, even though the same idea inside the policy wasn't.

**Idea 1 — velocity-alignment reward**: the existing `alignment_reward_weight` measures the *nose's* orientation, but the nose is known to sweep 30-60° every stroke from gait wobble alone (Step 9) even when net heading is fine — and it's *net displacement direction*, not nose pose, that actually determines success (Steps 27-29). New `velocity_alignment_reward_weight` rewards the cosine similarity between the head's own velocity vector and the direction to food instead — an instantaneous proxy much closer to the thing that's actually been shown to matter.
- **weight=0.5: 33%** — a real improvement over the 23% no-shaping baseline, but below plain nose-alignment's 47%. Likely because velocity, being a derivative-like quantity, is *more* exposed to gait-stroke noise than pose is — the same reason the original D-term attempt with omega failed (Step 9), not less as first assumed.

**Idea 2 — EMA-smooth the alignment signal**: average the (velocity-based) alignment signal over recent steps instead of using it raw, to filter the gait-wobble noise directly. Safe to keep state for this (unlike inside the policy/reflex) since `get_reward()` runs once per real step before anything reaches PPO's shuffled minibatches.
- alpha=0.02 (slow smoothing): **23%**
- alpha=0.2 (fast smoothing): **20%**
- Both worse than the 33% raw (unsmoothed) version at both tested strengths — **a dead end regardless of degree**, not a tuning problem. Likely a distinct failure mode from the minibatch-shuffling issue: averaging a *reward* over recent steps dilutes which action gets credited for the outcome ("credit-assignment lag"), which is a different problem from the policy-side memory constraint that motivated trying this in the first place.

**Idea 3 (front-loading the reward early in the episode) was not run** — deprioritized once the team's shared `FORAGING_FORWARD_PLAN.md` (see below) redirected effort toward the fixed-distance spawn curriculum instead.

---

## Structural bug found: `forward`/`lateral` are swapped in `make_foraging_reflex`

While reading teammate Luka's `foraging_with_neural_reuse` branch (which uses the opposite indexing, with an explicit justification in its comments), a check against the live simulator confirmed **our own reflex's `forward`/`lateral` axis labels are swapped relative to the true physical axes** — not just a naming quibble, a real bug in the steering law.

**How it was verified** (not taken on faith from Luka's comment): placed the food at known egocentric offsets and read the *actual* observation array via `env.task.get_observation(physics)`, cross-checked against the project's own long-established, always-correct `forward_velocity = -physics.named.data.sensordata['head_vel'][1]` convention from the base swim reward. Confirmed: observation component 0 is the local **lateral** (x) axis, component 1 is local **longitudinal** (y, true forward = **-y**) — the opposite of what `reflex_steering.py`'s `forward = to_target[...,0]` / `lateral = to_target[...,1]` assumes.

Plugging the confirmed raw values into the reflex's actual `atan2(-lateral, forward)` formula gives a clean, consistent **90-degree rotation error** across all four cardinal cases (dead-ahead food reads as 90°, dead-lateral reads as 0°, dead-behind reads as -90°, other-lateral reads as -180°). This predates this whole project — inherited from the original codebase, not introduced by any change in this branch.

**Why this doesn't invalidate the project's own world-frame diagnostics** (the "angle-off predicts success" finding from Steps 27-29, used in every `comprehensive_sweep.py`-style check all along): those diagnostics reconstruct the true world-frame bearing by rotating the *raw* `[obs[0], obs[1]]` pair through `head_orientation`, in their original order — an operation that is correct regardless of what the reflex's own code chooses to *name* each component. So the diagnostics have been measuring the real thing all along; it's specifically the reflex's own internal steering angle that has been wrong.

**Not yet fixed, on purpose**: fixing this at the same time as the fixed-distance spawn curriculum (below) would confound two effects in one training run. Plan is to test it as its own isolated ablation once there's a stable baseline to compare against — still open.

---

## Adopting the shared team plan (`FORAGING_FORWARD_PLAN.md`)

Sonja + Luka's shared forward plan formalizes scope going forward: physics metrics only (never training reward) for every decision, one change at a time, train on controlled spawn distance / evaluate on stock spawn as the generalization check, and MLP is a ceiling reference only — not a priority to keep retraining. Emmanuel + Lorenzo's branches (their own separate pirouette/run-and-tumble direction) are out of scope; Luka's circuit-steering branch is the other half of the planned controller bake-off.

### Phase 0 — Shared eval protocol

Built `src/macrocircuits/evaluation.py`: a shared physics-only eval module both Sonja and Luka can call on any trained checkpoint. Reads **ground-truth simulator state** (`physics.nose_to_target_dist()`, a dedicated eat counter added to `Swim`) rather than any controller's own internal observation slicing — deliberately independent of the axis-label bug above. Reports near-food (`min_dist<0.15`) and true-eat (`min_dist<food_size`) rates, stratified by start-distance/start-angle bins, mean ± 95% CI across seeds.

**Stock-spawn read (n=60: 2 seeds x 30 episodes each, no MLP)**:

| Arm | near_food | true_eat |
|---|---|---|
| floor (`controller=None`) | 25.0% ± 11.0 | 0% |
| current-best (reflex + alignment, the 47% config above) | 26.7% ± 11.3 | 0% |

**These are statistically indistinguishable** — a long way from the 47% recorded above. Not believed to be a harness bug (the respawn code path never even fired in either arm's 60 episodes, and the rollout mechanics match the historical `comprehensive_sweep.py` pattern exactly). Most likely explanation: the original 47% came from a single seed x 30 episodes, and the plan document itself already flags "SE ≈ 9% at 47%" for that measurement — wide enough that landing near 47% while the true rate sits around 25-30% is entirely plausible. This is the exact failure mode Phase 0 exists to catch. Not further resolved with more seeds (judgment call: accept the noisy read and move on rather than spend ~1-3 more hours of eval compute) — left as an open flag rather than a settled correction.

### Phase 1 — Fixed-distance spawn curriculum

Ported Luka's `target_distance` mechanism into `envs.py`'s `Swim` class (`_place_target`): pins the target to a fixed distance at a random bearing instead of dm_control's wide random-box spawn, with automatic local-nose respawn during multi-pellet foraging once `target_distance` is set. `target_distance=None` preserves the exact stock behavior. (Separately, `progress_reward_weight` already implemented Luka's signed-progress reward design correctly — nothing needed porting there.)

Tested directly on `foraging` (skipping the plan's suggested `swim_to_ball` debugging detour, since the team's actual focus is foraging) at a fixed spawn distance of 0.8, same reward config as the 47%/26.7% "current-best" arm:

| Arm | Trained @ | Evaluated @ | near_food (n=60) |
|---|---|---|---|
| floor | fixed 0.8 | fixed 0.8 | 10.0% ± 7.7 |
| floor | fixed 0.8 | stock spawn | 25.0% ± 11.0 |
| reflex + alignment | fixed 0.8 | fixed 0.8 | **60.0% ± 12.5** |
| reflex + alignment | fixed 0.8 | stock spawn | **40.0% ± 12.5** |

**Large jump** (the plan's own decision-rule term): the reflex arm more than doubled in-distribution (26.7% -> 60%), and floor did not improve — it got *worse* (25% -> 10%), since a fixed moderate distance removes the "lucky, nearly-adjacent stock spawn" free wins floor was otherwise living off. This is a clean control: the same manipulation helps the arm that can actually use directional information and hurts the one that can't, so it isn't just "closer targets are easier for everyone." The gain partially generalizes back to stock spawn too (40% vs floor's 25% there) — a real edge, not pure overfitting to the fixed distance. All of this with the axis-label bug above still unfixed.

**MLP ceiling**: explicitly deprioritized per team decision — not worth spending training time on until the NCAP side (reflex vs. Luka's circuit-steering) is settled; trivial to add back later. A far-better-trained MLP checkpoint (500k steps) already exists on a teammate's `obstacles-foraging-no-controller` worktree if a ceiling number is wanted without retraining, though it was trained under the stock combined reward rather than this project's isolated food-seeking setup.

### Phase 2 — Testing the plan's team-default reward (signed progress) against alignment

The plan names Luka's signed-progress reward as the shared team default, demoting alignment to "reference only." Tested it under the identical Phase 1 setup (fixed distance 0.8, same reflex, `progress_reward_weight=20.0` — the value already settled in Phase 7 — in place of `alignment_reward_weight=0.5`):

| Reward | Fixed 0.8 (in-distribution) | Stock spawn (generalization) |
|---|---|---|
| alignment (0.5) | 60.0% ± 12.5 | **40.0% ± 12.5** |
| progress (20.0) | 55.0% ± 12.7 | **28.3% ± 11.5** |
| floor (reference) | 10.0% ± 7.7 | 25.0% ± 11.0 |

Both reward terms clearly beat floor in-distribution and are statistically tied with each other there. But on the generalization check, alignment holds a real edge while progress barely clears floor (28.3% vs. floor's 25.0% — CIs [16.8,39.8] vs [14,36], mostly overlapping). **This is a tension with the plan's stated default**, not yet resolved: at n=60 it could still be noise, but the direction is consistent with what's been true all project long (alignment has always been the strongest single lever, back to the original 47%/23% comparison). Decision on which reward to carry into the Phase 3 bake-off is pending team discussion — not unilaterally overridden here.

**Next**: once the reward choice is settled, run the actual Phase 3 controller bake-off (reflex vs. Luka's circuit-steering vs. floor, no MLP for now). The axis-label bug fix is still an open, deliberately-deferred ablation.

---

## Current best config

```python
dict(network='ncap', task='foraging', controller='foraging',
     task_kwargs=dict(speed_reward_weight=0.0, alignment_reward_weight=0.5),
     swimmer_kwargs=dict(include_speed_control=True))
# angle_gain=16.0, rate_gain=112.0 (make_foraging_reflex's defaults)
```
**47% physics-only success (14/30 fresh episodes)** — the original single-seed measurement. **Caveat, not yet resolved**: a proper multi-seed re-check (n=60, 2 seeds x 30 episodes, see Phase 0 above) measured this same config at 26.7% ± 11.3 on stock spawn, statistically tied with the no-steering floor (25.0% ± 11.0). Most likely explanation is that the original 47% was noisy (single seed, plan's own SE ≈ 9% estimate), not that the reflex does nothing — the same config under a fixed-distance spawn curriculum (Phase 1) shows a clear, well-controlled 60% in-distribution / 40% stock-generalization result that floor does not share. Treat "47%" as provisional pending the team's Phase 3 bake-off, not as settled.

## Open questions

- Item 2 and item 4 both settled at 43% after tuning — neither beat 47%. Both hit a clean inverted-U dose-response and no other parameter is obviously left to tune within each mechanism as designed.
- Why do independently-good changes so consistently fail to stack? Checked directly at least 4 separate times (progress+align+eaten, align+no-weight-sharing, item4+alignment) — never fully explained.
- Learnable gains (item 1) needs multi-seed evaluation to properly settle whether it helps at all — not yet done.
- The gait's own baked-in curvature (Phase 3-4) has never been directly addressed, only worked around via better steering/reward — an open structural question about the base circuit.
- **The `forward`/`lateral` axis-label bug is still unfixed.** Once there's a stable baseline (post Phase 2/3), test the fix as its own isolated ablation — open question is whether correcting it helps, hurts, or does nothing (PPO's other free parameters may already be partially compensating for the consistent 90-degree error).
- **Is 47% real?** The multi-seed re-check ties it with floor at n=60; the fixed-distance curriculum result suggests the reflex does do something real, but the two findings haven't been fully reconciled with a large enough shared sample size yet.

*Full blow-by-blow detail, including every measurement bug and retraction preserved for provenance, lives in the project's running memory log — ask if you want the unabridged version.*
