# Foraging Reflex — Progress Log

What's been tried on `make_foraging_reflex` (branch `improve-foraging-reflex`), in the order it happened, including the dead ends. If you just want the current state, read the **TL;DR** and **Current best config** sections and skip the rest.

## TL;DR

- **Current best config**: the fixed `gain16x` reflex (`angle_gain=16.0`, `rate_gain=112.0`) plus `alignment_reward_weight=0.5` on the reward. **47% physics-only success rate** (up from a 23% baseline with no changes at all). Still far from solved — more than half of episodes still fail.
- **Two newer levers, each independently at ~43%, not yet beating 47%, currently being tuned**: a multiplicative "alignment-gated progress" reward, and an adaptive-gain reflex that reacts to its own P/D-term agreement. Combining them did not help (see below).
- **The single most important methodology point**: raw training reward (`log.csv` mean/max) is actively misleading in this environment and cannot be trusted on its own. Every number below is a **physics-only success rate** — the fraction of 30 fresh episodes where the worm's distance to food drops below 0.15 units at some point, measured directly from simulator state, independent of whatever reward function trained the checkpoint. This distinction mattered enormously and is the reason several early "great results" turned out to be nothing at all (see Phase 0 and Phase 3).

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

---

## Current best config

```python
dict(network='ncap', task='foraging', controller='foraging',
     task_kwargs=dict(speed_reward_weight=0.0, alignment_reward_weight=0.5),
     swimmer_kwargs=dict(include_speed_control=True))
# angle_gain=16.0, rate_gain=112.0 (make_foraging_reflex's defaults)
```
**47% physics-only success (14/30 fresh episodes)** — still the best found, and still not solved (53% of episodes fail).

## Open questions

- Item 2 and item 4 both settled at 43% after tuning — neither beat 47%. Both hit a clean inverted-U dose-response and no other parameter is obviously left to tune within each mechanism as designed.
- Why do independently-good changes so consistently fail to stack? Checked directly at least 4 separate times (progress+align+eaten, align+no-weight-sharing, item4+alignment) — never fully explained.
- Learnable gains (item 1) needs multi-seed evaluation to properly settle whether it helps at all — not yet done.
- The gait's own baked-in curvature (Phase 3-4) has never been directly addressed, only worked around via better steering/reward — an open structural question about the base circuit.

*Full blow-by-blow detail, including every measurement bug and retraction preserved for provenance, lives in the project's running memory log — ask if you want the unabridged version.*
