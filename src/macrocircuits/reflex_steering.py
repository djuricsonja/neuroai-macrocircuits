"""Fixed, non-learned steering reflexes for NCAP (plus one partial exception, see below).

Each derives the circuit's turn signals directly from the already-egocentric
to_target/to_obstacle vector a task puts in the observation, instead of learning that
mapping with a network -- the `MLPController` in `controllers` is the learned
alternative, and `controllers.CONTROLLERS` is the registry a run picks either from.

A reflex is a plain closure `reflex(observations) -> (right, left, speed)`, each
`(..., 1)` in `[0, 1]`; it holds no parameters, so a run using one still trains or
evolves only the circuit's own `bneuron_turn` weight.

`LearnableForagingReflex` below is the one exception: same fixed P+D formula as
`make_foraging_reflex`, but its two gains are `nn.Parameter`s instead of constants, so
it sits between a fully fixed reflex and the full `MLPController` -- structural prior
kept, only the two coefficients learned.
"""

import torch
import torch.nn as nn


def make_foraging_reflex(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0,
):
    """Builds a reflex that steers toward the nearest food, slowing down as it gets
    close to avoid overshooting past it.

    Steers using the angle to the food (atan2(lateral, forward)), not the raw lateral
    coordinate alone. Raw lateral can't tell "food is 10 degrees off dead ahead" apart
    from "food is 10 degrees off dead behind" -- both give a small lateral value, but
    the second needs a hard turn-around, not a gentle nudge. The angle is 0 when food
    is dead ahead and approaches +-pi when it's dead behind, so it stays large exactly
    when a big correction (including a near-reversal) is actually needed.

    angle_gain was 0.7 (full saturation ~57 degrees off) to fix an earlier oscillation:
    at gain=2.0, speed stayed pinned at 1 regardless of heading error, so a saturating
    turn command with no proportional taper just overshot past facing the food over
    and over. Now that speed below is also gated on alignment (see there), a weak
    gain traced as a stable *orbit* instead -- the turn was too gentle to ever out-pace
    the forward drift and close the distance, just correct enough to keep circling at
    a roughly constant radius. Raising it back up to 1.2 pulled the orbit radius in
    (traced directly) but didn't break it: reacting only to the current angle (a P
    controller) can't tell "40 degrees off and holding" apart from "40 degrees off and
    sweeping past" -- both call for the same correction under pure P, even though the
    orbit is exactly the second case, over and over.

    angle_rate adds a D (derivative) term on top: how fast the angle itself is
    changing, so a fast sweep gets corrected harder than the same instantaneous angle
    would justify on its own -- directly opposing the sweeping motion that is the
    orbit, rather than just reacting to a snapshot of it. The full rate of the
    egocentric angle is, exactly,
        d(angle)/dt = omega + (forward * vy - lateral * vx) / distance**2
    (the classic "line-of-sight rate" from missile guidance, verified directly against
    the simulator: finite-differenced angle across real steps vs. this formula from a
    single instant, correlation > 0.999), where omega is the head's own yaw rate and
    (vx, vy) its own local linear velocity -- both already sitting unused in
    body_velocities, right after to_target in the observation.

    A first version used the full expression, omega included, and traced as a genuine
    fix for the orbit -- but then got stuck facing the food without ever closing the
    distance. Tracing omega and the velocity term separately showed why: omega is
    dominated by the worm's own gait -- an undulating swimmer's head sweeps side to
    side as its body wave travels, by an order of magnitude more than anything
    reflecting real target geometry -- so the "derivative" term was really just
    reacting to the worm's own stride, flipping right/left on the gait's rhythm rather
    than on genuine sweep-past-the-target information, which fights the traveling wave
    the body needs to actually produce net thrust. Dropping omega and keeping only the
    velocity term isolates the part that's actually about the target: how fast the
    line to it is swinging due to *translation* relative to it, excluding the worm's
    own spin entirely. That term is naturally small (it's divided by distance**2, and
    distance is usually of order 1 or more), so rate_gain is much larger here (8.0)
    than it would need to be if omega -- order 1 by itself -- were still in the mix.

    Even with that fixed, some episodes still only orbit instead of closing in. Tracing
    it down further: right_control and left_control aren't just weaker on the "wrong"
    side of a turn -- from a clean, neutral test (forced turn signal, isolated from
    everything else), right_control has zero effect during the half of NCAP's own
    oscillator cycle where the oscillator is driving the *ventral* side, and
    left_control has zero effect during the *dorsal* half. This isn't a quirk of one
    trained checkpoint (checked against a fresh random init, a second random init, and
    the actual trained weights -- same pattern every time): pushing a muscle against
    whichever side the oscillator already dominates gets washed out by the muscle's
    own cross-inhibition; pushing with it is what has real leverage. So a correction
    computed every step regardless of phase is, half the time, being sent to a control
    that structurally cannot act on it right now.

    A phase gate (only ever sending right during NCAP's own oscillator's dorsal-active
    half, left during the ventral-active half -- since pushing a muscle against
    whichever side the oscillator already dominates gets washed out by the muscle's
    own cross-inhibition, confirmed directly with a clean neutral-pose test) was tried
    next, since right/left_control turned out to only have real leverage on their
    matching half. It's mechanistically well-founded, but a whole one-variable-at-a-
    time sweep on top of it (gain doubling, more turn joints, less action noise,
    shorter oscillator_period) never beat this version's own plain, ungated PD
    controller -- every variant landed in the low-single-digit-to-teens range, well
    under this version's 139 (peak test/episode_score/mean) and 43 (final). Shelved for
    now, not proven wrong, just not yet demonstrated better; see
    foraging_reflex_debugging_saga memory for the full comparison.

    deadband_degrees started as just that: below that many degrees of misalignment,
    correction is forced to exactly 0 rather than whatever small (possibly gait-noise-
    driven) value the P+D formula would otherwise give, so the worm swims straight,
    undisturbed, once it's already roughly facing the food. It traced as the best
    mid-training numbers of the whole project (peak test/episode_score/mean 86, max
    397) -- but checking actual world-frame head position (not just distance-to-food)
    showed why that's still not the real fix: total distance covered over an episode
    was substantial (path length up to ~3.9), but net displacement from start to end
    was near zero every time (~0.02-0.05) -- it travels, sometimes far, and reliably
    loops back close to where it started rather than committing to a direction. Smooth
    speed control (below, previously `near * cos(angle).clamp(0,1)`) let the worm keep
    translating even while still correcting a real misalignment, which is exactly what
    turns imperfect per-stroke correction into a closed loop instead of a straight
    path.

    So speed is now bang-bang, using the same deadband boundary as the turn gate above:
    zero speed (pure rotation, no translation at all) whenever outside the deadband,
    and only the normal distance-scaled speed once already aligned. This is meant to
    make it structurally impossible for translation and an unresolved heading error to
    happen at the same time, rather than continuously blending the two and hoping the
    correction wins the race against the drift it causes.

    First version of that kept deadband_degrees at 10 and made things worse, not
    better (max excursion from the start position collapsed to under 0.4, down from
    up to 1.7) -- traced why: a fresh controlled test confirmed the brake (speed=0)
    genuinely doesn't cripple turning (yaw was just as strong at speed_control=0 as at
    1.0), so that wasn't it. The real problem was the threshold itself: the gait's own
    stroke-locked wobble routinely swings the instantaneous angle through 30-60
    degrees even while the *average* heading is fine (same phenomenon that broke the
    first D-term attempt, back when omega was still in angle_rate) -- so a 10-degree
    window flickered in and out on nearly every step from that wobble alone, never
    sustaining speed long enough to build real propulsion (confirmed directly: traced
    rspeed sitting at exactly 0 on all but a handful of steps). Widened to 35 degrees.

    That fixed the flickering (real path length and excursion came back), but net
    displacement over a full episode was still ~0 every time -- still looping, not
    reaching. Chased one more hypothesis before abandoning bang-bang for now: is
    translation itself entangled with having an active turn signal, so the "aligned,
    go straight" (0, 0) state the deadband produces is actually a dead zone? Checked
    directly: with a truly constant right_control (anywhere from 0 to 1.0, held fixed
    for hundreds of steps) the head barely translates at all regardless of value --
    true for a fresh untrained circuit and for two differently-trained checkpoints
    alike. So neither a sustained zero nor a sustained max turn signal produces real
    forward progress on its own; real, substantial excursion only ever showed up under
    the reflex's natural continuously-varying signal (its ordinary operation). That
    means deliberately holding the correction at either extreme for a while -- which
    is exactly what both the smooth alignment gate and the bang-bang gate do -- fights
    against what actually produces thrust, rather than helping. Dropped both; back to
    plain, continuous P+D on both turn and speed (no deadband, no bang-bang, matching
    the version that scored 139/43 before any of this), with angle_gain and rate_gain
    raised substantially (1.2->4.0, 8.0->28.0) instead.

    That test uncovered a measurement bug, not a real behavior: the diagnostic used to
    track world-frame position was capturing the head's position *after* the episode
    had already reset internally on the final step, so every single episode's
    "ending" position was actually the *next* episode's fresh reset point -- making
    every version tested look like it looped back to its own start regardless of what
    it actually did. Fixed (capture position before checking for reset, not after),
    and the real picture is different: net displacement is substantial in most
    episodes and the worm travels in a genuinely committed direction, not a closed
    loop. So the earlier "add a constant trim for the body's inherent curvature"
    conclusion doesn't hold either -- retracted.

    What actually predicts success, checked directly across many episodes: the angle
    between the worm's net direction of travel and the true initial direction to food.
    Under ~20 degrees off, score is usually large (real success); over ~30 degrees
    off, score is almost always near zero. So the real remaining problem is
    directional precision, not locomotion or looping -- the committed direction is
    often just substantially wrong. Comparing first-half vs second-half displacement
    direction within the same episodes confirmed why: the correction genuinely
    converges over an episode (misalignment reliably drops well after the halfway
    point), but real thrust already happens during the early not-yet-converged phase,
    permanently dragging the whole episode's net direction away from the true
    bearing.

    Tried fixing this by suppressing speed harder during the unconverged phase --
    cubing the alignment term (cos(angle)**3), then gating on the full |correction|
    magnitude instead of angle alone ("settled"). Both traced as *worse*, the same
    "sustained low speed cripples real thrust" mechanism established earlier (see the
    bang-bang history above): dose-response across three gate strengths is clean,
    plain cos(angle) (mildest) beats both. Reverted to plain cos(angle).

    Instead, raising angle_gain/rate_gain to converge *faster* (rather than
    suppressing speed while converging) has been the one lever with a real, positive,
    monotonic effect: checked at 4x, 8x, 16x and 32x the original pd_noomega values,
    each step tightening the post-convergence direction accuracy (highgain ~1x: ~37
    degrees average; 2x more (8x total): ~26.5; 2x more again (16x total): ~21.2) --
    until 32x, which broke the trend and got worse (~44 degrees, both by this metric
    and by aggregate score). 16x (here) is the best point found on this sweep.

    Assumes observation layout: joints, to_target, ..., body_velocities, ..., timestep
    (see Swim.get_observation and tonic's time_feature wrapper) -- body_velocities'
    first 2 entries are the head's own local (vx, vy), since 'head' is body 0 in the
    model.
    """
    target_slice = slice(n_joints, n_joints + 2)  # [forward, lateral], head-egocentric
    head_vel_slice = slice(n_joints + 2, n_joints + 4)  # [vx, vy], head-local

    def reflex(observations):
        to_target = observations[..., target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        # lateral is negative when the target is physically to the right (verified
        # directly against the simulator -- the opposite of what the sign looks like
        # it should mean), so it's negated here before computing the angle.
        angle = torch.atan2(-lateral, forward)  # 0 = dead ahead, +-pi = dead behind

        head_vel = observations[..., head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        correction = angle_gain * angle + rate_gain * angle_rate

        right = correction.clamp(min=0, max=1)    # food's to my right -> turn right
        left = (-correction).clamp(min=0, max=1)  # food's to my left -> turn left

        # Tried tightening this twice more (cos(angle)**3, then gating on the full
        # |correction| i.e. "settled") to fix the early-episode-commitment problem --
        # both traced worse, not better: net displacement collapsed the same way
        # bang-bang's did (0.02-0.31 vs 0.36-1.85 for this plain version), the same
        # "sustained low speed cripples real thrust" failure from step 23. Reverted to
        # the plain, mild cos(angle) gate -- the mildest version tested is still the
        # best-performing one; gating harder is not the right lever, even though the
        # early-convergence diagnosis itself (checked directly, real) still stands.
        near = (distance / approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed

    return reflex


def _leaky_clamp_straight_through(x, min_val=0.0, max_val=1.0, leak=0.01):
    """clamp(x, min_val, max_val), but with a small leak-slope gradient outside that
    range instead of the hard clamp's exact zero -- a straight-through estimator: the
    forward value is bit-for-bit the plain hard clamp (`hard`, detached, so it
    contributes no gradient of its own), and `soft - soft.detach()` is exactly 0 in
    value but carries `soft`'s gradient backward. Confirmed necessary empirically, not
    just in theory: on a real trained rollout, the reflex's correction term sat exactly
    at 0 or 1 on 95.3% of all steps (see LearnableForagingReflex), so the plain hard
    clamp gave angle_gain/rate_gain a real gradient on well under 5% of steps --
    consistent with them moving by under 0.5% over 20k training steps. `soft` matches
    the hard clamp exactly inside `[min_val, max_val]` (unchanged gradient there), and
    only differs in the saturated tails, where it leaks a small (`leak`-scaled) slope
    instead of none.
    """
    hard = x.clamp(min=min_val, max=max_val).detach()
    soft = torch.where(
        x < min_val, min_val + leak * (x - min_val),
        torch.where(x > max_val, max_val + leak * (x - max_val), x),
    )
    return hard + (soft - soft.detach())


class LearnableForagingReflex(nn.Module):
    """Same P+D steering law as `make_foraging_reflex`, but angle_gain/rate_gain are
    `nn.Parameter`s instead of hand-tuned constants.

    Sits between the fixed reflex and the full `MLPController`: keeps the reflex's
    hand-derived formula and observation slicing (the structural prior), but lets
    whatever optimizer trains the rest of the actor also adjust these two numbers
    directly, instead of freezing them at whatever a manual gain sweep happened to
    land on (see `make_foraging_reflex`'s docstring for that sweep: 1.2/8.0 -> 4.0/28.0
    -> 8.0/56.0 -> 16.0/112.0 best -> 20x/24x noisy, not clearly better -> 32x breaks
    down).

    Registers as a submodule exactly the way `MLPController` does -- assigning an
    `nn.Module` to `SwimmerActor.controller` after its own `super().__init__()` auto-
    registers it via PyTorch's `nn.Module.__setattr__` -- so no changes to
    models.py/training.py's optimizer setup are needed: angle_gain and rate_gain are
    already part of `actor.parameters()` the moment this is plugged in.

    Initialized at the gain sweep's own best point (16.0/112.0) as a warm start, not
    from scratch: PPO adjusting a reflex already in the right ballpark is a very
    different (much easier) problem than discovering from a naive first guess that
    these need to be an order of magnitude larger, which is what the manual sweep spent
    a whole session doing.

    First version used a plain `correction.clamp(min=0, max=1)` for right/left, on the
    theory that it might starve angle_gain/rate_gain's gradient whenever saturated --
    confirmed directly (not just in theory): trained for 20k steps, angle_gain/
    rate_gain moved under 0.5% from their initial values, and a direct instrumented
    rollout showed the correction term saturated on 95.3% of all steps. That version's
    physics-only success rate (27%, 30 episodes) was statistically indistinguishable
    from the fixed-gain baseline's 23% -- the learnable capacity existed but couldn't
    actually receive a training signal. Switched right/left to
    `_leaky_clamp_straight_through`, which keeps the forward value bit-identical (so
    the actual steering behavior, and hence physics, is unchanged) but gives
    angle_gain/rate_gain a small nonzero gradient even while saturated.
    """

    def __init__(
        self, n_joints, approach_distance=0.3, angle_gain_init=16.0, rate_gain_init=112.0,
        leak=0.1,
    ):
        super().__init__()
        self.target_slice = slice(n_joints, n_joints + 2)
        self.head_vel_slice = slice(n_joints + 2, n_joints + 4)
        self.approach_distance = approach_distance
        self.angle_gain = nn.Parameter(torch.tensor(float(angle_gain_init)))
        self.rate_gain = nn.Parameter(torch.tensor(float(rate_gain_init)))
        self.leak = leak

    def forward(self, observations):
        to_target = observations[..., self.target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(-lateral, forward)  # 0 = dead ahead, +-pi = dead behind

        head_vel = observations[..., self.head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        correction = self.angle_gain * angle + self.rate_gain * angle_rate

        right = _leaky_clamp_straight_through(correction, leak=self.leak)
        left = _leaky_clamp_straight_through(-correction, leak=self.leak)

        near = (distance / self.approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed


def make_foraging_reflex_learnable(
    n_joints, approach_distance=0.3, angle_gain_init=16.0, rate_gain_init=112.0, leak=0.1,
):
    """Learnable-gain counterpart of `make_foraging_reflex` -- see
    `LearnableForagingReflex` for the rationale and mechanics."""
    return LearnableForagingReflex(
        n_joints, approach_distance, angle_gain_init, rate_gain_init, leak,
    )


def make_foraging_reflex_adaptive(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0, adapt_strength=0.3,
):
    """Same P+D steering law as `make_foraging_reflex`, but the gain is modulated every
    step by whether the P and D terms currently agree or disagree.

    Motivated by the Baldwin-effect/meta-RL Neuromatch lecture's "learn from recent
    experience within a lifetime" idea (see foraging_reflex_debugging_saga memory,
    "item 4") -- but a genuine multi-step memory (e.g. an EMA of recent oscillation)
    hits the same problem noted back at the phase-gate step: PPO trains on randomly
    shuffled minibatches of transitions, not ordered trajectories, so any state that
    depends on step order gets silently scrambled unless the training loop is built to
    preserve trajectory order (ours isn't). This is a memory-free approximation:
    instead of detecting oscillation from history, it uses only the *current* instant's
    own P and D terms, which are already computed either way.

    When `p_term` (proportional, reacts to the current angle) and `d_term`
    (derivative, reacts to how fast that angle is closing) share a sign, the heading
    error is still compounding -- push harder. When they oppose, the error is already
    closing fast on its own, so easing off avoids the classic PD overshoot (the same
    mechanism that motivated adding the D term in the first place, back when angle_gain
    alone caused an oscillating orbit). `tanh` on each term keeps `agreement` bounded
    to roughly [-1, 1] regardless of how large the raw terms get; `modulation` is
    clamped to [0.3, 1.7] so it can meaningfully scale the correction without ever
    zeroing it out or blowing it up.
    """
    target_slice = slice(n_joints, n_joints + 2)
    head_vel_slice = slice(n_joints + 2, n_joints + 4)

    def reflex(observations):
        to_target = observations[..., target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(-lateral, forward)

        head_vel = observations[..., head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        p_term = angle_gain * angle
        d_term = rate_gain * angle_rate
        agreement = torch.tanh(p_term) * torch.tanh(d_term)
        modulation = (1.0 + adapt_strength * agreement).clamp(min=0.3, max=1.7)
        correction = modulation * (p_term + d_term)

        right = correction.clamp(min=0, max=1)
        left = (-correction).clamp(min=0, max=1)

        near = (distance / approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed

    return reflex


class LearnableAdaptiveForagingReflex(nn.Module):
    """`make_foraging_reflex_adaptive`'s P/D-agreement gain modulation, but
    `adapt_strength` is an `nn.Parameter` instead of a hand-swept constant, warm-started
    at the sweep's own best point (0.5 -- see foraging_reflex_debugging_saga memory:
    0.3 -> 37%, 0.5 -> 43% (peak), 0.7 -> 23%).

    `angle_gain`/`rate_gain` stay fixed (item 1's own attempt to learn those was
    inconclusive -- noise-dominated at this training budget, not worth compounding
    into a second learnable-parameter experiment). Registers as a submodule of
    `SwimmerActor` exactly the way `LearnableForagingReflex` does.

    Uses `_leaky_clamp_straight_through` at *both* clamps on `adapt_strength`'s
    gradient path -- the modulation clamp (`[0.3, 1.7]`) and the final turn-command
    clamp (`[0, 1]`) -- rather than waiting to discover the same dead-gradient problem
    empirically the way item 1's first version did.
    """

    def __init__(
        self, n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0,
        adapt_strength_init=0.5, leak=0.1,
    ):
        super().__init__()
        self.target_slice = slice(n_joints, n_joints + 2)
        self.head_vel_slice = slice(n_joints + 2, n_joints + 4)
        self.approach_distance = approach_distance
        self.angle_gain = angle_gain
        self.rate_gain = rate_gain
        self.adapt_strength = nn.Parameter(torch.tensor(float(adapt_strength_init)))
        self.leak = leak

    def forward(self, observations):
        to_target = observations[..., self.target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(-lateral, forward)

        head_vel = observations[..., self.head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        p_term = self.angle_gain * angle
        d_term = self.rate_gain * angle_rate
        agreement = torch.tanh(p_term) * torch.tanh(d_term)
        raw_modulation = 1.0 + self.adapt_strength * agreement
        modulation = _leaky_clamp_straight_through(
            raw_modulation, min_val=0.3, max_val=1.7, leak=self.leak,
        )
        correction = modulation * (p_term + d_term)

        right = _leaky_clamp_straight_through(correction, leak=self.leak)
        left = _leaky_clamp_straight_through(-correction, leak=self.leak)

        near = (distance / self.approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed


def make_foraging_reflex_adaptive_learnable(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0,
    adapt_strength_init=0.5, leak=0.1,
):
    """Learnable-`adapt_strength` counterpart of `make_foraging_reflex_adaptive` --
    see `LearnableAdaptiveForagingReflex` for the rationale and mechanics."""
    return LearnableAdaptiveForagingReflex(
        n_joints, approach_distance, angle_gain, rate_gain, adapt_strength_init, leak,
    )


def make_foraging_reflex_phase_aware(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0,
    oscillator_period=60, phase_strength=0.3,
):
    """Same P+D steering law as `make_foraging_reflex`, but the gain is modulated by
    whether the *current* correction's direction actually has any leverage right now.

    Step 12 of the debugging saga found a hard structural fact, not a tuning issue:
    `right_control`/`left_control` have *exactly zero effect* during half of NCAP's own
    oscillator cycle (its own cross-inhibition washes out a muscle pushed against
    whichever side the oscillator currently dominates) -- confirmed on multiple random
    inits and the trained checkpoint alike. A hard phase *gate* (silence the signal
    entirely during the dead half) was tried on top of an earlier, cruder reflex
    (`ncap_reflex_foraging_phasegated`) and wasn't a clear win -- but that was a much
    blunter instrument (fully zero vs. a graduated boost/damp) on a much worse base
    reflex than today's. This is the softer version: boost the correction when its
    requested direction currently has real leverage, ease it when it doesn't, instead
    of an all-or-nothing gate.

    Phase is read the same way `SwimmerActor.forward` derives it from the observation's
    own appended timestep feature (`timestep_transform=(-1, 1, 0, 1000)`), and the same
    way `SwimmerModule.forward` derives which half is dorsal-/ventral-active -- no
    memory needed, so this doesn't hit the PPO-shuffled-minibatch problem a genuine
    multi-step memory would (see `make_foraging_reflex_adaptive`'s docstring).
    """
    target_slice = slice(n_joints, n_joints + 2)
    head_vel_slice = slice(n_joints + 2, n_joints + 4)
    half_period = oscillator_period / 2

    def reflex(observations):
        to_target = observations[..., target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(-lateral, forward)

        head_vel = observations[..., head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        correction = angle_gain * angle + rate_gain * angle_rate

        # Same timestep decode SwimmerActor itself uses, then the same dorsal/ventral
        # split SwimmerModule uses to pick which oscillator half is currently active.
        timestep = (observations[..., -1, None] + 1) / 2 * 1000
        phase = timestep.round().remainder(oscillator_period)
        # +1 during the dorsal-active half (right_control has real leverage there),
        # -1 during the ventral-active half (left_control has real leverage there).
        leverage_side = torch.where(phase < half_period, 1.0, -1.0)
        # correction > 0 requests right; matches leverage_side=+1 when that request
        # currently has real leverage. Same logic, opposite sign, for left.
        effective = torch.sign(correction) * leverage_side
        modulation = (1.0 + phase_strength * effective).clamp(min=0.3, max=1.7)
        correction = modulation * correction

        right = correction.clamp(min=0, max=1)
        left = (-correction).clamp(min=0, max=1)

        near = (distance / approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed

    return reflex


def make_foraging_reflex_distance_scaled(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0, distance_strength=0.3,
):
    """Same P+D steering law as `make_foraging_reflex`, but the gain is boosted while
    the food is still far away and eased off again once it's close.

    Motivated by the "runway" framing confirmed in the debugging saga (Step 39):
    success declines gradually with starting distance, consistent with an episode
    needing enough steps to converge before it's too late -- a farther target
    benefits from a stronger, faster-converging correction while there's still time,
    the way raising angle_gain/rate_gain across the whole gain sweep did. Once close,
    the original motivation for the P controller's own gain (Step 4 -- too strong a
    gain overshoots past facing the target) applies again, so easing off there avoids
    reintroducing that oscillation right at the finish.

    Reuses `near` (already computed for the existing speed gate) instead of a new
    formula: despite the name, it is 0 right at the food and rises to 1 once beyond
    `approach_distance`, exactly the "how far, saturating" signal this needs.
    """
    target_slice = slice(n_joints, n_joints + 2)
    head_vel_slice = slice(n_joints + 2, n_joints + 4)

    def reflex(observations):
        to_target = observations[..., target_slice]
        forward = to_target[..., 0, None]
        lateral = to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(-lateral, forward)

        head_vel = observations[..., head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = (forward * vy - lateral * vx) / distance.clamp(min=1e-3) ** 2

        near = (distance / approach_distance).clamp(min=0, max=1)
        distance_modulation = 1.0 + distance_strength * near
        correction = distance_modulation * (angle_gain * angle + rate_gain * angle_rate)

        right = correction.clamp(min=0, max=1)
        left = (-correction).clamp(min=0, max=1)

        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed

    return reflex


def make_obstacle_avoidance_reflex(n_joints, reaction_distance=0.6):
    """Builds a reflex that steers away from the nearest obstacle, scaled by proximity.

    Same observation layout as before, but now also uses distance (the magnitude
    of the to_obstacle vector) to scale the response: a distant obstacle barely
    registers, a close one dominates. reaction_distance is a fixed threshold.
    """
    obstacle_slice = slice(n_joints, n_joints + 2)

    def reflex(observations):
        to_obstacle = observations[..., obstacle_slice]  # [forward, lateral]
        lateral = to_obstacle[..., 1, None]
        distance = torch.norm(to_obstacle, dim=-1, keepdim=True)

        # 1 when right on top of the obstacle, 0 once farther than reaction_distance.
        urgency = (1 - distance / reaction_distance).clamp(min=0, max=1)

        right = (-lateral).clamp(min=0, max=1) * urgency
        left = lateral.clamp(min=0, max=1) * urgency
        speed = torch.ones_like(right)
        return right, left, speed

    return reflex

