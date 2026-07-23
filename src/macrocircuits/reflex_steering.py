"""Fixed, non-learned steering reflexes for NCAP.

Each derives the circuit's turn signals directly from the already-egocentric
to_target/to_obstacle vector a task puts in the observation, instead of learning that
mapping with a network -- the `MLPController` in `controllers` is the learned
alternative, and `controllers.CONTROLLERS` is the registry a run picks either from.

A reflex is a plain closure `reflex(observations) -> (right, left, speed)`, each
`(..., 1)` in `[0, 1]`; it holds no parameters, so a run using one still trains or
evolves only the circuit's own `bneuron_turn` weight.
"""

import torch


def make_foraging_reflex(
    n_joints, approach_distance=0.3, angle_gain=16.0, rate_gain=112.0,
):
    """Builds a reflex that steers toward the nearest food, easing off as it closes in.

    A P+D controller on the *bearing* to the food, not on the raw lateral offset. Raw
    lateral can't tell "10 degrees off dead ahead" from "10 degrees off dead behind" --
    both give a small lateral value -- so the proportional term uses
    angle = atan2(lateral, forward) instead, which is 0 dead ahead and +-pi dead behind.

    The derivative term is the line-of-sight rate (missile-guidance style): how fast the
    bearing is sweeping, which distinguishes "40 degrees off and holding" from "40
    degrees off and closing fast". It is computed from the head's own translation only.
    A version that also included the head's yaw rate converged beautifully for a few
    steps and then stalled facing the food without ever closing -- the yaw rate turned
    out to be dominated by the swimming gait's own wobble, not by real target motion.

    The gains are large because that is the one lever with a clean monotonic effect:
    swept at 4x, 8x, 16x and 32x a naive first guess, each step tightened
    post-convergence heading accuracy (~37 -> ~26.5 -> ~21.2 degrees average) until 32x
    broke the trend. 16x/112x is the best point on that sweep. Note the sweep was run on
    a mislabelled pair of axes (`improve-foraging-reflex`, which had forward and lateral
    swapped); the law here is the same law on the *measured* axes, so treat the exact
    numbers as a good starting point rather than as re-validated optima.

    Egocentric convention, measured directly against the simulator (food AHEAD gives
    to_target = [0, -0.5]; food LEFT gives [+0.5, 0]):
        lateral = to_target[0]   (+ve => food is to the worm's LEFT)
        forward = -to_target[1]  (+ve => food is AHEAD)
    and driving NCAP's `left` input turns the worm to its own left.

    Assumes observation layout: joints, to_target, body_velocities, ... (see
    Swim.get_observation) -- body_velocities' first two entries are the head's own local
    (vx, vy), since 'head' is body 0 in the model.
    """
    target_slice = slice(n_joints, n_joints + 2)        # [lateral, longitudinal]
    head_vel_slice = slice(n_joints + 2, n_joints + 4)  # head-local [vx, vy]

    def reflex(observations):
        to_target = observations[..., target_slice]
        lateral = to_target[..., 0, None]
        forward = -to_target[..., 1, None]
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        angle = torch.atan2(lateral, forward)  # 0 = dead ahead, +ve = food to the LEFT

        # d/dt atan2(lateral, forward), with the target held still so the whole rate
        # comes from the worm's own motion: the target's head-frame offset changes at
        # -v_head, i.e. d(lateral)/dt = -vx and d(forward)/dt = +vy (forward is -y).
        head_vel = observations[..., head_vel_slice]
        vx = head_vel[..., 0, None]
        vy = head_vel[..., 1, None]
        angle_rate = -(forward * vx + lateral * vy) / distance.clamp(min=1e-3) ** 2

        correction = angle_gain * angle + rate_gain * angle_rate  # +ve => turn left
        left = correction.clamp(min=0, max=1)
        right = (-correction).clamp(min=0, max=1)

        # Ease off near the food (so it doesn't shoot past) and while badly misaligned.
        # Kept deliberately mild: sharper gates (cos^3, a hard deadband) were each
        # tried and each traced *worse*, because a sustained near-zero speed signal
        # suppresses the undulation that produces real thrust in the first place.
        near = (distance / approach_distance).clamp(min=0, max=1)
        aligned = torch.cos(angle).clamp(min=0, max=1)
        speed = near * aligned

        return right, left, speed

    return reflex


def make_turn_reflex(n_joints, direction='left', strength=1.0, speed=1.0):
    """Hardcoded steering: hold NCAP's turn input to one side, every step.

    Unlike the other reflexes, this reads nothing from the observation -- it is not a
    foraging or avoidance strategy, just a direct actuator test. It drives the circuit's
    `right`/`left` turn signal constantly to one side so you can confirm that turn
    control actually bends the swimmer, and in which direction, independent of any
    sensing. `direction` picks the side; `strength` in [0, 1] sets how hard it turns;
    `speed` sets the constant forward drive (only used when the circuit is built with
    include_speed_control=True, otherwise ignored).

    In NCAP, `right` is an excitatory input to the dorsal head B-neuron and `left` to the
    ventral one (see SwimmerModule.forward), so the two directions bend the head to
    opposite sides. Which visual direction that is on screen is exactly what this reflex
    lets you check. Note that `strength=1.0` (a *constant, saturating* turn signal) tends
    to overpower the head oscillator so the body curls to one side instead of swimming in
    a clean arc; a smaller strength (e.g. ~0.3) biases the heading while leaving enough
    oscillation for real forward thrust, which traces a much clearer turning path.
    """
    if direction not in ('left', 'right'):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    def reflex(observations):
        # observations[..., :1] is only a template: it carries the right batch shape,
        # dtype and device for the constant turn signal, nothing about it is read.
        template = observations[..., :1]
        on = torch.ones_like(template) * strength
        off = torch.zeros_like(template)
        right = on if direction == 'right' else off
        left = on if direction == 'left' else off
        return right, left, torch.ones_like(template) * speed

    return reflex


def make_turn_left_reflex(n_joints):
    """Constant left turn -- see `make_turn_reflex`."""
    return make_turn_reflex(n_joints, direction='left')


def make_turn_right_reflex(n_joints):
    """Constant right turn -- see `make_turn_reflex`."""
    return make_turn_reflex(n_joints, direction='right')


def make_steer_to_food_reflex(n_joints, strength=0.75, gain=3.0):
    """Hardcoded navigation: turn toward the food and swim. A fixed reflex (no learning),
    built on the MEASURED egocentric convention. The simplest thing that works: a pure
    proportional turn on the bearing, with no rate term and no speed gating, where
    `make_foraging_reflex` is the full P+D version of the same idea.

    Convention (see the check_convention diagnostic):
        lateral = to_target[0]   (+ve => food is to the worm's LEFT)
        forward = -to_target[1]  (+ve => food is AHEAD)
    angle = atan2(lateral, forward) is 0 dead ahead, +ve when food is to the left; the
    turn command drives NCAP's `left` input to turn the worm to its left. `strength` is
    the calibrated turn magnitude (0.75 gives the tightest clean turn; see
    make_turn_reflex); `gain` sets how sharply the command grows with the angle.

    On the foraging task this reaches ~90% physics-only success on its own -- the point
    being that once the geometry is right, navigating to food needs no learning at all.
    """
    target_slice = slice(n_joints, n_joints + 2)

    def reflex(observations):
        to_target = observations[..., target_slice]
        lateral = to_target[..., 0, None]
        forward = -to_target[..., 1, None]
        angle = torch.atan2(lateral, forward)     # 0 = ahead, +ve = food to the LEFT
        u = torch.tanh(gain * angle) * strength    # +ve => turn left
        left = u.clamp(min=0)
        right = (-u).clamp(min=0)
        speed = torch.ones_like(u)
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

