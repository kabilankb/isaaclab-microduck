"""The kick reward must stay scaled to the target speed, and stay inside the stack.

The mjlab recipe documents the invariant "at-target payoff ~= +3/step (weight ~=
3/target)" and warns to rescale the weights whenever the target moves. That file
then breaks its own rule: the target was raised 0.25 -> 1.0 m/s with the weights
left at 12.0 / -4.0, so the kick paid 12/step against an 8/step standing stack.
The policy did the rational thing and leaned over to farm the kick, which is what
drove `Episode_Reward/upright` to ~1.5% of its weight while every other term rose.

These tests keep the two numbers tied together, so changing the target can never
again silently quadruple the kick's share of the reward mass.

CPU-only: reading cfg objects needs no simulator.
"""

from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import (
    TARGET_BALL_SPEED,
    MicroduckBallKickFlatEnvCfg,
)

#: mjlab's documented at-target payoff, in reward per step.
AT_TARGET_PAYOFF = 3.0

#: Positive terms that pay for standing rather than for kicking.
_STANDING_TERMS = ("support_foot_grounded", "upright", "height_stand", "pose_stand_legs", "pose_stand_neck")


def _rewards():
    return MicroduckBallKickFlatEnvCfg().rewards


def test_at_target_payoff_matches_the_documented_rule():
    r = _rewards()
    assert r.ball_forward_velocity.weight * TARGET_BALL_SPEED == AT_TARGET_PAYOFF


def test_kick_does_not_outweigh_the_whole_standing_stack():
    # The regression that collapsed `upright`: when the kick alone pays more than
    # every standing term combined, leaning to strike harder is the better trade.
    r = _rewards()
    standing = sum(getattr(r, name).weight for name in _STANDING_TERMS)
    assert r.ball_forward_velocity.weight * TARGET_BALL_SPEED < standing


def test_overshoot_penalty_is_weaker_than_the_capped_reward():
    # Erring hard must stay cheaper than not kicking at all, or the policy learns
    # not to touch the ball.
    r = _rewards()
    assert abs(r.ball_speed_overshoot.weight) < r.ball_forward_velocity.weight


def test_overshoot_zero_crossing_stays_at_four_times_the_target():
    # mjlab's asymmetry: net kick reward reaches zero at 4x the target speed.
    r = _rewards()
    payoff = r.ball_forward_velocity.weight * TARGET_BALL_SPEED
    crossing = TARGET_BALL_SPEED + payoff / abs(r.ball_speed_overshoot.weight)
    assert crossing == 4.0 * TARGET_BALL_SPEED


def test_overshoot_weight_is_negative():
    # The function returns >= 0, so a positive weight would pay for overshooting.
    assert _rewards().ball_speed_overshoot.weight < 0
