"""Heading must be worth enough to compete with speed, and be densely rewarded.

A measured policy walked at the commanded 0.4 m/s but ARCED instead of holding a
line: straightness 0.74, mean |yaw rate| 1.19 rad/s against a commanded 0.0. Two
causes, both the same shapes that had already bitten linear tracking:

* heading was worth 2.0 against 12.0 for linear (Gaussian + dense), so the policy
  optimised speed and ignored where it pointed; and
* at std sqrt(0.5) the Gaussian scores exp(-2.83) ~ 0.06 at 1.19 rad/s of error --
  nearly no payment and, worse, nearly no gradient.

CPU-only.
"""

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.running.running_env_cfg import MicroduckRunningFlatEnvCfg
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import MicroduckVelocityFlatEnvCfg

CFGS = (MicroduckVelocityFlatEnvCfg, MicroduckRunningFlatEnvCfg)


@pytest.mark.parametrize("C", CFGS)
def test_dense_yaw_term_exists(C):
    assert C().rewards.yaw_tracking.func is mdp.yaw_rate_error_l1


@pytest.mark.parametrize("C", CFGS)
def test_yaw_term_is_a_cost_with_a_negative_weight(C):
    # The function returns |error| >= 0. A positive weight would PAY for yawing --
    # the double-negation trap that has produced reward-farmed behaviour before.
    assert C().rewards.yaw_tracking.weight < 0


@pytest.mark.parametrize("C", CFGS)
def test_yaw_error_is_clipped(C):
    # A 25 cm robot tumbles at 3.5-5.5 rad/s naturally; an unclipped term would price
    # a tumble far above anything the policy can act on.
    import inspect

    sig = inspect.signature(mdp.yaw_rate_error_l1)
    assert sig.parameters["max_error"].default <= 3.0


@pytest.mark.parametrize("C", CFGS)
def test_heading_can_compete_with_speed(C):
    """Not parity -- speed should still lead -- but not a sixth of it either."""
    r = C().rewards
    linear = r.track_linear_velocity.weight + r.forward_speed.weight
    angular = r.track_angular_velocity.weight + abs(r.yaw_tracking.weight)
    assert angular > linear / 3, (angular, linear)
    assert angular < linear, "speed should still lead; a task that turns beats one that moves"
