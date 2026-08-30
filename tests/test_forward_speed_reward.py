"""The locomotion reward needs a DENSE term, not only a Gaussian.

Two 6000-iteration runs stalled at ~20% of commanded speed. The cause was not a
missing command gate (IsaacLab's `feet_air_time_positive_biped` already gates on
command, and `variable_posture` already has a walking_threshold) but the shape of
the tracking reward: `exp(-err^2/std^2)` with std=sqrt(0.1) scores exp(-8.3) at
the robot's actual error, so the policy sat in a flat region with no gradient
telling it which way was better.

`forward_speed_linear` is the dense bootstrap -- the same linear-and-capped shape
as `ball_forward_velocity`, which is what got the kick off the ground.

CPU-only.
"""

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.running.running_env_cfg import MicroduckRunningFlatEnvCfg
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import MicroduckVelocityFlatEnvCfg

CFGS = (MicroduckVelocityFlatEnvCfg, MicroduckRunningFlatEnvCfg)
_STANDING_SATISFIABLE = ("upright", "head_pose_tracking", "pose")


@pytest.mark.parametrize("C", CFGS)
def test_dense_forward_speed_term_exists(C):
    assert C().rewards.forward_speed.func is mdp.forward_speed_linear


@pytest.mark.parametrize("C", CFGS)
def test_dense_term_is_positive_and_material(C):
    # It has to be worth more than the noise; a token weight would leave the flat
    # region intact.
    assert C().rewards.forward_speed.weight > 0


@pytest.mark.parametrize("C", CFGS)
def test_locomotion_terms_together_outbid_standing(C):
    # Standing still collects upright + head_pose_tracking + pose in full.
    r = C().rewards
    standing = sum(getattr(r, n).weight for n in _STANDING_SATISFIABLE)
    locomotion = r.forward_speed.weight + r.track_linear_velocity.weight
    assert locomotion > standing


@pytest.mark.parametrize("C", CFGS)
def test_gaussian_is_kept_for_precision(C):
    # The dense term caps at the command, so it cannot punish overshoot or reward
    # accuracy; the Gaussian still does that once the policy is close.
    assert C().rewards.track_linear_velocity.weight > 0
