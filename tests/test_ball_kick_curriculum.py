"""BallKick must carry the mjlab recipe's action_rate ramp.

The mjlab cfg holds `action_rate_l2` at its stage-0 weight until the swing has
formed, then ramps it to full strength by iteration 1500. The port originally
declared no curriculum at all, which pinned the weight at -0.1 forever: the
strike never got the smoothness discipline and ball speed kept hunting instead of
settling on the target. These tests lock the ramp — and the wiring that makes it
run — against that regressing back.

CPU-only: reading cfg objects needs no simulator.
"""

import pytest

from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import (
    MicroduckBallKickFlatEnvCfg,
)
from isaaclab_microduck.tasks.mdp.curriculums import _stage_value
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import NUM_STEPS_PER_ENV

#: (iteration, expected weight) — the mjlab stages verbatim.
_MJLAB_STAGES = [(0, -0.1), (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)]


def _stages() -> list[dict]:
    cfg = MicroduckBallKickFlatEnvCfg()
    return cfg.curriculum.action_rate_weight.params["weight_stages"]


def test_curriculum_is_wired_into_the_env_cfg():
    # A curriculum class that is never assigned to the cfg is a silent no-op: the
    # manager only runs terms it is handed.
    cfg = MicroduckBallKickFlatEnvCfg()
    assert cfg.curriculum is not None
    assert cfg.curriculum.action_rate_weight.params["reward_name"] == "action_rate_l2"


def test_stages_match_the_mjlab_recipe():
    assert [(s["step"] // NUM_STEPS_PER_ENV, s["weight"]) for s in _stages()] == _MJLAB_STAGES


@pytest.mark.parametrize("iteration,expected", _MJLAB_STAGES)
def test_step_function_resolves_each_stage(iteration, expected):
    # Stages are a STEP function over env steps, not an interpolation.
    assert _stage_value(_stages(), iteration * NUM_STEPS_PER_ENV, "weight") == expected


def test_ramp_is_monotonically_harsher_and_never_positive():
    # action_rate_l2 is a cost >= 0, so its weight must stay negative or the
    # penalty double-negates into a reward for thrashing.
    weights = [s["weight"] for s in _stages()]
    assert all(w <= 0 for w in weights)
    assert weights == sorted(weights, reverse=True)


def test_stage_zero_matches_the_declared_reward_weight():
    # The curriculum only takes effect on its first evaluation; until then the
    # declared weight is what runs, so a mismatch is a silent step at iteration 0.
    cfg = MicroduckBallKickFlatEnvCfg()
    assert cfg.rewards.action_rate_l2.weight == _stages()[0]["weight"]
