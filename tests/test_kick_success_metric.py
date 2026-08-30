"""The task needs a success metric that actually measures kicking.

`Metrics/success_rate` comes from Isaac Lab's UniformVelocityCommand and scores
twist tracking against the +/-0.01 command this task carries only for obs-shape
parity. It read 0.0000 through a run whose ball speed rose 100x, and it flickered
to 1.00 for one iteration when the robot happened to hold still -- so it is not
merely uninformative here, it is actively misleading. `mdp.kick_success` is the
real criterion; these tests keep it wired and keep its two conditions honest.

CPU-only: no simulator needed.
"""

import inspect

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import (
    KICK_SUCCESS_DISTANCE,
    KICK_SUCCESS_MAX_TILT_DEG,
    MicroduckBallKickFlatEnvCfg,
)

FELL_OVER_LIMIT_DEG = 70.0


def _term():
    return MicroduckBallKickFlatEnvCfg().curriculum.kick_success


def test_metric_is_wired_into_the_curriculum_manager():
    # The curriculum manager is the only per-reset hook that runs BEFORE the reset
    # events re-place the ball, so the metric must live there to see where the
    # episode actually left it.
    assert _term().func is mdp.kick_success


def test_thresholds_are_passed_through_from_the_cfg_constants():
    params = _term().params
    assert params["min_distance"] == KICK_SUCCESS_DISTANCE
    assert params["max_tilt_deg"] == KICK_SUCCESS_MAX_TILT_DEG


def test_success_tilt_is_stricter_than_the_fall_termination():
    # `fell_over` fires at 70 deg; scoring a kick as successful at 69 deg of lean
    # would pass exactly the leaning-but-alive behaviour this metric exists to catch.
    assert KICK_SUCCESS_MAX_TILT_DEG < FELL_OVER_LIMIT_DEG


def test_success_distance_clears_the_ball_spawn_offset():
    # The ball spawns ~9 cm ahead of the robot; a threshold below that could be
    # met by nudging rather than kicking.
    from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import BALL_OFFSET

    assert KICK_SUCCESS_DISTANCE > abs(BALL_OFFSET[0])


@pytest.mark.parametrize("name", ["min_distance", "max_tilt_deg"])
def test_thresholds_are_tunable_parameters(name):
    # Both defaults are guesses; `Metrics/kick_distance` exists so they can be
    # calibrated from a run. They must stay parameters, not constants baked in.
    assert name in inspect.signature(mdp.kick_success).parameters


def test_metric_runs_before_reset_events_in_the_env_loop():
    # Guards the ordering the metric depends on: if Isaac Lab ever moved the
    # curriculum call after the reset events, the ball would already be re-placed
    # and the metric would silently read ~0 for every episode.
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    src = inspect.getsource(ManagerBasedRLEnv._reset_idx)
    assert src.index("curriculum_manager.compute") < src.index('event_manager.apply(mode="reset"')
