"""Running: the velocity recipe pushed from a walk into a run.

The walking policy topped out at 0.4 m/s because that was its entire command
range, and `air_time` stayed low because walking that slowly never demanded a
flight phase. These tests lock the three changes that alter the DEMAND, and the
guard rails around them.

CPU-only.
"""

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.running.running_env_cfg import (
    AIR_TIME_THRESHOLD,
    AIR_TIME_WEIGHT,
    FOOT_CLEARANCE_HEIGHT,
    RUN_SPEED,
    WALK_SPEED,
    MicroduckRunningFlatEnvCfg,
    MicroduckRunningFlatEnvCfg_PLAY,
)
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import MicroduckVelocityFlatEnvCfg


def _cfg():
    return MicroduckRunningFlatEnvCfg()


def _obs_terms(group):
    from isaaclab.managers import ObservationTermCfg

    return [k for k, v in group.__dict__.items() if isinstance(v, ObservationTermCfg)]


def test_obs_contract_unchanged_from_velocity():
    # Running must stay hot-swappable with the rest of the locomotion family.
    assert _obs_terms(_cfg().observations.policy) == _obs_terms(
        MicroduckVelocityFlatEnvCfg().observations.policy
    )


def test_speed_starts_at_the_proven_walking_range():
    # Widening the range from step 0 once "outpaced the robot's capability"; the
    # ramp must begin where the velocity recipe is known to work.
    stages = _cfg().curriculum.twist_speed.params["range_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["lin_vel_x"] == (-WALK_SPEED, WALK_SPEED)


def test_speed_ramp_is_monotonic_and_reaches_the_target():
    stages = _cfg().curriculum.twist_speed.params["range_stages"]
    tops = [s["lin_vel_x"][1] for s in stages]
    assert tops == sorted(tops)
    assert tops[-1] == RUN_SPEED > WALK_SPEED


def test_first_widening_waits_for_a_gait():
    # Before ~1000 iterations the velocity run was still falling; handing the
    # policy a command it cannot track would tax the gait it is still forming.
    stages = _cfg().curriculum.twist_speed.params["range_stages"]
    assert stages[1]["step"] >= 1000 * 24


def test_flight_phase_is_actually_demanded():
    r = _cfg().rewards
    v = MicroduckVelocityFlatEnvCfg().rewards
    assert r.air_time.weight == AIR_TIME_WEIGHT > v.air_time.weight
    assert r.air_time.params["threshold"] == AIR_TIME_THRESHOLD > v.air_time.params["threshold"]


def test_foot_clearance_target_is_raised():
    r = _cfg().rewards
    v = MicroduckVelocityFlatEnvCfg().rewards
    assert (
        r.foot_clearance.params["target_height"]
        == FOOT_CLEARANCE_HEIGHT
        > v.foot_clearance.params["target_height"]
    )
    # It is a PENALTY on deviation, so the sign must stay negative for the raised
    # target to lift the foot rather than merely permit it.
    assert r.foot_clearance.weight < 0


def test_only_the_intended_terms_differ_from_velocity():
    # A bad outcome must be attributable: nothing beyond the gait terms changes.
    r, v = _cfg().rewards, MicroduckVelocityFlatEnvCfg().rewards
    changed = {
        n
        for n in vars(v)
        if not n.startswith("_") and hasattr(r, n) and getattr(r, n).weight != getattr(v, n).weight
    }
    assert changed <= {"air_time"}


def test_play_variant_commands_an_actual_run():
    # The velocity PLAY variant samples commands randomly, so ~1 in 10 ducks stands
    # still and the rest meander -- which reads as "the robot will not run".
    p = MicroduckRunningFlatEnvCfg_PLAY()
    assert p.commands.twist.ranges.lin_vel_x == (RUN_SPEED, RUN_SPEED)
    assert p.commands.twist.rel_standing_envs == 0.0
    assert p.commands.twist.rel_turn_in_place_envs == 0.0


def test_play_variant_disables_the_speed_curriculum():
    # Otherwise the curriculum would reset the pinned range back to stage 0.
    assert MicroduckRunningFlatEnvCfg_PLAY().curriculum.twist_speed is None


def test_curriculum_mutates_through_the_manager():
    assert _cfg().curriculum.twist_speed.func is mdp.twist_range_curriculum
