"""RunParallel: two ducks running side by side, one learner and a frozen pacer.

The constraints these lock in are the ones that make "stay abreast" trainable
rather than farmable. Paying for position directly invites the policy to stop
whenever it drifts ahead of the pacer, producing a lurching non-gait; and a
formation tax applied during gait discovery makes "stand still next to the pacer"
the winning move outright.

CPU-only: no simulator needed.
"""

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.run_parallel.run_env_cfg import (
    ABREAST_STD,
    LANE_SEPARATION,
    MIN_SEPARATION,
    PACER_SPEED,
    MicroduckRunParallelFlatEnvCfg,
)


def _cfg():
    return MicroduckRunParallelFlatEnvCfg()


def _obs_terms(group):
    from isaaclab.managers import ObservationTermCfg

    return [k for k, v in group.__dict__.items() if isinstance(v, ObservationTermCfg)]


def test_actor_is_partner_blind():
    # The real robot has no teammate sensing; a policy that homed on the partner
    # would not transfer, and it would break the 61D contract besides.
    terms = _obs_terms(_cfg().observations.policy)
    assert terms and not any("partner" in t or "pair" in t for t in terms)


def test_actor_obs_matches_the_velocity_task_exactly():
    # Hot-swap parity: RunParallel is the walking task plus a pacer, so its actor
    # observation must stay identical to the locomotion policy family's.
    from isaaclab_microduck.tasks.velocity.velocity_env_cfg import MicroduckVelocityFlatEnvCfg

    assert _obs_terms(_cfg().observations.policy) == _obs_terms(
        MicroduckVelocityFlatEnvCfg().observations.policy
    )


def test_formation_reward_starts_at_zero():
    # A formation tax active during gait discovery makes standing still win.
    assert _cfg().rewards.abreast.weight == 0.0


def test_formation_reward_is_ramped_in_after_a_gait_exists():
    stages = _cfg().curriculum.abreast_weight.params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert stages[0]["step"] == 0
    assert stages[-1]["weight"] > 0.0
    # introduced late, not early
    assert stages[1]["step"] >= 1000 * 24


def test_formation_never_outbids_actually_running():
    # If matching the pacer paid more than tracking the commanded velocity, the
    # optimum would be to shadow the pacer rather than run.
    r = _cfg().rewards
    final = _cfg().curriculum.abreast_weight.params["weight_stages"][-1]["weight"]
    assert final < r.track_linear_velocity.weight


def test_only_longitudinal_offset_is_rewarded():
    # Rewarding lateral proximity would invite the pair to converge and collide.
    assert _cfg().rewards.abreast.func is mdp.abreast_error


def test_collision_penalty_is_negative_and_hinged_below_lane_spacing():
    # Hinged: exactly zero at normal lane spacing, biting only on convergence.
    r = _cfg().rewards
    assert r.pair_separation.weight < 0
    assert r.pair_separation.params["min_separation"] == MIN_SEPARATION < LANE_SEPARATION


def test_learner_command_matches_the_pacer_speed():
    # "Keep up" and "track your command" must be the same objective, not competing.
    cfg = _cfg()
    assert cfg.commands.twist.ranges.lin_vel_x == (PACER_SPEED, PACER_SPEED)
    assert cfg.pacer_speed == PACER_SPEED


def test_abreast_std_is_not_tighter_than_a_robot_length():
    # A std tight enough to punish normal gait phase differences taxes running
    # itself -- the trap AGENTS.md records for head tracking.
    assert ABREAST_STD >= 0.2


def test_scene_has_two_ducks_in_separate_lanes():
    scene = _cfg().scene
    assert scene.partner.prim_path.endswith("/Partner")
    assert scene.partner.init_state.pos[1] == pytest.approx(LANE_SEPARATION)
    assert scene.partner.init_state.pos[0] == pytest.approx(0.0)  # abreast at spawn


def test_env_spacing_clears_both_lanes():
    assert _cfg().scene.env_spacing > LANE_SEPARATION


def test_spawn_heading_is_repeatable():
    assert _cfg().events.reset_base.params["pose_range"] == {}


def test_velocity_gait_stack_is_inherited_intact():
    # Built on the velocity recipe, not from scratch: the gait terms must survive.
    r = _cfg().rewards
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance"):
        assert hasattr(r, name), name
