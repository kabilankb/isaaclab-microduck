"""BallRally: two ducks, one ball, one learner.

The design constraint that drove every choice here is AGENTS.md's hard invariant:
the actor observation stays 61D across the whole policy family so policies remain
hot-swappable in the runtime. That is why duck B is a FROZEN policy driven by the
env rather than a second head on the network -- a shared net would have doubled
the observation and made this task's policy undeployable.

The other thing these tests protect is the anti-jackpot design. A rally pays per
pass, and an unlatched per-pass bonus is farmable by shoving the ball into the
partner and leaving it there.

CPU-only: no simulator needed.
"""

import math

import pytest

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.ball_rally.rally_env_cfg import (
    BALL_MAX_DISTANCE,
    PARTNER_DISTANCE,
    REARM_FRACTION,
    REACH_RADIUS,
    MicroduckBallRallyFlatEnvCfg,
)

ACTOR_OBS_DIM = 61


def _cfg():
    return MicroduckBallRallyFlatEnvCfg()


def _obs_terms(group):
    """Term names in declaration order, excluding ObsGroup's own config fields."""
    from isaaclab.managers import ObservationTermCfg

    return [k for k, v in group.__dict__.items() if isinstance(v, ObservationTermCfg)]


def test_actor_stays_ball_blind_and_partner_blind():
    # The whole point of the frozen-partner design: neither the ball nor the
    # partner may enter the ACTOR group, or the 61D contract breaks.
    terms = _obs_terms(_cfg().observations.policy)
    assert terms and not any("ball" in t or "partner" in t for t in terms)


def test_actor_obs_terms_match_the_shared_contract():
    # Same terms, same order, as every other Microduck policy -- this ordering IS
    # the contract the runtime's single obs buffer depends on.
    expected = [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "twist_command", "head_command", "body_command",
    ]
    assert _obs_terms(_cfg().observations.policy) == expected


def test_actor_obs_matches_ball_kick_exactly():
    # Hot-swap parity: if this ever diverges from ball_kick, the two policies can
    # no longer share one observation buffer in the runtime.
    from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import MicroduckBallKickFlatEnvCfg

    assert _obs_terms(_cfg().observations.policy) == _obs_terms(
        MicroduckBallKickFlatEnvCfg().observations.policy
    )


def test_privileged_group_does_not_inherit_the_policy_group():
    # The runner concatenates ["policy", "privileged"]; inheriting would feed the
    # critic all 61 proprioception dims twice.
    from isaaclab_microduck.tasks.ball_rally.rally_env_cfg import BallRallyObservationsCfg

    assert not issubclass(BallRallyObservationsCfg.PrivilegedCfg, BallRallyObservationsCfg.PolicyCfg)


def test_privileged_group_is_named_for_the_runner():
    # obs_groups = {"critic": ["policy", "privileged"]} -- a group named anything
    # else fails at runner construction, not at cfg build.
    assert hasattr(_cfg().observations, "privileged")


def test_pass_reward_is_latched_not_a_per_step_bonus():
    # `pass_completed` must re-arm only after the ball comes back; a rearm
    # fraction of 0 would let one crossing score every step it stays arrived.
    params = _cfg().rewards.pass_completed.params
    assert params["rearm_fraction"] == REARM_FRACTION > 0.0
    assert params["reach_radius"] == REACH_RADIUS > 0.0


def test_dense_rally_term_is_potential_based():
    # `ball_progress_to_partner` pays the CHANGE in gap, so holding pays zero and
    # a closed round trip integrates to zero. A plain distance bonus would be
    # farmable by parking the ball near the partner.
    assert _cfg().rewards.ball_progress.func is mdp.ball_progress_to_partner


def test_every_penalty_weight_is_negative():
    r = _cfg().rewards
    for name in ("body_ang_vel", "dof_pos_limits", "action_rate_l2"):
        assert getattr(r, name).weight < 0, name


def test_rally_rewards_are_positive():
    r = _cfg().rewards
    assert r.pass_completed.weight > 0 and r.ball_progress.weight > 0


def test_partner_is_placed_within_one_kick_but_out_of_nudge_range():
    # The measured ball_kick policy sends the ball ~1.8 m; the ball spawns ~9 cm
    # ahead of the learner. The partner must sit between those.
    from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import BALL_OFFSET

    assert abs(BALL_OFFSET[0]) < PARTNER_DISTANCE < 1.8


def test_ball_lost_termination_ends_a_dead_rally():
    # Without this the episode keeps paying the standing stack after the ball is
    # gone, which pays the policy to end the rally.
    t = _cfg().terminations
    assert t.ball_lost.func is mdp.ball_out_of_play
    assert t.ball_lost.params["max_distance"] == BALL_MAX_DISTANCE > PARTNER_DISTANCE


def test_episode_is_long_enough_to_contain_a_rally():
    # ball_kick's 5 s cannot hold an exchange; a rally needs several.
    assert _cfg().episode_length_s >= 15.0


def test_reset_clears_the_rally_latch():
    # A stale latch would carry one episode's "already passed" state into the next.
    assert _cfg().events.reset_rally.func is mdp.reset_rally_state


def test_spawn_facing_is_repeatable():
    # The learner cannot see the partner, so a random yaw would put the partner
    # somewhere different every episode for a blind policy.
    assert _cfg().events.reset_base.params["pose_range"] == {}


def test_partner_policy_path_points_at_an_exported_torchscript():
    # Never a raw checkpoint: the exported .pt has the obs normalizer baked in,
    # and an unnormalized partner behaves like a different robot -- invisible in
    # sim, because in-sim play applies the normalizer anyway.
    assert _cfg().partner_policy_path.endswith("policy.pt")


def test_scene_has_two_ducks_facing_each_other():
    scene = _cfg().scene
    assert scene.partner.prim_path.endswith("/Partner")
    assert scene.robot.prim_path.endswith("/Robot")
    assert scene.partner.init_state.pos[0] == pytest.approx(PARTNER_DISTANCE)
    # yaw = pi about z -> (w, x, y, z) = (0, 0, 0, 1)
    assert scene.partner.init_state.rot == (0.0, 0.0, 0.0, 1.0)


def test_env_spacing_exceeds_the_pair_footprint():
    # Two ducks plus a ball that can travel BALL_MAX_DISTANCE need room, or
    # neighbouring arenas overlap and balls cross between them.
    assert _cfg().scene.env_spacing > PARTNER_DISTANCE
