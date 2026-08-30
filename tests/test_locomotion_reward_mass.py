"""Locomotion must outbid standing still, and be measured in metres.

A 6000-iteration velocity run and a 6000-iteration running run both converged to
STANDING STILL -- 0.8 mm travelled in 6 s under a 1.2 m/s command -- while
`Episode_Reward/track_linear_velocity` read ~1.2 the whole time. A stationary
upright robot still collects upright + head_pose_tracking + pose +
track_angular_velocity, so at the original weight of 2.0 walking risked ~7.0 of
banked reward to gain at most 2.0.

These tests keep the two lessons: locomotion has to outweigh what standing pays,
and the check has to be a MEASUREMENT, not a reward.

CPU-only.
"""

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.running.running_env_cfg import MicroduckRunningFlatEnvCfg
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import MicroduckVelocityFlatEnvCfg

#: Terms a robot standing still, upright, at HOME can collect in full.
_STANDING_SATISFIABLE = ("upright", "head_pose_tracking", "pose")

CFGS = (MicroduckVelocityFlatEnvCfg, MicroduckRunningFlatEnvCfg)


def test_locomotion_outbids_the_standing_stack():
    for C in CFGS:
        r = C().rewards
        standing = sum(getattr(r, n).weight for n in _STANDING_SATISFIABLE)
        assert r.track_linear_velocity.weight > standing, C.__name__


def test_tracking_weight_exceeds_every_standing_term_individually():
    for C in CFGS:
        r = C().rewards
        for n in _STANDING_SATISFIABLE:
            assert r.track_linear_velocity.weight > getattr(r, n).weight, (C.__name__, n)


def test_measured_speed_metric_is_wired():
    # Reward terms were misread as evidence of motion for two entire runs. The
    # metric publishes m/s, which a stationary policy cannot fake.
    for C in CFGS:
        assert C().curriculum.locomotion_metrics.func is mdp.locomotion_metrics, C.__name__


def test_metric_runs_on_the_reset_hook_before_events_move_the_robot():
    import inspect

    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    src = inspect.getsource(ManagerBasedRLEnv._reset_idx)
    assert src.index("curriculum_manager.compute") < src.index('event_manager.apply(mode="reset"')
