"""Microduck Running: the velocity recipe, pushed from a walk into a run.

The walking policy trained from `velocity` tops out at 0.4 m/s because that is the
whole of its command range -- it is not a shuffling walker that failed to run, it
is a walker that was never asked to run. `air_time` sat low for the same reason:
walking at <=0.4 m/s does not need much foot lift, so nothing demanded a flight
phase. This task changes the demand, not the algorithm.

THREE CHANGES, and only three, so a bad outcome is attributable:

1. SPEED, ramped by curriculum from the known-good 0.4 to RUN_SPEED. Ramped rather
   than set wide, because the velocity cfg records that widening the ranges once
   "outpaced the robot's capability and tracked a post-iteration-1000 decline in
   both reward and episode length". Every stage stays inside what the policy can
   already do.
2. FLIGHT PHASE demanded: `air_time` weight up and its threshold raised, so long
   single-stance phases pay rather than quick shuffling steps.
3. FOOT CLEARANCE raised, so the swing foot actually leaves the ground.

Everything else -- DR, observation noise, the 61D contract, the action_rate ramp,
the head/body pose stack -- is inherited from velocity untouched. This is a
locomotion RECIPE change, not a new locomotion problem.

HONEST EXPECTATION: a 25 cm biped with 14 XL330 servos may simply not have the
power density to reach a true flight-phase run. RUN_SPEED is a target to train
against, not a promise. `Metrics/twist/error_vel_xy` against the ramped range is
what says whether the robot is actually keeping up, and `Episode_Reward/air_time`
is what says whether it is stepping rather than gliding. Watch both, not reward.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    CurriculumCfg,
    MicroduckVelocityFlatEnvCfg,
    RewardsCfg,
)

##
# Tuned constants.
##

WALK_SPEED = 0.4
"""m/s -- the velocity recipe's proven ceiling. Curriculum stage 0."""

RUN_SPEED = 1.2
"""m/s target top speed, 3x the walk.

A guess to train against, not a measured capability: for a 25 cm robot this is
~4.8 body-lengths/s. If the speed curriculum's later stages track a DECLINE in
reward or episode length, that is the same "outpaced the robot" failure the
velocity cfg warns about, and the fix is to stop the ramp lower.
"""

AIR_TIME_WEIGHT = 6.0
"""Doubled from velocity's 3.0. A flight phase has to outbid the smoothness and
pose terms that a low glide already satisfies."""

AIR_TIME_THRESHOLD = 0.25
"""s of single stance that counts, doubled from velocity's 0.125.

The threshold is what distinguishes a stride from a shuffle: quick alternating
contacts satisfy a low threshold without ever leaving the ground.
"""

FOOT_CLEARANCE_HEIGHT = 0.05
"""m of swing-foot height, up from velocity's 0.02.

Note the term's weight is NEGATIVE -- it penalises deviation from this target, so
raising the target raises the foot rather than merely permitting it.
"""


@configclass
class RunningRewardsCfg(RewardsCfg):
    """Velocity's stack with the gait terms sharpened. Weights elsewhere untouched."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.air_time.weight = AIR_TIME_WEIGHT
        self.air_time.params["threshold"] = AIR_TIME_THRESHOLD
        self.foot_clearance.params["target_height"] = FOOT_CLEARANCE_HEIGHT


@configclass
class RunningCurriculumCfg(CurriculumCfg):
    """Velocity's curricula plus the speed ramp.

    Phase-aligned with the gait: the first widening waits until iteration 1000, by
    which point the velocity run had falls under 2% and a stable gait. Widening
    before that would hand the policy a command it cannot yet track and tax the
    gait it is still forming.
    """

    twist_speed = CurrTerm(
        func=mdp.twist_range_curriculum,
        params={
            "command_name": "twist",
            "range_stages": [
                {"step": 0, "lin_vel_x": (-WALK_SPEED, WALK_SPEED)},
                {"step": 1000 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.5, 0.6)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.5, 0.8)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.5, 1.0)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.5, RUN_SPEED)},
            ],
        },
    )


@configclass
class MicroduckRunningFlatEnvCfg(MicroduckVelocityFlatEnvCfg):
    """Run on flat ground: velocity's recipe with speed and flight phase demanded."""

    rewards: RunningRewardsCfg = RunningRewardsCfg()
    curriculum: RunningCurriculumCfg = RunningCurriculumCfg()


@configclass
class MicroduckRunningFlatEnvCfg_PLAY(MicroduckRunningFlatEnvCfg):
    """Small variant for watching -- and it COMMANDS A RUN.

    The velocity PLAY variant samples commands randomly, so ~1 in 10 ducks is told
    to stand still and the rest meander at assorted speeds and headings. That looks
    exactly like "the robot will not run" even when the policy is fine. Here the
    command is pinned to RUN_SPEED forward with no turn and no standing envs, so
    what you watch is the thing the task is named after.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_mass = None
        self.events.base_com = None
        self.events.joint_armature = None
        # Command a straight run, every env.
        self.commands.twist.ranges.lin_vel_x = (RUN_SPEED, RUN_SPEED)
        self.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.twist.rel_standing_envs = 0.0
        self.commands.twist.rel_turn_in_place_envs = 0.0
        # The PLAY curriculum would otherwise reset the range back to stage 0.
        self.curriculum.twist_speed = None
