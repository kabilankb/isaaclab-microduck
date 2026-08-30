"""Microduck RunParallel: two ducks running side by side.

Duck A (`robot`) learns; duck B (`partner`) replays a frozen LOCOMOTION policy and
paces at a fixed forward speed. Built on the velocity recipe rather than from
scratch, so the whole gait reward stack, DR, observation noise and curricula stay
in sync for free -- this task is "the walking task, plus a pacer to stay level
with", not a new locomotion problem.

The learner is PARTNER-BLIND: partner state is critic-only, so the 61D actor
contract is unchanged and policies stay hot-swappable. The real robot has no
teammate sensing, so a policy that homed on the partner would not transfer.

REWARD DESIGN. "Stay abreast" is a position relationship, and paying for position
directly is farmable -- the cheapest way to match a pacer is to stop whenever you
drift ahead, which yields a lurching non-gait. Three defences:
  * only the LONGITUDINAL error is priced (`abreast`), never lateral proximity,
    which would invite the pair to converge and collide;
  * its weight stays BELOW `track_linear_velocity`, so matching the pacer can never
    outbid actually running at the commanded speed;
  * it is introduced by CURRICULUM at zero, after a gait exists -- a formation tax
    during gait discovery makes "stand still next to the pacer" win.

PREREQUISITE: `partner_policy_path` needs a trained velocity policy exported to
TorchScript. There was none when this was written, hence the velocity run.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_microduck.robot.microduck_cfg import make_microduck_cfg
from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    CurriculumCfg,
    MicroduckSceneCfg,
    MicroduckVelocityFlatEnvCfg,
    RewardsCfg,
)

##
# Tuned constants.
##

LANE_SEPARATION = 0.35
"""m between the two lanes, in y.

Wide enough that normal gait sway never trips the collision penalty (the robot is
~25 cm tall with feet at +/-4.2 cm), narrow enough that the pair reads as running
together rather than in separate rooms.
"""

PACER_SPEED = 0.4
"""m/s the pacer is commanded to run at.

Held FIXED per episode: a pacer that chased the learner would make the formation
reward satisfiable by standing still, since the target would come to you.
"""

ABREAST_STD = 0.25
"""m of longitudinal offset still worth pricing (~one robot length).

Set to the error we care about, NOT the maximum. A std tight enough to punish
normal gait phase differences would tax running itself -- the trap AGENTS.md
records for head tracking, where an over-tight std made the policy stand still.
"""

MIN_SEPARATION = 0.25
"""m below which the pair is treated as converging. Anti-collision only."""

PARTNER_POLICY_PATH = ""
"""Exported TorchScript for the pacer -- normalizer BAKED IN.

Fill in once the velocity run finishes:
  logs/rsl_rl/microduck_velocity/<timestamp>/exported/policy.pt
Never a raw checkpoint: an unnormalized partner behaves like a different robot and
the bug is invisible in sim.
"""


##
# Scene / rewards / curriculum.
##


@configclass
class RunParallelSceneCfg(MicroduckSceneCfg):
    """Walking scene plus a pacer duck one lane over."""

    partner = make_microduck_cfg("walk", prim_path="{ENV_REGEX_NS}/Partner")

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Same heading as the learner, one lane to the side.
        self.partner.init_state.pos = (0.0, LANE_SEPARATION, self.robot.init_state.pos[2])


@configclass
class RunParallelRewardsCfg(RewardsCfg):
    """The full velocity gait stack, plus formation terms.

    Weights inherited unchanged so the gait keeps the balance that trains it; the
    two additions are deliberately small relative to `track_linear_velocity`.
    """

    abreast = RewTerm(
        func=mdp.abreast_error,
        weight=0.0,  # ramped in by curriculum, AFTER a gait exists
        params={"std": ABREAST_STD},
    )
    pair_separation = RewTerm(
        func=mdp.pair_separation_penalty,
        weight=-2.0,  # anti-collision; hinged, so zero at normal lane spacing
        params={"min_separation": MIN_SEPARATION},
    )


@configclass
class RunParallelCurriculumCfg(CurriculumCfg):
    """Velocity's curricula, plus the formation ramp and the pair metrics."""

    # Scoring hook, not a curriculum -- the per-reset hook that fires before the
    # reset events move the robots.
    pair_metrics = CurrTerm(func=mdp.pair_metrics)

    # OFF until a gait exists: a formation tax during gait discovery makes
    # "stand still next to the pacer" the winning move.
    abreast_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "abreast",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 0.5},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 1.0},
            ],
        },
    )


##
# Environment.
##


@configclass
class MicroduckRunParallelFlatEnvCfg(MicroduckVelocityFlatEnvCfg):
    """Run side by side with a frozen pacer, on flat ground."""

    scene: RunParallelSceneCfg = RunParallelSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: RunParallelRewardsCfg = RunParallelRewardsCfg()
    curriculum: RunParallelCurriculumCfg = RunParallelCurriculumCfg()

    partner_policy_path: str = PARTNER_POLICY_PATH
    pacer_speed: float = PACER_SPEED

    def __post_init__(self):
        super().__post_init__()
        # The learner is commanded to run at the pacer's speed, so "keep up" and
        # "track your command" are the same objective rather than competing ones.
        self.commands.twist.ranges.lin_vel_x = (PACER_SPEED, PACER_SPEED)
        self.commands.twist.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.twist.ranges.ang_vel_z = (-0.1, 0.1)
        # A repeatable heading: the learner cannot see the pacer, so a random yaw
        # would put it somewhere different every episode for a blind policy.
        self.events.reset_base.params["pose_range"] = {}


@configclass
class MicroduckRunParallelFlatEnvCfg_PLAY(MicroduckRunParallelFlatEnvCfg):
    """Small, quiet variant for watching."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_mass = None
        self.events.base_com = None
        self.events.joint_armature = None
