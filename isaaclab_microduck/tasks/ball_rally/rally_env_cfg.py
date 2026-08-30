"""Microduck BallRally: two ducks, one ball, passed back and forth.

Duck A (`robot`) is the LEARNER. Duck B (`partner`) replays a frozen `ball_kick`
policy, driven by `MicroduckBallRallyEnv` -- so from the manager's point of view
this is an ordinary single-agent env and the 61D / 14-D contract is untouched. A
single net driving both ducks would have doubled the observation and broken
hot-swapping in the runtime forever; that is why it is built this way.

Built on the ball_kick recipe rather than from scratch, so the DR, observation
noise, NaN guard and action_rate curriculum stay in sync for free.

WHAT THE PARTNER ACTUALLY IS. It is a standing-kick policy, ball-blind, trained
to swing when a ball sits at its right foot from a stationary start. It cannot
chase, aim, or trap. Expect it to return SOME balls that arrive near its right
foot and ignore the rest. So a long rally is not reachable yet, and the honest
first target is 1-2 completed passes; the metric below counts them so that claim
is measured rather than asserted. Refreshing the partner from a BallRally
checkpoint (turning this into self-play) is the follow-up that makes real rallies
possible.

REWARD DESIGN. A rally pays per pass, and a per-pass bonus is a textbook jackpot:
arriving at the goal early and often is worth arbitrary violence. Two defences:
`pass_completed` is LATCHED so one crossing scores exactly once and only re-arms
after the ball genuinely comes back, and the dense term is POTENTIAL-BASED
(delta-gap), which pays zero for holding position and integrates to zero over any
closed cycle. Neither can be farmed by parking the ball.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_microduck.robot.microduck_cfg import make_microduck_cfg
from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.ball_kick.ball_kick_env_cfg import (
    BALL_OFFSET,
    BALL_PLACEMENT_NOISE,
    BALL_RADIUS,
    STAND_HEIGHT,
    SUPPORT_FOOT,
    BallKickEventsCfg,
    BallKickSceneCfg,
    MicroduckBallKickFlatEnvCfg,
)
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    TRUNK_BODY,
    ActionsCfg,
    CommandsCfg,
)

##
# Tuned constants.
##

PARTNER_DISTANCE = 0.8
"""m between the two ducks, along the learner's heading.

Sized off the MEASURED ball_kick policy, not a guess: it sends the ball ~1.8 m at
the 1.0 m/s target, so 0.8 m is comfortably inside one kick while still far
enough that a nudge cannot reach.
"""

REACH_RADIUS = 0.20
"""m from the partner's trunk that counts as the ball having arrived."""

REARM_FRACTION = 0.5
"""The ball must come back past this fraction of the gap before another pass can
score. This is the anti-jackpot latch; do not raise it to 0."""

BALL_MAX_DISTANCE = 2.0
"""m from the learner before the ball is out of play and the episode ends."""

EPISODE_LENGTH_S = 15.0
"""Long enough for several exchanges; a 5 s episode cannot contain a rally."""

PARTNER_POLICY_PATH = (
    "logs/rsl_rl/microduck_ball_kick/2026-08-29_11-14-48/exported/policy.pt"
)
"""Exported TorchScript for duck B -- normalizer BAKED IN.

Produced by play.py / scripts/export.py. Never point this at a raw checkpoint: an
unnormalized partner behaves like a different robot and the bug is invisible in
sim, because in-sim play applies the normalizer anyway.
"""


##
# Scene.
##


@configclass
class BallRallySceneCfg(BallKickSceneCfg):
    """ball_kick's scene plus a second duck facing the first."""

    partner = make_microduck_cfg("allcollisions", prim_path="{ENV_REGEX_NS}/Partner")

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Face the learner, PARTNER_DISTANCE away: yaw pi about z is (w=0, z=1).
        self.partner.init_state.pos = (PARTNER_DISTANCE, 0.0, self.robot.init_state.pos[2])
        self.partner.init_state.rot = (0.0, 0.0, 0.0, 1.0)


##
# Observations.
##


@configclass
class BallRallyObservationsCfg:
    """61D actor, unchanged. Ball AND partner state are critic-only."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_imu_misaligned, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity_imu_misaligned, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.25, n_max=0.25))
        actions = ObsTerm(func=mdp.last_action)
        twist_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        head_command = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_command = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic only, and NOT a subclass of PolicyCfg.

        The runner concatenates ["policy", "privileged"], so inheriting the actor
        group here would feed the critic all 61 proprioception dims twice. Adding
        any of this to the ACTOR would change the task: both ducks must stay
        ball-blind.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ball_position = ObsTerm(func=mdp.ball_pos_in_base)
        ball_velocity = ObsTerm(func=mdp.ball_vel_in_base)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


##
# Rewards.
##


@configclass
class BallRallyRewardsCfg:
    """Rally terms plus ball_kick's standing stack.

    Sign convention: penalties return >= 0 and carry NEGATIVE weights, so every
    `Episode_Reward/<penalty>` must log <= 0 on every run.
    """

    # -- the rally --
    pass_completed = RewTerm(
        func=mdp.pass_completed,
        weight=10.0,  # one-shot per crossing; the latch makes this unfarmable
        params={"reach_radius": REACH_RADIUS, "rearm_fraction": REARM_FRACTION},
    )
    ball_progress = RewTerm(
        func=mdp.ball_progress_to_partner,
        weight=3.0,  # potential-based: holding pays 0, a round trip integrates to 0
    )

    # -- stay standing while doing it (ball_kick's stack, same weights) --
    support_foot_grounded = RewTerm(
        func=mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[SUPPORT_FOOT])},
    )
    upright = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY])},
    )
    height_stand = RewTerm(
        func=mdp.height_target_gaussian,
        weight=1.0,
        params={"target_height": STAND_HEIGHT, "std": 0.04},
    )
    pose_stand_legs = RewTerm(
        func=mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=[r"^(?!passive_|.*neck.*|.*head.*).*"])},
    )
    pose_stand_neck = RewTerm(
        func=mdp.pose_target_match,
        weight=1.0,
        params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*neck.*", r".*head.*"])},
    )

    # -- regularizers --
    body_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


##
# Terminations, events, curriculum.
##


@configclass
class BallRallyTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    nan_state = DoneTerm(func=mdp.robot_state_is_nan)
    ball_lost = DoneTerm(func=mdp.ball_out_of_play, params={"max_distance": BALL_MAX_DISTANCE})


@configclass
class BallRallyEventsCfg(BallKickEventsCfg):
    """ball_kick's reset plus the rally latch. Declaration order matters."""

    reset_rally = EventTerm(func=mdp.reset_rally_state, mode="reset")


@configclass
class BallRallyCurriculumCfg:
    # Scoring hook, not a curriculum: fires on reset before the events clear the
    # counters, so the finished episode's tally is still readable.
    rally_length = CurrTerm(func=mdp.rally_length)

    # Ramp ends at -0.6, NOT ball_kick's -1.0. Measured reason: the first 6000-iter
    # rally run peaked at ~2.1 passes/episode around iteration 4200-4500 and then
    # declined ~25% (p90 5.90 -> 4.50) over the final 1200 iterations, while
    # `upright` and fall rate kept IMPROVING and the `max` passes column contracted
    # 33 -> 15. That signature -- fewer long rallies, steadier standing -- is the
    # smoothness tax damping the aggressive swings a rally needs. mjlab's ball_kick
    # cfg names exactly this knob for exactly this symptom: "if the converged kick
    # is too weak, softening the ramp end (-1.0 -> -0.6) is the first knob to try".
    # A rally is a more dynamic task than a single kick, so it wants the softer end.
    action_rate_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
            ],
        },
    )


##
# Environment.
##


@configclass
class MicroduckBallRallyFlatEnvCfg(MicroduckBallKickFlatEnvCfg):
    """Pass a ball back and forth with a frozen partner, on flat ground."""

    scene: BallRallySceneCfg = BallRallySceneCfg(num_envs=4096, env_spacing=3.0)
    observations: BallRallyObservationsCfg = BallRallyObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: BallRallyRewardsCfg = BallRallyRewardsCfg()
    terminations: BallRallyTerminationsCfg = BallRallyTerminationsCfg()
    events: BallRallyEventsCfg = BallRallyEventsCfg()
    curriculum: BallRallyCurriculumCfg = BallRallyCurriculumCfg()

    partner_policy_path: str = PARTNER_POLICY_PATH

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
        # Ball starts at the learner's right foot, exactly as in ball_kick.
        self.events.reset_ball.params["offset"] = BALL_OFFSET
        self.events.reset_ball.params["noise_xy"] = BALL_PLACEMENT_NOISE
        self.events.reset_ball.params["ball_radius"] = BALL_RADIUS
        # A rally needs a REPEATABLE facing: a random yaw would put the partner
        # somewhere different every episode for a policy that cannot see it.
        self.events.reset_base.params["pose_range"] = {}


@configclass
class MicroduckBallRallyFlatEnvCfg_PLAY(MicroduckBallRallyFlatEnvCfg):
    """Small, quiet variant for watching."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_mass = None
        self.events.base_com = None
        self.events.joint_armature = None
