"""Microduck BallKick task: kick a 70 mm / 15 g ball forward with the right foot.

Port of `mjlab_microduck.tasks.microduck_ball_kick_env_cfg`. An episodic trick,
5 s episodes, flat terrain only (a ball on rough terrain is a different task).

THE DEFINING CONSTRAINT: **the actor is ball-blind.** Ball position and velocity
are PRIVILEGED, critic-only observations. The actor keeps the same 61D contract
as every other Microduck policy:

    [ base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
      actions(14), twist(3), head_pose(4), body_pose(6) ]

Since this task drives no head or body pose command, those two slots are
ZERO-PADDED rather than removed — deleting them would shift every later
observation and break policy hot-swapping in the runtime.

Because the policy cannot see the ball, the reset placement noise is doing the
real work: it forces a swing that survives the real world's placement error
instead of a trajectory tuned to one exact ball position.

KNOWN GAPS vs the mjlab recipe (sim milestone, not a deployable policy):
the BAM actuator (P3), observation delays, encoder bias, and the
`angular_momentum` regularizer (needs a body-momentum sensor) are not ported.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_microduck.robot.microduck_cfg import SERVO_JOINTS_EXPR, make_microduck_cfg
from isaaclab_microduck.tasks import mdp
from isaaclab_microduck.tasks.velocity.velocity_env_cfg import (
    FOOT_BODIES,
    NUM_STEPS_PER_ENV,
    TRUNK_BODY,
    ActionsCfg,
    CommandsCfg,
    EventsCfg,
    MicroduckSceneCfg,
)

##
# Tuned constants.
##

BALL_RADIUS = 0.035
"""70 mm diameter ball."""

BALL_MASS = 0.015
"""15 g."""

BALL_OFFSET = (0.09, -0.042)
"""Ball centre in the robot's yaw frame — about 1 cm in front of the right toe."""

BALL_PLACEMENT_NOISE = 0.015
"""+/- 15 mm per axis. The policy is ball-blind, so this IS the task."""

KICK_SUCCESS_DISTANCE = 0.30
"""m of forward ball travel required to score an episode as a kick.

A STARTING GUESS, not a measurement: `Metrics/kick_distance` logs the raw
distribution so this can be calibrated against a real run.
"""

KICK_SUCCESS_MAX_TILT_DEG = 30.0
"""Trunk tilt allowed at episode end for a kick to count.

Much tighter than the 70 deg `fell_over` termination on purpose -- that limit
leaves a wide band of "leaning but not fallen" which a SCORING metric must not
wave through, even though a TERMINATION should.
"""

TARGET_BALL_SPEED = 1.0
"""m/s. Paired with the overshoot penalty so this speed is the actual optimum,
not a floor to exceed."""

STAND_HEIGHT = 0.115
"""Trunk z target while kicking. MEASURED in this simulator (P2 parity puts a
standing walk-model trunk at 116.4 mm); never carried across stacks."""

SUPPORT_FOOT = "ankle_left"
KICKING_FOOT = "ankle_right"


##
# Scene: the walking scene plus a ball.
##


@configclass
class BallKickSceneCfg(MicroduckSceneCfg):
    """Full-collision Microduck plus the ball prop.

    Uses the `allcollisions` model, not `walk`: the kicking leg can reach the
    trunk, and the task needs those contacts to exist.
    """

    robot = make_microduck_cfg("allcollisions")

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.4, dynamic_friction=0.4, restitution=0.4
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.5, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BALL_OFFSET[0], BALL_OFFSET[1], BALL_RADIUS)),
    )


##
# Observations.
##


@configclass
class BallKickObservationsCfg:
    """Actor: the 61D contract, ball-blind. Critic: adds the ball state."""

    @configclass
    class PolicyCfg(ObsGroup):
        """61D. Head and body command slots are ZERO-PADDED — see module docstring."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_imu_misaligned,
            params={"max_angle_deg": 6.0},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_imu_misaligned,
            params={"max_angle_deg": 6.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SERVO_JOINTS_EXPR])},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SERVO_JOINTS_EXPR])},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)

        twist_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        head_command = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_command = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic only. Adding any of this to the actor would change the task."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ball_position = ObsTerm(func=mdp.ball_pos_in_base)
        ball_velocity = ObsTerm(func=mdp.ball_vel_in_base)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


##
# Events.
##


@configclass
class BallKickEventsCfg(EventsCfg):
    """Walking DR plus the ball placement.

    `reset_ball` is declared LAST on purpose: events run in declaration order and
    it reads the robot's final root pose to place the ball in its yaw frame.
    """

    reset_ball = EventTerm(
        func=mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={
            "offset": BALL_OFFSET,
            "noise_xy": BALL_PLACEMENT_NOISE,
            "ball_radius": BALL_RADIUS,
        },
    )

    def __post_init__(self):
        # The robot must start from a repeatable standing pose: a kick from a
        # randomized walking spawn is a different (harder, unspecified) task.
        self.reset_base.params["pose_range"] = {"yaw": (-math.pi, math.pi)}
        self.reset_base.params["velocity_range"] = {}
        self.reset_robot_joints.params["position_range"] = (-0.03, 0.03)
        self.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


##
# Rewards.
##


@configclass
class BallKickRewardsCfg:
    """Weights from the mjlab recipe.

    Penalties return >= 0 and take NEGATIVE weights; every
    `Episode_Reward/<penalty>` must log <= 0.
    """

    # -- the kick --
    # Weights follow the mjlab recipe's DOCUMENTED scaling rule, not its literal
    # numbers: "at-target payoff ~= +3/step (weight ~= 3/target) -- if you change
    # the target, rescale the weights with it". That file sets weight 12.0 with a
    # comment reading "Weight 12.0 = 3.0/target", which only holds at the ORIGINAL
    # target of 0.25 m/s; the target was later raised to 1.0 and the weights were
    # never rescaled, so the kick pays 12/step instead of 3/step.
    #
    # That is what collapsed `upright`: at 12/step the kick alone outweighs the
    # ENTIRE standing stack (support 2 + upright 2 + height 1 + legs 2 + neck 1 =
    # 8/step), so leaning over to strike harder is simply the better trade and the
    # policy takes it. PPO sees relative advantage, so compare reward MASS, not
    # weights. At 3/step the kick sits inside a ~11/step total, which is the
    # "task reward mass ~10" the mjlab docstring claims.
    ball_forward_velocity = RewTerm(
        func=mdp.ball_forward_velocity,
        weight=3.0 / TARGET_BALL_SPEED,
        params={"max_speed": TARGET_BALL_SPEED},
    )
    # Keeps the recipe's 3:1 asymmetry: net kick reward crosses zero at 4x the
    # target, so erring hard stays much cheaper than not kicking at all, while
    # |weight| stays BELOW the capped reward's.
    ball_speed_overshoot = RewTerm(
        func=mdp.ball_speed_overshoot_penalty,
        weight=-1.0 / TARGET_BALL_SPEED,
        params={"target_speed": TARGET_BALL_SPEED},
    )

    # -- stay standing while doing it --
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
        params={
            "std": 0.5,  # generous: the kicking leg MUST leave the pose
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[r"^(?!passive_|.*neck.*|.*head.*).*"]
            ),
        },
    )
    pose_stand_neck = RewTerm(
        func=mdp.pose_target_match,
        weight=1.0,
        params={
            "std": 0.3,  # tighter: the head has no job here
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*neck.*", r".*head.*"]),
        },
    )

    # -- regularizers --
    body_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class BallKickTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    nan_state = DoneTerm(func=mdp.robot_state_is_nan)


@configclass
class BallKickCurriculumCfg:
    """The mjlab recipe's action_rate ramp — ported verbatim.

    Smoothness is a MOTION-BLOCKER while the swing is still being discovered and a
    jitter-damper once it exists, so the mjlab cfg holds it at the stage-0 value
    until the kick has formed and only then ramps it to full strength. Freezing it
    at -0.1 (which is what this task did before this curriculum existed) never
    applies that discipline: the strike stays as thrashy at iteration 4000 as at
    400, and ball speed keeps hunting instead of settling on TARGET_BALL_SPEED.

    Stages are the mjlab ones (-0.1 -> -1.0 by iteration 1500). Per that cfg's own
    note: if the converged kick comes out too WEAK, softening the ramp end
    (-1.0 -> -0.6) is the first knob to try, not the stage timing.
    """

    # NOT a curriculum: the only per-reset hook that fires before the reset events
    # move the ball. See `mdp.kick_success` -- it publishes Metrics/kick_success_rate
    # and Metrics/kick_distance, the task's ONLY real measure of whether the ball
    # was kicked. `Metrics/success_rate` is Isaac Lab's twist-tracking metric and
    # says nothing about the ball.
    kick_success = CurrTerm(
        func=mdp.kick_success,
        params={"min_distance": KICK_SUCCESS_DISTANCE, "max_tilt_deg": KICK_SUCCESS_MAX_TILT_DEG},
    )

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )


##
# Environment.
##


@configclass
class MicroduckBallKickFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Kick the ball forward, from a standing start, on flat ground."""

    scene: BallKickSceneCfg = BallKickSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: BallKickObservationsCfg = BallKickObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: BallKickRewardsCfg = BallKickRewardsCfg()
    terminations: BallKickTerminationsCfg = BallKickTerminationsCfg()
    events: BallKickEventsCfg = BallKickEventsCfg()
    curriculum: BallKickCurriculumCfg = BallKickCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        # 5 s: an episodic trick, not a gait. Long episodes here just pay the
        # standing rewards for a policy that never kicks.
        self.episode_length_s = 5.0

        from isaaclab.sim import SimulationCfg
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=self.decimation,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=10, ls_iterations=20), num_substeps=1),
        )
        self.viewer.eye = (0.8, 0.8, 0.4)
        self.viewer.lookat = (0.1, 0.0, 0.1)

        # The twist command is carried for obs-shape parity only: this task is a
        # standing kick. Ranges stay tiny but NON-ZERO so the input neurons stay
        # alive for policies that later share these weights.
        self.commands.twist.ranges.lin_vel_x = (-0.01, 0.01)
        self.commands.twist.ranges.lin_vel_y = (-0.01, 0.01)
        self.commands.twist.ranges.ang_vel_z = (-0.01, 0.01)
        self.commands.twist.rel_turn_in_place_envs = 0.0


@configclass
class MicroduckBallKickFlatEnvCfg_PLAY(MicroduckBallKickFlatEnvCfg):
    """Small, quiet variant for watching a policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_mass = None
        self.events.base_com = None
        self.events.joint_armature = None
