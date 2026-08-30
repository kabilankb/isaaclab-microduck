"""Microduck velocity (walking) task for Isaac Lab / Newton MJWarp.

Port of `mjlab_microduck.tasks.microduck_velocity_env_cfg`. Term names, weights,
standard deviations and curriculum stages are carried over from the mjlab recipe,
which is the product of many runs — treat the numbers as calibration, not defaults.

THE OBSERVATION CONTRACT (do not break):

    actor obs = 61D = 48 proprioception + 13D command block

    [ base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
      actions(14), twist(3), head_pose(4), body_pose(6) ]

in exactly that order, so policies stay hot-swappable in the deployment runtime.
An env that does not use a command slot ZERO-PADS it (keeps the term, samples a
tiny range) — never deletes it. `tests/test_velocity_cfg.py` locks this down.

KNOWN GAPS vs the mjlab recipe (this is a sim milestone, not a deployable policy):

* **Actuator** — the P2 conversion-check PD, not BAM. Swap in the BAM actuator
  (P3) before any policy is exported to hardware; expect a retrain.
* **Observation delays** and the **encoder-bias** term are not ported yet; both
  need infrastructure Isaac Lab lacks (per-term random lag).
* Rough terrain, foot terrain-height ray sensors and the backlash variants are
  out of scope for this first cut.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_microduck.robot.microduck_cfg import SERVO_JOINTS_EXPR, make_microduck_cfg
from isaaclab_microduck.tasks import mdp

##
# Toggles and tuned constants (mirrors the mjlab cfg's header block).
##

NUM_STEPS_PER_ENV = 24
"""Curriculum steps are env steps: iteration * NUM_STEPS_PER_ENV."""

ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_COM_RANDOMIZATION = True
ENABLE_MASS_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True

IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # deg, zero-centered random axis
COM_RANDOMIZATION_RANGE = 0.003            # +/- 3 mm, ramped by curriculum
MASS_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
"""+/-0.3 m/s. An earlier +/-0.5 was found to be a large fraction of the robot's
own top speed, i.e. a shove rather than a disturbance."""

TURN_IN_PLACE_FRACTION = 0.15
"""Explicit turn-in-place bucket. Independent uniform sampling makes
'lin about 0, |ang| large' ~2% of experience, so spinning never trained."""

HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

FOOT_BODIES = ("ankle_left", "ankle_right")
"""The bodies carrying the sole collision geometry."""

TRUNK_BODY = "trunk_base"


##
# Scene.
##


@configclass
class MicroduckSceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with the walk-collision Microduck."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    robot = make_microduck_cfg("walk")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lights are scene ASSETS, not bare spawn cfgs: the scene builder rejects a
    # raw DomeLightCfg with "Unknown asset config type".
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP.
##


@configclass
class CommandsCfg:
    """The 13D command block: twist(3) + head_pose(4) + body_pose(6)."""

    twist = mdp.MicroduckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        # FIXED ranges, deliberately modest. Widening to lin +/-0.4 / ang +/-2.0
        # once outpaced the robot's capability and tracked a post-iteration-1000
        # decline in both reward and episode length. ang +/-1.0 is the change that
        # makes turning learnable at all.
        ranges=mdp.MicroduckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
        rel_turn_in_place_envs=TURN_IN_PLACE_FRACTION,
    )

    # Head pose: 4D deltas from HOME in joint order
    # [neck_pitch, head_pitch, head_yaw, head_roll]. A PRIMARY objective here.
    # Ranges start small but NON-ZERO so the input neurons stay alive from step 0;
    # the curriculum widens them toward each joint's mechanical limit.
    head_pose = mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll — much smaller mechanical range
        ),
        zero_command_prob=0.1,
    )

    # Body pose: 6D [x, y, z, roll, pitch, yaw] delta from nominal standing.
    # Carried at reward weight 0 purely for runtime obs-shape parity; the standup
    # family raises the weight and widens these.
    body_pose = mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
        zero_command_prob=0.1,
    )


@configclass
class ActionsCfg:
    """Position targets on the 14 servos, offset from HOME.

    scale 1.0 and `use_default_offset=True`: the action IS the joint delta from
    HOME. Policies are UNFILTERED — no action low-pass in training — because a
    trained-with / deployed-without mismatch (either direction) breaks transfer.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[SERVO_JOINTS_EXPR],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Actor sees `policy`; critic sees `policy` + `privileged`."""

    @configclass
    class PolicyCfg(ObsGroup):
        """THE 61D CONTRACT. Term order here IS the observation layout."""

        # -- 48D proprioception --
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_imu_misaligned,
            params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_imu_misaligned,
            params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
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

        # -- 13D command block, in runtime order --
        twist_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        head_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic-only extras. Never part of the 61D actor contract."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventsCfg:
    """Domain randomization.

    Every term here must be NON-ACCUMULATING across resets. Isaac Lab's
    `randomize_*` ops re-read the stored defaults each time, which is what makes
    them safe; a custom op must restore-then-apply. An accumulating CoM randomizer
    once degraded every long run for months.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODIES)),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.6, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": MASS_RANDOMIZATION_RANGE,
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY]),
            "com_range": {
                "x": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
                "y": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
                "z": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[SERVO_JOINTS_EXPR]),
            "armature_distribution_params": ARMATURE_RANDOMIZATION_RANGE,
            "operation": "scale",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=VELOCITY_PUSH_INTERVAL_S,
        params={"velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
    )


@configclass
class RewardsCfg:
    """Weights carried over from the mjlab recipe.

    Sign convention: every penalty function returns >= 0 and carries a NEGATIVE
    weight. On every run, each `Episode_Reward/<penalty>` must log <= 0.
    """

    # -- task --
    # Weight 6.0, NOT the mjlab recipe's 2.0. Measured reason: at 2.0 the policy
    # converged to standing still and stayed there for a full 6000-iteration run --
    # 0.8 mm travelled in 6 s under a 1.2 m/s command. The terms a STATIONARY
    # upright robot still collects (upright 2.0 + head_pose_tracking 2.0 + pose 1.0
    # + track_angular_velocity 2.0 at low commands) total ~7.0, so walking risked
    # all of that to gain at most 2.0. Standing was simply the better trade, and
    # PPO took it.
    #
    # Compare reward MASS, not weights: locomotion has to outbid the standing stack
    # or it will not happen. The measured check is `Metrics/base_speed_x` -- metres
    # per second, not a reward -- because `Episode_Reward/track_linear_velocity`
    # rose to ~1.2 throughout the run that never moved.
    # DENSE bootstrap, linear in forward speed. The Gaussian below is precision
    # shaping and has no gradient far from target -- at std sqrt(0.1) a robot doing
    # 0.29 m/s under a 1.2 m/s command scores exp(-8.3) ~ 0.00025, which is where
    # two 6000-iteration runs stalled. Linear pays for every bit of progress from a
    # standing start, the same shape that bootstrapped `ball_forward_velocity`.
    forward_speed = RewTerm(
        func=mdp.forward_speed_linear,
        weight=6.0,
        params={"command_name": "twist"},
    )
    track_linear_velocity = RewTerm(
        func=mdp.track_linear_velocity,
        weight=6.0,
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )
    upright = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY])},
    )
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            # LEG joints only — the head is command-driven (see head_pose_tracking).
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r"^(?!passive_|.*neck.*|.*head.*).*"]),
            "command_name": "twist",
            "std_standing": {
                r".*hip_yaw.*": 0.1,
                r".*hip_roll.*": 0.05,  # hold the 5-deg inward stance, stop leg splay
                r".*hip_pitch.*": 0.15,
                r".*knee.*": 0.15,
                r".*ankle.*": 0.1,
            },
            "std_walking": {
                r".*hip_yaw.*": 0.3,
                r".*hip_roll.*": 0.05,
                r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.25,
            },
            "std_running": {
                r".*hip_yaw.*": 0.3,
                r".*hip_roll.*": 0.05,
                r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.25,
            },
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
        },
    )
    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    body_pose_tracking = RewTerm(
        func=mdp.body_pose_tracking_6d,
        weight=0.0,  # infra alive, not steering the policy
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # -- gait --
    air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES)),
            "command_name": "twist",
            "threshold": 0.125,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.02,
            "command_name": "twist",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODIES)),
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "target_height": 0.02,
            "command_name": "twist",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODIES)),
        },
    )
    foot_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODIES)),
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODIES)),
        },
    )

    # -- regularizers --
    body_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,  # a MOTION-BLOCKER: keep low
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TRUNK_BODY])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    head_pose_bias = RewTerm(
        func=mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped in by curriculum, AFTER a gait exists
        params={"command_name": "head_pose", "tau_s": 1.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    nan_state = DoneTerm(func=mdp.robot_state_is_nan)


@configclass
class CurriculumCfg:
    """Step functions, phase-aligned with what the policy has actually learned.

    If a wandb metric steps DOWN exactly at a stage boundary, the pacing is wrong:
    stretch the stage or move the introduction LATER, never earlier.
    """

    # Smoothness AFTER skill discovery: an attempt-tax during exploration makes
    # "do nothing" win.
    # Scoring hook, not a curriculum: publishes MEASURED speed in m/s so a
    # stationary policy cannot look successful through reward terms again.
    locomotion_metrics = CurrTerm(func=mdp.locomotion_metrics)

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.15},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.2},
            ],
        },
    )

    # Zero-command behaviour must be trained explicitly — it is the idle state.
    standing_envs = CurrTerm(
        func=mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.05},
                {"step": 1000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.1},
            ],
        },
    )

    head_pose_range = CurrTerm(
        func=mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 500 * NUM_STEPS_PER_ENV, "ranges": ((-0.3, 0.3), (-0.3, 0.3), (-0.4, 0.4), (-0.1, 0.1))},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": ((-0.7, 0.7), (-0.7, 0.7), (-0.9, 0.9), (-0.2, 0.2))},
                {"step": 1500 * NUM_STEPS_PER_ENV, "ranges": ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))},
            ],
        },
    )

    # OFF until a gait exists: a posture-precision term is a distraction before
    # then. See head_pose_bias_penalty for why this is a DC term, not a tighter std.
    head_pose_bias_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": -1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -3.0},
            ],
        },
    )


##
# Environment.
##


@configclass
class MicroduckVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Microduck walking on flat ground, 50 Hz control."""

    scene: MicroduckSceneCfg = MicroduckSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 200 Hz physics, 50 Hz control — matches the mjlab recipe and the runtime.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=self.decimation,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=10, ls_iterations=20), num_substeps=1),
        )
        self.viewer.eye = (1.0, 1.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.1)

        if not ENABLE_IMU_ORIENTATION_RANDOMIZATION:
            self.observations.policy.base_ang_vel.params["max_angle_deg"] = 0.0
            self.observations.policy.projected_gravity.params["max_angle_deg"] = 0.0
        if not ENABLE_COM_RANDOMIZATION:
            self.events.base_com = None
        if not ENABLE_MASS_RANDOMIZATION:
            self.events.base_mass = None
        if not ENABLE_ARMATURE_RANDOMIZATION:
            self.events.joint_armature = None
        if not ENABLE_VELOCITY_PUSHES:
            self.events.push_robot = None


@configclass
class MicroduckVelocityFlatEnvCfg_PLAY(MicroduckVelocityFlatEnvCfg):
    """Small, quiet variant for looking at a policy rather than training one."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_mass = None
        self.events.base_com = None
        self.events.joint_armature = None
