"""Isaac Lab articulation configs for the Microduck robot.

Port of `mjlab_microduck.robot.microduck_constants`. Every number here is either
read from the MuJoCo reference dump (see `reference.py`) or carried over from the
mjlab constants with its original comment — these are calibrated values, not
defaults, and several of them cost real debugging weeks.

Model choice per task family mirrors the mjlab stack:

* ``walk``          — minimal collision (feet only), the velocity recipe
* ``allcollisions`` — full collision, standup / sitstand / ground-pick / ball-kick
* ``rollers``       — full collision + passive wheel hinges, roller tasks
* ``*_backlash``    — the backlash twin of each of the above

A ``-Backlash-`` task MUST use the backlash twin of its base task's model, or the
backlash A/B is confounded.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_microduck.assets import usd_path

from .reference import joint_limits, joint_property, servo_joint_names

##
# Joint selectors.
##

#: Every actuated joint. Unactuated joints (wheels, backlash hinges, the jaw
#: linkage) are ALL named ``passive_*``, and every actuator/observation/reward
#: selector excludes them with this pattern. Keep the prefix convention when
#: adding joints.
SERVO_JOINTS_EXPR = r"^(?!passive_).*"

#: The four neck/head servos, in canonical order (indices 5-8).
HEAD_JOINT_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

#: Passive wheel hinges on the roller models. Deliberately NOT ``^passive_.*``:
#: that would also match the backlash hinges on the roller-backlash model.
WHEEL_JOINTS_EXPR = r"^passive_.*wheel"

##
# HOME frame.
##

#: STAND2 pose. The trunk sits ~5 mm forward over the feet so the CoM is over the
#: ankle axis; at the older HOME it sat ~5 mm behind, which biased the robot
#: backward and made the standup policy droop its head forward as a counterweight.
#: Leg pitch chain leaned forward: hip_pitch 30 deg -> 26.24, ankle 30 -> 25.95,
#: knee 0 -> 0.28. Matches the STAND keyframe in the MJCF scenes.
#:
#: Keys are EXACT joint names, not the mjlab stack's `.*hip_pitch.*` patterns.
#: Isaac Lab and mjlab both resolve these with `re.match`, whose leading `.*`
#: happily consumes a `passive_` prefix: on the backlash models
#: `.*hip_pitch.*` also matches `passive_left_hip_pitch_backlash`, handing a
#: +-1 deg gear-play hinge a 0.4579 rad start (26x its range). Verified against
#: mjlab's own `resolve_expr`: 10 of 14 backlash hinges get a non-zero init and 9
#: of those land outside their limits. Exact names cannot do that, and
#: `test_home_pose_covers_every_servo_exactly_once` holds the line.
HOME_JOINT_POS: dict[str, float] = {
    # Left leg (servo indices 0-4).
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    # Neck / head (servo indices 5-8).
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    # Right leg (servo indices 9-13), mirrored.
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

#: Deep-crouch pose, READ OFF THE REAL ROBOT (Dynamixel XL330) — a holdable pose.
#: Ported verbatim from `microduck_roller_crouch_env_cfg.CROUCH_POSE`.
#:
#: Useful as ground truth for what "squat" means on this robot, which is not
#: obvious from HOME: the pitch chain CROSSES ZERO on the way down
#: (left_hip_pitch -0.4579 -> +1.4082), so "bend further in the direction HOME
#: already leans" is wrong for hip_pitch and ankle.
#:
#: Note the mirrored convention: every SYMMETRIC change carries opposite signs on
#: the left and right joints. An ASYMMETRIC move (leaning to one side) therefore
#: uses the same sign on both.
CROUCH_JOINT_POS: dict[str, float] = {
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}

#: Trunk spawn height (m). The MJCF spawns the trunk at z = 0.12; standing trunk z
#: is a MEASURED quantity, not this one — measure it in Isaac Lab under a standing
#: policy rather than carrying a value across model revisions or across stacks.
SPAWN_HEIGHT = 0.12

##
# Per-joint physical properties (from the MuJoCo reference dump).
##

# XL330 reflected rotor inertia. Armature DOES affect the BAM actuator (it is set,
# not zeroed), which is why it is randomized in DR while joint frictionloss is not.
_ARMATURE = joint_property("walk", "armature")
_DAMPING = joint_property("walk", "damping")
_FRICTION = joint_property("walk", "frictionloss")

#: XL330 torque ceiling, from the MJCF actuator force range. Physical, keep it.
_EFFORT_LIMIT = 0.96

#: Gains for the P2 conversion check ONLY — not a physical model of anything.
#:
#: The MJCF's own position-servo gain (kp = 0.55) cannot hold the robot upright in
#: EITHER stack: plain MuJoCo collapses to ~178 deg tilt under it, because that
#: gain is an identified XL330 parameter for BAM to fit against, not a usable
#: standalone controller. A gain that does hold the pose is needed to compare the
#: two stacks at all, and the same value is used on both sides of the comparison.
_CHECK_KP = 5.0
_CHECK_KD = 0.1


def _servo_actuator_placeholder() -> IdealPDActuatorCfg:
    """TEMPORARY actuator, for the P2 conversion check only.

    NOT the deployment actuator: Microduck's servos are BAM (voltage-controlled
    XL330 with load-dependent friction), and a policy trained against this
    placeholder will not transfer to hardware. P3 replaces it; nothing else should
    depend on it.

    Deliberately EXPLICIT (`IdealPDActuator`, torque computed in Python and applied
    as an effort) rather than implicit. `ImplicitActuator` delegates the PD law to
    the simulator's own joint drives, and this asset has none: the MJCF importer
    reports "Gain and bias prm arrays are not in the expected format ... physics
    drive stiffness and damping will not be created" for all 14 joints, and the
    MuJoCo model Newton ends up building has `nu = 0`. Under an implicit actuator
    the robot is therefore completely limp — it looks exactly like broken collision
    and cost a full debugging detour to pin down. BAM is explicit too, so this
    limitation never binds the port.
    """
    return IdealPDActuatorCfg(
        joint_names_expr=[SERVO_JOINTS_EXPR],
        stiffness=_CHECK_KP,
        damping=_CHECK_KD,
        effort_limit=_EFFORT_LIMIT,
        armature=_ARMATURE,
        friction=_FRICTION,
    )


def make_microduck_cfg(model: str = "walk", *, prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Build the articulation config for one Microduck model.

    Args:
        model: Key into ``isaaclab_microduck.assets.ROBOT_MJCF`` (``walk``,
            ``allcollisions``, ``rollers``, or a ``*_backlash`` twin).
        prim_path: Scene prim path for the articulation.
    """
    usd = usd_path(model)
    if not usd.is_file():
        raise FileNotFoundError(
            f"USD for model '{model}' not found at {usd}. "
            f"Run: python scripts/convert_assets.py --model {model}"
        )

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            # Contact sensors drive the feet air-time / self-collision terms.
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                # A 25 cm robot tumbles at 3.5-5.5 rad/s naturally; do not impose
                # human-scale speed intuitions through caps here.
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # The full-collision models exist so limbs CAN hit the trunk, and
                # `self_collisions` is a live reward term.
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, SPAWN_HEIGHT),
            joint_pos=dict(HOME_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={"servos": _servo_actuator_placeholder()},
    )


#: Convenience handles mirroring the mjlab constant names.
MICRODUCK_WALK_CFG = None  # built lazily by make_microduck_cfg("walk")


def servo_joint_order(model: str = "walk") -> list[str]:
    """Canonical servo order for a model (0-4 left leg, 5-8 head, 9-13 right leg).

    NEVER hardcode joint indices against this: on roller/backlash models the
    passive joints interleave, and Isaac Lab orders articulation joints its own
    way. Resolve indices at runtime by name.
    """
    return servo_joint_names(model)


def servo_joint_limits(model: str = "walk") -> dict[str, tuple[float, float]]:
    """Per-joint (lower, upper) limits in radians, from the MJCF."""
    return joint_limits(model)
