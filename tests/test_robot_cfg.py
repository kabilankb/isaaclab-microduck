"""P2 gate (config half): the robot models keep the invariants the port depends on.

CPU-only and simulator-free — everything here reads the MuJoCo reference dumps in
`assets/reference/`, so it runs in CI. The physics half of the gate (contact,
standing height, tilt) lives in `scripts/check_asset_parity.py`, which needs a GPU.
"""

import re

import pytest

from isaaclab_microduck.assets import ROBOT_MJCF, mjcf_path
from isaaclab_microduck.robot.microduck_cfg import (
    HEAD_JOINT_NAMES,
    HOME_JOINT_POS,
    SERVO_JOINTS_EXPR,
    WHEEL_JOINTS_EXPR,
)
from isaaclab_microduck.robot.reference import (
    body_name_map,
    joint_limits,
    load_reference,
    passive_joint_names,
    servo_joint_names,
    site,
)

# Every model except the ball prop is a robot with the full servo set.
ROBOT_MODELS = [m for m in sorted(ROBOT_MJCF) if m != "ball"]

#: The canonical 14-servo layout: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
CANONICAL_SERVO_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_mjcf_source_exists(model):
    assert mjcf_path(model).is_file()


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_canonical_servo_order(model):
    """All robot models share one servo order, and it is the documented one."""
    assert servo_joint_names(model) == CANONICAL_SERVO_ORDER


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_unactuated_joints_use_the_passive_prefix(model):
    """Every unactuated joint is `passive_*` — the convention every selector relies on."""
    reference = load_reference(model)
    servos = set(servo_joint_names(model))
    for joint in reference["joints"]:
        if joint["type"] == "free" or joint["name"] in servos:
            continue
        assert joint["name"].startswith("passive_"), joint["name"]


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_servo_expr_selects_exactly_the_servos(model):
    """`^(?!passive_).*` must select the 14 servos and nothing else."""
    pattern = re.compile(SERVO_JOINTS_EXPR)
    reference = load_reference(model)
    selected = [
        j["name"] for j in reference["joints"] if j["type"] != "free" and pattern.match(j["name"])
    ]
    assert selected == servo_joint_names(model)


def test_wheel_expr_does_not_match_backlash_joints():
    """The wheel selector must not catch backlash hinges.

    On `rollers_backlash` both kinds of `passive_*` joint coexist, so a naive
    `^passive_.*` silently swallows the 14 backlash hinges along with the 4 wheels.
    """
    pattern = re.compile(WHEEL_JOINTS_EXPR)
    passives = passive_joint_names("rollers_backlash")
    matched = [name for name in passives if pattern.match(name)]
    assert sorted(matched) == ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
    assert not any("backlash" in name for name in matched)


def test_backlash_models_add_one_hinge_per_servo():
    for model in ("walk_backlash", "allcollisions_backlash", "rollers_backlash"):
        backlash = [n for n in passive_joint_names(model) if n.endswith("_backlash")]
        assert len(backlash) == len(CANONICAL_SERVO_ORDER)
        for servo in CANONICAL_SERVO_ORDER:
            assert f"passive_{servo}_backlash" in backlash


def test_backlash_joints_interleave_with_servos():
    """Backlash hinges sit IN SERIES with their servo, so the joint array interleaves.

    This is exactly why no mdp function may hardcode a joint index: indices written
    against the 14-servo layout select the wrong joint here.
    """
    joints = [j["name"] for j in load_reference("walk_backlash")["joints"] if j["type"] != "free"]
    assert joints[:4] == [
        "left_hip_yaw",
        "passive_left_hip_yaw_backlash",
        "left_hip_roll",
        "passive_left_hip_roll_backlash",
    ]


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_home_pose_covers_every_servo_exactly_once(model):
    """HOME must name all 14 servos and touch nothing else.

    A HOME regex that quietly misses a joint leaves it at 0 rad, which is a
    different robot; one that over-matches would drag a passive joint with it.
    """
    servos = servo_joint_names(model)
    passives = passive_joint_names(model)
    matched: dict[str, list[str]] = {name: [] for name in servos}
    for expr in HOME_JOINT_POS:
        pattern = re.compile(expr)
        for name in servos:
            if pattern.match(name):
                matched[name].append(expr)
        for name in passives:
            assert not pattern.match(name), f"HOME expr {expr!r} matches passive joint {name}"

    unmatched = [name for name, exprs in matched.items() if not exprs]
    duplicated = {name: exprs for name, exprs in matched.items() if len(exprs) > 1}
    assert not unmatched, f"servos not covered by HOME: {unmatched}"
    assert not duplicated, f"servos matched by several HOME entries: {duplicated}"


def test_head_joint_names_are_the_neck_head_block():
    """The head command block is servo indices 5-8, in that order."""
    assert list(HEAD_JOINT_NAMES) == CANONICAL_SERVO_ORDER[5:9]


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_home_pose_is_inside_the_joint_limits(model):
    limits = joint_limits(model)
    for expr, value in HOME_JOINT_POS.items():
        pattern = re.compile(expr)
        for name, (low, high) in limits.items():
            if pattern.match(name):
                assert low <= value <= high, f"{name}: HOME {value} outside [{low}, {high}]"


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_foot_sites_resolve(model):
    """Foot sites drive the terrain-height sensors and several rewards."""
    for name in ("left_foot", "right_foot"):
        entry = site(model, name)
        assert entry["body"].startswith("ankle_")
        assert len(entry["pos"]) == 3


@pytest.mark.parametrize("model", ROBOT_MODELS)
def test_body_rename_rule_only_touches_joint_name_clashes(model):
    """The importer renames a body only when its name collides with a joint name."""
    reference = load_reference(model)
    joint_names = {j["name"] for j in reference["joints"]}
    for mujoco_name, usd_name in body_name_map(model).items():
        if mujoco_name in joint_names:
            assert usd_name == f"{mujoco_name}_1"
        else:
            assert usd_name == mujoco_name
