"""The teleop key map drives real joints, with correctly mirrored signs.

CPU-only. A mirrored sign error here twists the robot instead of squatting it —
the kind of thing that is easy to miss by eye in a viewer and trivial to assert.
"""

import pytest

from isaaclab_microduck.robot.microduck_cfg import (
    CROUCH_JOINT_POS,
    HEAD_JOINT_NAMES,
    HOME_JOINT_POS,
)
from isaaclab_microduck.robot.reference import servo_joint_names
from isaaclab_microduck.utils.teleop_keys import HEAD_KEYS, HELP, LEG_KEYS

SERVOS = set(servo_joint_names("walk"))


def test_every_mapped_joint_exists():
    for joint, _ in HEAD_KEYS.values():
        assert joint in SERVOS, joint
    for mapping in LEG_KEYS.values():
        for joint in mapping:
            assert joint in SERVOS, joint


def test_head_keys_cover_all_four_head_joints_both_ways():
    """The head block is the 4 servos the `head_pose` command drives."""
    per_joint: dict[str, set[float]] = {}
    for joint, sign in HEAD_KEYS.values():
        per_joint.setdefault(joint, set()).add(sign)
    assert set(per_joint) == set(HEAD_JOINT_NAMES)
    for joint, signs in per_joint.items():
        assert signs == {+1.0, -1.0}, f"{joint} is not drivable in both directions"


@pytest.mark.parametrize("forward,backward", [("s", "w"), ("a", "d")])
def test_opposed_keys_are_exact_inverses(forward, backward):
    assert set(LEG_KEYS[forward]) == set(LEG_KEYS[backward])
    for joint, sign in LEG_KEYS[forward].items():
        assert LEG_KEYS[backward][joint] == -sign, joint


def test_squat_is_mirrored_across_the_legs():
    """Left and right carry OPPOSITE signs on the pitch chain.

    HOME mirrors them (left_hip_pitch -0.4579 vs right +0.4579), so a symmetric
    squat needs opposite signs per side; the same sign on both would twist the
    robot instead of lowering it.
    """
    squat = LEG_KEYS["s"]
    for suffix in ("hip_pitch", "knee", "ankle"):
        left, right = squat[f"left_{suffix}"], squat[f"right_{suffix}"]
        assert left == -right, f"{suffix}: {left} vs {right} — squat is not mirrored"


def test_lean_applies_the_same_sign_to_both_hips():
    """hip_roll is also mirrored at HOME, so a LEAN uses the same sign on both."""
    lean = LEG_KEYS["a"]
    assert lean["left_hip_roll"] == lean["right_hip_roll"]


def test_squat_moves_every_joint_toward_the_measured_crouch():
    """Each squat sign must point from HOME toward the real robot's crouch pose.

    Not a heuristic: `CROUCH_JOINT_POS` was read off the hardware. It matters
    because the pitch chain CROSSES ZERO on the way down (left_hip_pitch
    -0.4579 -> +1.4082), so the intuitive "bend further the way HOME already
    leans" rule gives the wrong sign for hip_pitch and ankle — and an earlier
    hand-written map had both knees inverted, which this catches.
    """
    for joint, sign in LEG_KEYS["s"].items():
        delta = CROUCH_JOINT_POS[joint] - HOME_JOINT_POS[joint]
        assert sign * delta > 0, (
            f"{joint}: squat sign {sign:+.0f} moves away from crouch "
            f"(HOME {HOME_JOINT_POS[joint]:+.4f} -> CROUCH {CROUCH_JOINT_POS[joint]:+.4f})"
        )


def test_symmetric_moves_use_opposite_signs_per_side():
    """The left/right mirrored convention, verified against the measured crouch.

    Every symmetric change carries opposite signs on the two sides. That is why a
    LEAN — an asymmetric move — uses the SAME sign on both hips.
    """
    for suffix in ("hip_pitch", "knee", "ankle", "hip_roll"):
        left = CROUCH_JOINT_POS[f"left_{suffix}"] - HOME_JOINT_POS[f"left_{suffix}"]
        right = CROUCH_JOINT_POS[f"right_{suffix}"] - HOME_JOINT_POS[f"right_{suffix}"]
        assert left * right < 0, f"{suffix}: expected mirrored deltas, got {left:+.3f} / {right:+.3f}"


def test_no_key_is_bound_twice():
    overlap = set(HEAD_KEYS) & set(LEG_KEYS)
    assert not overlap, f"keys bound in both maps: {overlap}"
    reserved = {"q", "h", "r", "[", "]", "up", "down"}
    assert not (set(HEAD_KEYS) | set(LEG_KEYS)) & reserved


def test_help_text_mentions_every_bound_key():
    for key in list(HEAD_KEYS) + list(LEG_KEYS):
        assert key in HELP, f"key '{key}' is bound but undocumented in HELP"
