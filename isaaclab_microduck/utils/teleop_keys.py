"""Key map for keyboard teleop of the Microduck joint targets.

Kept in the package (rather than in the script that uses it) so it can be unit
tested without launching a simulator — a mirrored sign error here twists the robot
instead of squatting it, and that is exactly the sort of thing a test should catch
rather than your eyes in a viewer.

Grouped moves come first because they are what you actually want to drive: the head
block is the same 4 servos the `head_pose` command controls, and squat / lean are
the coordinated leg moves. Per-joint nudging is the escape hatch.
"""

from __future__ import annotations

HELP = """
  head      i / k   neck_pitch down / up        j / l   head_yaw left / right
            u / o   head_pitch down / up        n / m   head_roll left / right
  legs      w / s   stand / squat (hip+knee+ankle)
            a / d   lean left / right (hip_roll)
  joint     [ / ]   select previous / next joint
            up/down nudge the selected joint
  other     r       reset to HOME               h   this help
            q       quit
"""

#: key -> (joint name, sign). One step per press.
HEAD_KEYS: dict[str, tuple[str, float]] = {
    "i": ("neck_pitch", -1.0), "k": ("neck_pitch", +1.0),
    "u": ("head_pitch", -1.0), "o": ("head_pitch", +1.0),
    "j": ("head_yaw", +1.0),   "l": ("head_yaw", -1.0),
    "n": ("head_roll", +1.0),  "m": ("head_roll", -1.0),
}

# Coordinated leg moves: key -> {joint name: sign}.
#
# Signs follow the mirrored HOME convention — the left and right legs carry
# opposite signs on the pitch chain (left_hip_pitch -0.4579 vs right +0.4579), so
# a symmetric squat must use opposite signs per side. Using the same sign on both
# would yaw/twist the robot instead of lowering it.
# Signs are taken from the measured HOME -> CROUCH_JOINT_POS deltas, not guessed:
# the pitch chain crosses zero on the way down, so "bend further in the direction
# HOME already leans" gives the wrong answer for hip_pitch and ankle, and an
# earlier hand-written version had both knees inverted (caught by the tests).
LEG_KEYS: dict[str, dict[str, float]] = {
    "s": {
        "left_hip_pitch": +1.0, "right_hip_pitch": -1.0,
        "left_knee": +1.0, "right_knee": -1.0,
        "left_ankle": -1.0, "right_ankle": +1.0,
    },
    # hip_roll is mirrored the same way at HOME (left -0.0873, right +0.0873), so a
    # lean applies the SAME sign to both sides to tip the whole body one way.
    "a": {"left_hip_roll": +1.0, "right_hip_roll": +1.0},
}
LEG_KEYS["w"] = {joint: -sign for joint, sign in LEG_KEYS["s"].items()}
LEG_KEYS["d"] = {joint: -sign for joint, sign in LEG_KEYS["a"].items()}
