"""Access to the MuJoCo reference dumps in ``assets/reference/``.

The dumps are produced by ``scripts/dump_mjcf_reference.py`` straight from the
MJCFs, and they are the authority on everything the USD conversion must preserve:
the canonical 14-servo joint ORDER, per-joint limits/armature/friction, body
masses, site frames, and the per-geom contact parameters USD has no concept of.

Reading them at runtime (rather than code-generating a Python table) keeps the
cfgs in lockstep with the robot: regenerate the dump and everything follows.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference"


@lru_cache(maxsize=None)
def load_reference(model: str) -> dict:
    """Load one model's MuJoCo reference dump."""
    path = _REFERENCE_DIR / f"{model}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No reference dump for model '{model}' at {path}. "
            "Run: python scripts/dump_mjcf_reference.py"
        )
    return json.loads(path.read_text())


def servo_joint_names(model: str) -> list[str]:
    """The 14 actuated joints, in canonical order.

    0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8 neck/head
    (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
    """
    return list(load_reference(model)["servo_joints"])


def passive_joint_names(model: str) -> list[str]:
    """Unactuated joints (wheels, backlash hinges) — always ``passive_*``."""
    return list(load_reference(model)["passive_joints"])


def joint_property(model: str, prop: str) -> dict[str, float]:
    """``{joint_name: value}`` for a per-joint MuJoCo property, servos only.

    Useful props: ``armature``, ``damping``, ``frictionloss``.
    """
    servos = set(servo_joint_names(model))
    return {j["name"]: j[prop] for j in load_reference(model)["joints"] if j["name"] in servos}


def joint_limits(model: str) -> dict[str, tuple[float, float]]:
    """``{joint_name: (lower, upper)}`` in radians, servos only."""
    servos = set(servo_joint_names(model))
    return {
        j["name"]: (j["range"][0], j["range"][1])
        for j in load_reference(model)["joints"]
        if j["name"] in servos
    }


def site(model: str, name: str) -> dict:
    """One site's frame: ``{"name", "body", "pos", "quat"}``.

    MuJoCo sites survive the USD conversion as geometry, but the port does not
    rely on that: sensors and rewards use these offsets relative to the parent
    BODY, which is a representation both stacks agree on.
    """
    for entry in load_reference(model)["sites"]:
        if entry["name"] == name:
            return entry
    known = [s["name"] for s in load_reference(model)["sites"]]
    raise KeyError(f"No site '{name}' in model '{model}'. Known: {known}")


def contact_geoms(model: str) -> list[dict]:
    """Per-geom contact parameters from MuJoCo (``condim``, ``priority``, ``friction``).

    USD carries none of this, so it is re-authored on the Isaac Lab side; this is
    what that re-authoring is checked against.
    """
    return list(load_reference(model)["geoms"])


def usd_body_name(model: str, mujoco_name: str) -> str:
    """USD prim name for a MuJoCo body.

    The Isaac Sim MJCF importer disambiguates a body whose name collides with a
    JOINT name by appending ``_1`` (on Microduck only ``neck_pitch`` collides: the
    MJCF has both a ``neck_pitch`` hinge and a ``neck_pitch`` body). Everything
    else keeps its name. The parity check verifies this rule per model rather
    than trusting it.
    """
    reference = load_reference(model)
    joint_names = {j["name"] for j in reference["joints"]}
    return f"{mujoco_name}_1" if mujoco_name in joint_names else mujoco_name


def body_name_map(model: str) -> dict[str, str]:
    """``{mujoco_body_name: usd_prim_name}`` for every body in the model."""
    return {
        b["name"]: usd_body_name(model, b["name"])
        for b in load_reference(model)["bodies"]
        if b["name"] != "world"
    }
