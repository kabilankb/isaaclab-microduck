#!/usr/bin/env python
"""Dump MuJoCo ground truth for every Microduck model to JSON.

This is the baseline the MJCF -> USD conversion is checked against
(`check_asset_parity.py`), and the source of the generated site-offset table. It
reads the MJCF with plain MuJoCo, so it needs neither Isaac Sim nor a GPU.

    python scripts/dump_mjcf_reference.py            # all models
    python scripts/dump_mjcf_reference.py --model walk

Output: `isaaclab_microduck/assets/reference/<model>.json` (committed on purpose —
it is small, and a diff in it means the robot changed).

What is captured is exactly what the port is not allowed to get wrong: the
14-servo joint ORDER, per-joint limits/armature/friction, body mass and inertia,
site frames (used for foot sensors and the mouth tip), and the per-geom contact
parameters (`condim`, `priority`, `friction`) that USD has no native concept of.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from isaaclab_microduck.assets import ROBOT_MJCF, mjcf_path

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "isaaclab_microduck" / "assets" / "reference"

# MuJoCo joint type enum -> readable name.
_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def _name(model: mujoco.MjModel, obj_type: int, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or f"<unnamed_{index}>"


def _round(value, digits: int = 9):
    """JSON-friendly rounding; keeps diffs readable and float noise out of them."""
    array = np.asarray(value, dtype=float).round(digits)
    return array.item() if array.ndim == 0 else array.tolist()


def dump_model(model_name: str) -> dict:
    """Extract the reference dictionary for one model."""
    path = mjcf_path(model_name)
    model = mujoco.MjModel.from_xml_path(str(path))

    joints = []
    for i in range(model.njnt):
        jtype = _JOINT_TYPES.get(mujoco.mjtJoint(model.jnt_type[i]), "unknown")
        entry = {
            "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
            "type": jtype,
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i]),
            "axis": _round(model.jnt_axis[i]),
            "pos": _round(model.jnt_pos[i]),
            "limited": bool(model.jnt_limited[i]),
            "range": _round(model.jnt_range[i]),
            "qpos_adr": int(model.jnt_qposadr[i]),
            "dof_adr": int(model.jnt_dofadr[i]),
        }
        # Free joints have 6 dofs; per-dof properties are read at the first one.
        dof = int(model.jnt_dofadr[i])
        entry.update(
            armature=_round(model.dof_armature[dof]),
            damping=_round(model.dof_damping[dof]),
            frictionloss=_round(model.dof_frictionloss[dof]),
        )
        joints.append(entry)

    actuators = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])),
            "ctrl_range": _round(model.actuator_ctrlrange[i]),
            "force_range": _round(model.actuator_forcerange[i]),
            "gain_prm": _round(model.actuator_gainprm[i][:3]),
            "bias_prm": _round(model.actuator_biasprm[i][:3]),
        }
        for i in range(model.nu)
    ]

    bodies = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, i),
            "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[i])),
            "mass": _round(model.body_mass[i]),
            "ipos": _round(model.body_ipos[i]),
            "iquat": _round(model.body_iquat[i]),
            "diag_inertia": _round(model.body_inertia[i]),
        }
        for i in range(model.nbody)
    ]

    sites = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, i),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[i])),
            "pos": _round(model.site_pos[i]),
            "quat": _round(model.site_quat[i]),
        }
        for i in range(model.nsite)
    ]

    # Contact parameters have no USD equivalent and must be re-authored Isaac-Lab
    # side; capture them so the re-authoring is checkable rather than remembered.
    geoms = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])),
            "type": int(model.geom_type[i]),
            "group": int(model.geom_group[i]),
            "condim": int(model.geom_condim[i]),
            "priority": int(model.geom_priority[i]),
            "friction": _round(model.geom_friction[i]),
            "solref": _round(model.geom_solref[i]),
            "solimp": _round(model.geom_solimp[i]),
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
        }
        for i in range(model.ngeom)
    ]

    servo_joints = [j["name"] for j in joints if j["type"] != "free" and not j["name"].startswith("passive_")]
    passive_joints = [j["name"] for j in joints if j["name"].startswith("passive_")]

    return {
        "model": model_name,
        "mjcf": path.name,
        "mujoco_version": mujoco.__version__,
        "counts": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
        },
        "total_mass": _round(model.body_mass.sum()),
        # The canonical 14-servo layout: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
        "servo_joints": servo_joints,
        # Unactuated joints (wheels, backlash hinges, jaw linkage) — always `passive_*`.
        "passive_joints": passive_joints,
        "joints": joints,
        "actuators": actuators,
        "bodies": bodies,
        "sites": sites,
        "geoms": geoms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ROBOT_MJCF), help="Dump one model (default: all).")
    parser.add_argument("--out-dir", type=Path, default=_REFERENCE_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in [args.model] if args.model else sorted(ROBOT_MJCF):
        reference = dump_model(model_name)
        out = args.out_dir / f"{model_name}.json"
        out.write_text(json.dumps(reference, indent=2) + "\n")
        counts = reference["counts"]
        print(
            f"{model_name:24s} -> {out.name:34s} "
            f"servos={len(reference['servo_joints']):2d} passive={len(reference['passive_joints']):2d} "
            f"bodies={counts['nbody']:3d} geoms={counts['ngeom']:3d} "
            f"mass={reference['total_mass'] * 1000:.1f} g"
        )


if __name__ == "__main__":
    main()
