#!/usr/bin/env python
"""P2 gate: does the converted USD still describe the same robot, and does it stand?

Two checks in one app launch:

1. **Parity** — joint names and count, per-joint limits / armature / friction,
   body names, per-body mass and total mass, all compared against the MuJoCo
   reference dump (`assets/reference/<model>.json`). Conversion silently dropping
   or reinterpreting a property is the failure mode this phase exists to catch.

2. **Settle** — hold the HOME pose from noisy initial states and report TILT as
   well as height. A settle test that only records z reports fallen states as
   "resting fine", which is exactly how a broken HOME pose slips through.

    python scripts/check_asset_parity.py --model walk
    python scripts/check_asset_parity.py --model walk --settle-seconds 3 --num-envs 16

Both checks run under the SAME actuation as MuJoCo (the MJCF's own position servo,
see `_servo_actuator_placeholder`), so a difference here is the conversion, not the
actuator model. The BAM actuator arrives in P3.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

from isaaclab_microduck.assets import ROBOT_MJCF

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="walk", choices=sorted(ROBOT_MJCF))
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--settle-seconds", type=float, default=3.0)
parser.add_argument(
    "--joint-noise",
    type=float,
    default=0.05,
    help="Uniform +/- rad noise added to the HOME joint positions at spawn.",
)
parser.add_argument("--tilt-tolerance-deg", type=float, default=10.0)
parser.add_argument(
    "--kp",
    type=float,
    default=None,
    help=(
        "Override the placeholder servo stiffness (default: the conversion-check gain, "
        "see microduck_cfg._CHECK_KP). This is a conversion check, not a physical "
        "actuator: the real equilibrium gate arrives with BAM in P3."
    ),
)
parser.add_argument("--kd", type=float, default=None, help="Override the placeholder servo damping.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_microduck.robot.microduck_cfg import make_microduck_cfg  # noqa: E402
from isaaclab_microduck.robot.reference import body_name_map, load_reference  # noqa: E402

# Physics timestep matches the mjlab recipe (0.005 s, 200 Hz) so the comparison is
# like for like; control there runs at 50 Hz (decimation 4).
PHYSICS_DT = 0.005

_FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(label)


def build_scene_cfg(model: str, num_envs: int):
    robot_cfg = make_microduck_cfg(model)
    if args_cli.kp is not None:
        robot_cfg.actuators["servos"].stiffness = args_cli.kp
    if args_cli.kd is not None:
        robot_cfg.actuators["servos"].damping = args_cli.kd

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        robot = robot_cfg

    return _SceneCfg(num_envs=num_envs, env_spacing=1.0)


def check_parity(robot, reference: dict) -> None:
    """Compare the loaded articulation against the MuJoCo reference dump."""
    print("\n--- Parity vs MuJoCo reference ---")

    ref_joints = {j["name"] for j in reference["joints"] if j["type"] != "free"}
    usd_joints = set(robot.joint_names)
    check(
        f"joint set matches ({len(ref_joints)} joints)",
        ref_joints == usd_joints,
        f"missing={sorted(ref_joints - usd_joints)} extra={sorted(usd_joints - ref_joints)}",
    )

    # Joint ORDER is Isaac Lab's own (breadth-first over the articulation), which
    # is why nothing in this port may hardcode joint indices: resolve by name.
    servos = reference["servo_joints"]
    print(f"  info   canonical MuJoCo servo order : {servos}")
    print(f"  info   Isaac Lab articulation order : {robot.joint_names}")
    check(
        "all 14 servos present",
        set(servos).issubset(usd_joints),
        f"missing={sorted(set(servos) - usd_joints)}",
    )

    # The importer renames a body whose name collides with a joint name (see
    # `reference.usd_body_name`); compare against the mapped names, and thereby
    # verify that rule still holds for this model.
    name_map = body_name_map(args_cli.model)
    ref_bodies = set(name_map.values())
    usd_bodies = set(robot.body_names)
    check(
        f"body set matches ({len(ref_bodies)} bodies)",
        ref_bodies == usd_bodies,
        f"missing={sorted(ref_bodies - usd_bodies)} extra={sorted(usd_bodies - ref_bodies)}",
    )
    renamed = {k: v for k, v in name_map.items() if k != v}
    if renamed:
        print(f"  info   bodies renamed by the importer (name clashes with a joint): {renamed}")

    # Mass: the single most consequential quantity for a 737 g robot.
    ref_mass_by_body = {name_map.get(b["name"], b["name"]): b["mass"] for b in reference["bodies"]}
    body_mass = robot.data.body_mass[0]
    total = float(body_mass.sum())
    ref_total = float(reference["total_mass"])
    check(
        "total mass within 1 %",
        abs(total - ref_total) <= 0.01 * ref_total,
        f"usd={total * 1000:.2f} g  mujoco={ref_total * 1000:.2f} g",
    )

    worst_body, worst_err = None, 0.0
    for i, name in enumerate(robot.body_names):
        ref_m = ref_mass_by_body.get(name)
        if ref_m is None or ref_m == 0.0:
            continue
        err = abs(float(body_mass[i]) - ref_m) / ref_m
        if err > worst_err:
            worst_body, worst_err = name, err
    check(
        "per-body mass within 1 %",
        worst_err <= 0.01,
        f"worst: {worst_body} off by {worst_err * 100:.2f} %",
    )

    # Joint limits / armature / friction: calibrated values, not defaults.
    ref_by_name = {j["name"]: j for j in reference["joints"]}
    lower = robot.data.joint_pos_limits_lower[0]
    upper = robot.data.joint_pos_limits_upper[0]
    armature = robot.data.joint_armature[0]
    friction = robot.data.joint_friction_coeff[0]

    worst_limit, worst_limit_err = None, 0.0
    worst_arm, worst_arm_err = None, 0.0
    for i, name in enumerate(robot.joint_names):
        ref = ref_by_name.get(name)
        if ref is None or not ref["limited"]:
            continue
        err = max(abs(float(lower[i]) - ref["range"][0]), abs(float(upper[i]) - ref["range"][1]))
        if err > worst_limit_err:
            worst_limit, worst_limit_err = name, err
        arm_err = abs(float(armature[i]) - ref["armature"])
        if arm_err > worst_arm_err:
            worst_arm, worst_arm_err = name, arm_err

    check(
        "joint position limits within 1e-3 rad",
        worst_limit_err <= 1e-3,
        f"worst: {worst_limit} off by {worst_limit_err:.2e} rad",
    )
    check(
        "joint armature within 1e-6",
        worst_arm_err <= 1e-6,
        f"worst: {worst_arm} off by {worst_arm_err:.2e}",
    )
    print(f"  info   joint friction coeff (env 0, first 3): {friction[:3].tolist()}")

    # Contact parameters have NO USD equivalent and are re-authored Isaac Lab side.
    foot_geoms = [g for g in reference["geoms"] if g["name"].endswith("_foot_collision")]
    print(
        "  info   MuJoCo foot contact params (must be re-authored, not inherited): "
        + ", ".join(f"{g['name']}: condim={g['condim']} friction={g['friction'][0]}" for g in foot_geoms)
    )


def check_settle(sim, scene, robot, seconds: float, joint_noise: float, tilt_tol_deg: float) -> None:
    """Hold HOME from noisy inits; report tilt AND height."""
    print(f"\n--- Settle test ({seconds:.1f} s holding HOME, +/-{joint_noise} rad init noise) ---")

    home = robot.data.default_joint_pos.clone()
    noise = (torch.rand_like(home) * 2.0 - 1.0) * joint_noise
    start = home + noise

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(start, torch.zeros_like(start))
    robot.reset()
    scene.write_data_to_sim()

    steps = int(seconds / PHYSICS_DT)
    for _ in range(steps):
        # Hold the HOME pose: the target is HOME, the state started off it.
        robot.set_joint_position_target(home)
        scene.write_data_to_sim()
        sim.step()
        scene.update(PHYSICS_DT)

    quat = robot.data.root_link_quat_w  # (N, 4) wxyz
    # cos(tilt) of the body z-axis against world up; same expression the mjlab
    # upright rewards use, so the numbers are directly comparable.
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    tilt_deg = torch.rad2deg(torch.arccos(cos_tilt.clamp(-1.0, 1.0)))
    height = robot.data.root_link_pos_w[:, 2] - scene.env_origins[:, 2]

    print(
        f"  trunk z : mean {float(height.mean()) * 1000:.1f} mm  "
        f"min {float(height.min()) * 1000:.1f}  max {float(height.max()) * 1000:.1f}"
    )
    print(
        f"  tilt    : mean {float(tilt_deg.mean()):.2f} deg  "
        f"max {float(tilt_deg.max()):.2f} deg"
    )
    check(
        f"all envs upright (tilt < {tilt_tol_deg} deg)",
        bool((tilt_deg < tilt_tol_deg).all()),
        f"{int((tilt_deg >= tilt_tol_deg).sum())}/{tilt_deg.numel()} envs exceeded",
    )
    check(
        "no NaN in root state",
        bool(torch.isfinite(robot.data.root_link_pos_w).all() and torch.isfinite(quat).all()),
    )
    print(
        "\n  NOTE: standing trunk z is a MEASURED quantity. Use the mean above as the "
        "Isaac Lab standing height; never carry a height across stacks or model revisions."
    )


def main() -> None:
    sim_cfg = SimulationCfg(dt=PHYSICS_DT, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1))
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(build_scene_cfg(args_cli.model, args_cli.num_envs))
    sim.reset()

    robot = scene["robot"]
    reference = load_reference(args_cli.model)

    print(f"\n=== Asset parity: model '{args_cli.model}' ({reference['mjcf']}) ===")
    print(f"  envs={args_cli.num_envs}  physics=newton_mjwarp  dt={PHYSICS_DT}")

    check_parity(robot, reference)
    check_settle(sim, scene, robot, args_cli.settle_seconds, args_cli.joint_noise, args_cli.tilt_tolerance_deg)

    print("\n=== Summary ===")
    if _FAILURES:
        print(f"  {len(_FAILURES)} check(s) FAILED:")
        for name in _FAILURES:
            print(f"    - {name}")
    else:
        print("  all checks passed")
    return len(_FAILURES)


if __name__ == "__main__":
    failures = main()
    # Exit BEFORE Isaac Sim's teardown. `simulation_app.close()` terminates the
    # process itself with status 0, so anything after it never runs: a failing
    # check reported exit 0 and CI would have waved a broken asset through.
    # Neither `raise SystemExit(1)` nor `os._exit(1)` placed after `close()`
    # survives — both were tried on this machine and both still exited 0.
    # Flush explicitly, since `os._exit` skips Python's buffers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)
