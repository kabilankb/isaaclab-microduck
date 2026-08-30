#!/usr/bin/env python
"""Look at the Microduck in a GUI: spawn it, hold a pose, watch it stand.

There is no Microduck TASK yet (that lands in P5), so this is the way to see the
actual robot rather than a placeholder cartpole. It spawns the converted USD on a
ground plane, holds the HOME pose, and keeps stepping until you close the window.

    # stand at HOME and hold
    DISPLAY=:1 python scripts/view_robot.py --visualizer newton

    # keyboard teleop of the joint targets (keep focus on the TERMINAL)
    DISPLAY=:1 python scripts/view_robot.py --visualizer newton --teleop

    # other models, more copies, or a drop test
    DISPLAY=:1 python scripts/view_robot.py --model rollers --num-envs 4 --visualizer newton
    DISPLAY=:1 python scripts/view_robot.py --drop 0.25 --visualizer newton

Actuation is the P2 conversion-check PD (see `microduck_cfg._CHECK_KP`), NOT the BAM
actuator — this shows that the asset is right, not how the real servos behave. The
robot holding HOME here is the same thing `check_asset_parity.py` measures
numerically (trunk z 116.4 mm on the walk model, 139.6 mm on rollers).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from isaaclab_microduck.assets import ROBOT_MJCF

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="walk", choices=sorted(m for m in ROBOT_MJCF if m != "ball"))
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--env-spacing", type=float, default=0.6)
parser.add_argument(
    "--drop",
    type=float,
    default=None,
    help="Spawn the trunk at this height (m) instead of the default, to watch it land.",
)
parser.add_argument(
    "--joint-noise",
    type=float,
    default=0.0,
    help="Uniform +/- rad noise on the initial joint positions.",
)
parser.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = run until closed).")
parser.add_argument(
    "--teleop",
    action="store_true",
    help="Drive the joint targets from the keyboard. Keys are read from the TERMINAL, "
    "not the viewer window, so keep the terminal focused (see utils/terminal_input.py).",
)
parser.add_argument("--step", type=float, default=0.02, help="Teleop step per keypress, rad (default 0.02 = 1.15 deg).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Deliberately NOT forcing headless: the whole point is the window. Pass
# --visualizer newton (light, kit-less) or --visualizer kit (full Isaac Sim UI).

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

from isaaclab_microduck.robot.microduck_cfg import HEAD_JOINT_NAMES, make_microduck_cfg  # noqa: E402
from isaaclab_microduck.utils import teleop_keys  # noqa: E402
from isaaclab_microduck.utils.terminal_input import TerminalInput  # noqa: E402

PHYSICS_DT = 0.005  # matches the mjlab recipe (200 Hz physics, 50 Hz control)

# Teleop key map lives in the package so it can be unit tested (see
# utils/teleop_keys.py); a mirrored sign error there twists the robot.
TELEOP_HELP = teleop_keys.HELP
_HEAD_KEYS = teleop_keys.HEAD_KEYS
_LEG_KEYS = teleop_keys.LEG_KEYS


def main() -> None:
    robot_cfg = make_microduck_cfg(args_cli.model)
    if args_cli.drop is not None:
        pos = robot_cfg.init_state.pos
        robot_cfg.init_state.pos = (pos[0], pos[1], args_cli.drop)

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        robot = robot_cfg

    sim = sim_utils.SimulationContext(
        SimulationCfg(dt=PHYSICS_DT, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1))
    )
    scene = InteractiveScene(_SceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
    sim.reset()

    # Frame the robot: it is 25 cm tall, so a default locomotion camera is far too wide.
    sim.set_camera_view(eye=(0.6, 0.6, 0.35), target=(0.0, 0.0, 0.1))

    robot = scene["robot"]
    home = robot.data.default_joint_pos.clone()
    start = home.clone()
    if args_cli.joint_noise:
        start += (torch.rand_like(home) * 2.0 - 1.0) * args_cli.joint_noise

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(start, torch.zeros_like(start))
    robot.reset()
    scene.write_data_to_sim()

    # flush=True throughout: stdout is block-buffered when redirected to a log,
    # so without it a backgrounded viewer looks silent (and therefore hung) even
    # while it is stepping happily.
    print(f"\n[view_robot] model '{args_cli.model}', {args_cli.num_envs} env(s), holding HOME.", flush=True)
    if args_cli.teleop:
        print("[view_robot] TELEOP on — keep this TERMINAL focused, watch the viewer window.")
        print(TELEOP_HELP, flush=True)
    else:
        print("[view_robot] close the viewer window (or Ctrl-C) to stop.\n", flush=True)

    # Resolve joint names to indices ONCE, by name. Isaac Lab orders the
    # articulation its own way (right leg first here), so an index written
    # against MuJoCo's canonical order would drive the wrong joint.
    name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    lower = robot.data.joint_pos_limits_lower[0].clone()
    upper = robot.data.joint_pos_limits_upper[0].clone()

    target = home.clone()
    selected = 0
    max_steps = int(args_cli.seconds / PHYSICS_DT) if args_cli.seconds > 0 else None
    step = 0

    def nudge(joint_name: str, delta: float) -> None:
        """Move one joint's target, clamped to its limit."""
        idx = name_to_idx.get(joint_name)
        if idx is None:  # joint absent on this model
            return
        target[:, idx] = (target[:, idx] + delta).clamp(lower[idx], upper[idx])

    def report() -> None:
        z = float((robot.data.root_link_pos_w[:, 2] - scene.env_origins[:, 2]).mean())
        quat = robot.data.root_link_quat_w
        cos_tilt = (1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1.0, 1.0)
        tilt = float(torch.rad2deg(torch.arccos(cos_tilt)).mean())
        extra = f"  [{robot.joint_names[selected]} = {float(target[0, selected]):+.3f}]" if args_cli.teleop else ""
        print(
            f"  t={step * PHYSICS_DT:6.1f}s  trunk z={z * 1000:6.1f} mm  tilt={tilt:5.2f} deg{extra}",
            flush=True,
        )

    with TerminalInput() as keyboard:
        while simulation_app.is_running():
            if args_cli.teleop:
                for key in keyboard.get_keys():
                    if key == "q":
                        print("[view_robot] quit", flush=True)
                        return
                    elif key == "h":
                        print(TELEOP_HELP, flush=True)
                    elif key == "r":
                        target = home.clone()
                        print("  reset to HOME", flush=True)
                    elif key in _HEAD_KEYS:
                        joint, sign = _HEAD_KEYS[key]
                        nudge(joint, sign * args_cli.step)
                        report()
                    elif key in _LEG_KEYS:
                        for joint, sign in _LEG_KEYS[key].items():
                            nudge(joint, sign * args_cli.step)
                        report()
                    elif key in ("[", "]"):
                        selected = (selected + (1 if key == "]" else -1)) % len(robot.joint_names)
                        print(f"  selected: {robot.joint_names[selected]}", flush=True)
                    elif key in ("up", "down"):
                        nudge(robot.joint_names[selected], (1.0 if key == "up" else -1.0) * args_cli.step)
                        report()

            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(PHYSICS_DT)
            step += 1

            if not args_cli.teleop and step % 200 == 0:  # once a simulated second
                report()

            if max_steps is not None and step >= max_steps:
                break


if __name__ == "__main__":
    main()
    simulation_app.close()
