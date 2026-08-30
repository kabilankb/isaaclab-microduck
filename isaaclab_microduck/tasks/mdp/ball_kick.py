"""BallKick MDP terms: kick a 70 mm / 15 g ball forward with the right foot.

Ported from `mjlab_microduck.tasks.mdp`. The defining constraint of this task is
that **the actor is ball-blind**: ball position and velocity are critic-only
observations, so the 61D actor contract is untouched and the policy must learn a
swing that works across the ball's placement spread rather than homing on it.
That is also why the placement noise on reset is doing real work.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab_microduck.utils.arrays import as_torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

# Deferred: resolving `Articulation` / `RigidObject` eagerly forces Isaac Lab's
# lazy loader to import simulation_context -> scene_data_provider -> pxr (pip's
# USD) at CFG-IMPORT time, which happens BEFORE Kit starts. Kit then loads its
# own USD copy and aborts in libusd_tf.so static init. These names are used only
# in annotations, and PEP 526 never evaluates local annotations at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _kick_dir(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env unit kick direction in world XY, latched at reset."""
    if not hasattr(env, "_ball_kick_dir"):
        env._ball_kick_dir = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir[:, 0] = 1.0
    return env._ball_kick_dir


def _ball_spawn(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env ball spawn XY latched at reset, plus a "has ever spawned" mask."""
    if not hasattr(env, "_ball_spawn_pos"):
        env._ball_spawn_pos = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._ball_spawn_pos, env._ball_kick_valid


def reset_ball_in_front_of_foot(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    offset: tuple[float, float] = (0.09, -0.042),
    noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place the ball just ahead of the right foot and latch the kick direction.

    `offset` is the ball centre in the robot's YAW frame: at HOME the right foot
    is centred at (0, -0.042) with the toe tip near x = 0.034, so (0.09, -0.042)
    puts a 35 mm ball about a centimetre in front of the toe.

    `noise_xy` (uniform +/- per axis) is the placement randomization, and it is
    the point of the task rather than a detail: the policy cannot see the ball,
    so this is what forces a swing robust to real-world placement error.

    MUST be registered AFTER the robot reset events — events run in declaration
    order, and this reads the robot's final root pose.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[asset_cfg.name]

    root_pos = as_torch(robot.data.root_link_pos_w)[env_ids]
    yaw_only = yaw_quat(as_torch(robot.data.root_link_quat_w)[env_ids])
    # Rotate the body-frame offset into the world by the robot's yaw.
    cos_y = yaw_only[:, 0] ** 2 * 2.0 - 1.0 + 2.0 * yaw_only[:, 3] ** 2 * 0.0
    yaw = 2.0 * torch.atan2(yaw_only[:, 3], yaw_only[:, 0])
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    count = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(count, 1)
    off += (torch.rand(count, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(count, 7, device=env.device)
    pose[:, 0] = root_pos[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root_pos[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = as_torch(env.scene.env_origins)[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0  # identity quaternion (w, x, y, z)

    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(count, 6, device=env.device), env_ids)

    kick_dir = _kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y

    # Latch the spawn position so `kick_success` can measure how far the ball
    # actually travelled. `_ball_kick_valid` keeps the very first reset -- which
    # happens before any ball has been placed -- out of the metric.
    spawn, valid = _ball_spawn(env)
    spawn[env_ids] = pose[:, :2]
    valid[env_ids] = True


def ball_forward_velocity(
    env: "ManagerBasedRLEnv",
    max_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball speed along the latched kick direction, clamped to [0, max_speed].

    Dense and linear in speed: every extra bit of forward ball speed pays more on
    every step the ball keeps rolling, so ordinary exploration bootstraps the kick
    with no peak-detection machinery. Backward or lateral ball motion earns zero
    rather than a penalty — a mis-hit must not scare the policy away from
    touching the ball at all.

    With `max_speed` set to a TARGET rather than a large cap, pair this with
    `ball_speed_overshoot_penalty`. Saturating the reward alone does NOT remove
    "harder is better": a harder kick keeps the ball at or above the cap for more
    steps, so the rolling-time integral still grows with strike speed. The
    overshoot penalty is what makes the target the actual optimum.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    vel_xy = as_torch(ball.data.root_link_lin_vel_w)[:, :2]
    forward = (vel_xy * _kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(forward, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: "ManagerBasedRLEnv",
    target_speed: float = 1.0,
    max_penalty: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Forward ball speed in excess of `target_speed` (linear, >= 0).

    Below the target this is zero — the capped reward supplies the upward
    gradient. Keep this term's |weight| BELOW the capped reward's weight so the
    combined landscape peaks at the target with a gentler slope on the overshoot
    side: erring slightly hard must stay cheaper than not kicking at all.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    vel_xy = as_torch(ball.data.root_link_lin_vel_w)[:, :2]
    forward = torch.nan_to_num((vel_xy * _kick_dir(env)).sum(dim=1), nan=0.0)
    return torch.clamp(forward - target_speed, min=0.0, max=max_penalty)


def single_foot_grounded_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """1 while the sensed (SUPPORT) foot is loaded, else 0.

    Anti-hop: swinging the kicking leg is free, but lifting the support foot
    costs this reward every step it is airborne.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = as_torch(contact_sensor.data.net_forces_w_history)[:, 0, sensor_cfg.body_ids]
    return (torch.norm(forces, dim=-1) > threshold).float().flatten(1).amax(dim=1)


def pose_target_match(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian pose match against HOME, meaned over the selected joints.

    A SINGLE fixed target held from t=0 to the end of the episode — no waypoints
    and no episode-progress interpolation, because a policy camps at waypoints
    and the path is what RL is supposed to discover.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    current = as_torch(asset.data.joint_pos)[:, joint_ids]
    target = as_torch(asset.data.default_joint_pos)[:, joint_ids]
    return torch.exp(-(((current - target) / std) ** 2)).mean(dim=-1)


def height_target_gaussian(
    env: "ManagerBasedRLEnv",
    target_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian on trunk height against a single fixed target.

    `target_height` is MEASURED in this simulator, never carried across stacks or
    model revisions: a 5 mm error once turned the goal into an impossible target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = torch.nan_to_num(
        as_torch(asset.data.root_link_pos_w)[:, 2] - as_torch(env.scene.env_origins)[:, 2], nan=0.0
    )
    return torch.exp(-(((height - target_height) / std) ** 2))


##
# Critic-only observations. NEVER add these to the actor group: the whole task is
# defined by the actor being ball-blind.
##


def ball_pos_in_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position relative to the robot base, in the base frame."""
    ball: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    delta = as_torch(ball.data.root_link_pos_w) - as_torch(robot.data.root_link_pos_w)
    return quat_apply_inverse(as_torch(robot.data.root_link_quat_w), delta)


def ball_vel_in_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball linear velocity in the robot base frame."""
    ball: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(as_torch(robot.data.root_link_quat_w), as_torch(ball.data.root_link_lin_vel_w))


def zero_command_padding(env: "ManagerBasedRLEnv", dim: int) -> torch.Tensor:
    """A constant zero block, holding an unused command slot in the layout.

    The 61D contract is shared across the whole policy family so the runtime can
    hot-swap policies. An env that does not drive a slot pads it rather than
    deleting it — deleting would shift every later observation.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def kick_success(
    env: "ManagerBasedRLEnv",
    env_ids,
    min_distance: float = 0.30,
    max_tilt_deg: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> float:
    """Did the episode end with the ball kicked forward and the robot still up?

    THIS is the task's success criterion. `Metrics/success_rate` does NOT measure
    kicking: it comes from Isaac Lab's `UniformVelocityCommand` and asks whether
    the robot tracked its twist command, which this task pins to +/-0.01 purely to
    keep the 61D obs contract alive. It therefore reads ~0 no matter how well the
    ball is struck, and reading it as a kick score is a trap.

    Two conditions, both required:
      * the ball travelled at least `min_distance` ALONG the latched kick
        direction -- projected, so sideways scatter and backward knocks do not
        count; and
      * the trunk is within `max_tilt_deg` of vertical at episode end, so a kick
        that ends in a fall is not a success. This is deliberately far tighter
        than the 70 deg `fell_over` termination, which leaves a wide band of
        "leaning but technically alive" that a scoring metric should not pass.

    Also logs `Metrics/kick_distance` -- the mean projected travel in metres.
    That one is threshold-free, so `min_distance` can be CALIBRATED from a real
    run instead of trusted: watch the distance distribution first, then set the
    threshold. Treat the default 0.30 m as a starting guess, not a measurement.

    Runs as a CURRICULUM term because that is the only per-reset hook that fires
    BEFORE the reset events re-place the ball (`_reset_idx` calls
    `curriculum_manager.compute()` at line 369, `event_manager.apply(mode=
    "reset")` at 375), so the ball is still where the episode left it. Values are
    written straight to `env.extras["log"]` under `Metrics/`, mirroring how Isaac
    Lab's own velocity command publishes its success rate.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    spawn, valid = _ball_spawn(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    env_ids = env_ids[valid[env_ids]]
    if len(env_ids) == 0:
        return 0.0

    # Forward travel: displacement projected on the kick direction latched at reset.
    travel = as_torch(ball.data.root_link_pos_w)[env_ids, :2] - spawn[env_ids]
    forward = (travel * _kick_dir(env)[env_ids]).sum(dim=1)
    forward = torch.nan_to_num(forward, nan=0.0)

    # Still standing: |projected gravity xy| == sin(tilt).
    quat_w = as_torch(robot.data.root_link_quat_w)[env_ids]
    gravity_b = quat_apply_inverse(quat_w, as_torch(robot.data.GRAVITY_VEC_W)[env_ids])
    tilt_sin = torch.linalg.norm(gravity_b[:, :2], dim=1)
    standing = tilt_sin < math.sin(math.radians(max_tilt_deg))

    success = ((forward >= min_distance) & standing).float()

    log = env.extras.setdefault("log", {})
    log["Metrics/kick_success_rate"] = success.mean().item()
    log["Metrics/kick_distance"] = forward.mean().item()
    return success.mean().item()
