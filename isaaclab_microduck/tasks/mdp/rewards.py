"""Reward terms for the Microduck tasks.

Ported from the mjlab velocity recipe, kernels preserved exactly — these weights
and standard deviations are the product of many runs, so the shapes matter.

**Sign convention.** The mjlab stack carries two styles: mjlab-base costs return
>= 0 and take a negative weight, while some microduck functions self-negate
(return <= 0) and take a POSITIVE weight. A negative weight on a self-negating
penalty double-negates into a reward for the violation, and the policy farms it.
This port removes the footgun by unifying on **one** convention:

    every penalty here returns >= 0 and is given a NEGATIVE weight.

The infallible check is unchanged: on every run, every `Episode_Reward/<penalty>`
in the logs must be <= 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_microduck.utils.arrays import as_torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names_values

# Deferred: resolving `Articulation` / `RigidObject` eagerly forces Isaac Lab's
# lazy loader to import simulation_context -> scene_data_provider -> pxr (pip's
# USD) at CFG-IMPORT time, which happens BEFORE Kit starts. Kit then loads its
# own USD copy and aborts in libusd_tf.so static init. These names are used only
# in annotations, and PEP 526 never evaluates local annotations at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task tracking.
##


def track_linear_velocity(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the commanded base linear velocity; commanded z is assumed zero.

    Kernel is `exp(-err/std**2)` with err a SUM OF SQUARES (xy error plus a z
    penalty), matching mjlab. Isaac Lab's own `track_lin_vel_xy_exp` omits the z
    term and squares differently, which is why this is ported rather than reused.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = as_torch(asset.data.root_link_lin_vel_b)
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw rate; commanded roll/pitch rates are assumed zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = as_torch(asset.data.root_link_ang_vel_b)
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


def upright(
    env: "ManagerBasedRLEnv",
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the trunk upright: `exp(-|projected gravity xy|^2 / std^2)`.

    Deliberately strong on Microduck (weight 2.0, std^2 = 0.05). At the older
    1.0 / 0.1 setting a 4 deg steady forward lean cost ~0.05/step — effectively
    free — and the measured gait walked with a persistent lean whose falls were
    two thirds forward. At 2.0 / 0.05 that lean costs ~0.19/step: enough to hold
    the trunk level in steady gait while transient lean stays affordable.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        body_quat_w = as_torch(asset.data.body_link_quat_w)[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        body_quat_w = as_torch(asset.data.root_link_quat_w)
    projected_gravity_b = quat_apply_inverse(body_quat_w, as_torch(asset.data.GRAVITY_VEC_W))
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-xy_squared / std**2)


class variable_posture(ManagerTermBase):
    """Deviation from HOME with a SPEED-DEPENDENT per-joint tolerance.

    `exp(-mean(err^2 / std^2))`, with std selected per joint from three speed
    regimes. Smaller std is stricter. Standing wants a tight pose; walking needs
    room to swing the legs, so a single tolerance either freezes the gait or lets
    the robot slouch while standing.

    On Microduck this term covers the LEG joints only. The head is command-driven
    (`head_pose_tracking`); if it were in both, this term would pull it to HOME
    while the command pulled it elsewhere, and since this reward dominates once
    the tracking gradient dies at large commands, the policy converges to
    ignoring the head command.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = as_torch(asset.data.default_joint_pos)

        joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        self.joint_ids = joint_ids

        def _resolve(key: str) -> torch.Tensor:
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            return torch.tensor(values, device=env.device, dtype=torch.float32)

        self.std_standing = _resolve("std_standing")
        self.std_walking = _resolve("std_walking")
        self.std_running = _resolve("std_running")

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        std_standing: dict,
        std_walking: dict,
        std_running: dict,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # resolved once in __init__
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = (speed < walking_threshold).float().unsqueeze(1)
        walking = ((speed >= walking_threshold) & (speed < running_threshold)).float().unsqueeze(1)
        running = (speed >= running_threshold).float().unsqueeze(1)
        std = self.std_standing * standing + self.std_walking * walking + self.std_running * running

        error = as_torch(asset.data.joint_pos)[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids]
        return torch.exp(-torch.mean(torch.square(error) / std**2, dim=1))


##
# Head pose (the 4-slot head command block).
##


class head_pose_tracking(ManagerTermBase):
    """Per-joint Gaussian on the commanded neck/head deltas, MEANED over 4 joints.

    Mean rather than sum-of-squares so partial tracking pays partially: under SOS
    one large joint error kills the whole term and the gradient with it.

    `std` is the per-joint tolerance (err = std scores 1/e). Keep it on the order
    of the command range so the gradient survives curriculum widening. **Do not
    tighten this to fix head droop** — a run that tried `fine_std=0.1` stopped
    walking entirely by iteration 300 (air time 1.01 -> 0.02, peak foot height
    15 mm -> 2 mm). A head that is 38% of body mass MUST oscillate while
    stepping, so an instantaneous tight tolerance is an unescapable tax on
    walking, and standing still scored higher. Price the escapable DC component
    with `head_pose_bias_penalty` instead.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        # Resolve by NAME: Isaac Lab orders the articulation its own way, and on
        # backlash/roller models passive joints interleave with the servos.
        from isaaclab_microduck.robot.microduck_cfg import HEAD_JOINT_NAMES

        self.joint_ids = [asset.joint_names.index(name) for name in HEAD_JOINT_NAMES]
        self.default_joint_pos = as_torch(asset.data.default_joint_pos)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "head_pose",
        std: float = 0.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)  # (N, 4)
        measured = as_torch(asset.data.joint_pos)[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids]
        error = measured - command
        return torch.mean(torch.exp(-torch.square(error / std)), dim=1)


class head_pose_bias_penalty(ManagerTermBase):
    """Cost on the TIME-AVERAGED (DC) head tracking error: `mean(|EMA(err)|)`.

    Companion to `head_pose_tracking`, which scores the instantaneous error.
    Walking unavoidably shakes a head that is 38% of the robot's mass, so pricing
    the instantaneous error hard is a permanent tax no policy can escape. The
    steady-state droop IS escapable — the policy can bias its neck command up to
    cancel gravity sag — so averaging over `tau_s` lets the oscillation cancel and
    charges only the bias. At the optimum this costs a walking policy nothing.

    L1 rather than Gaussian on purpose: the gradient stays constant at large bias,
    where a tight Gaussian would be flat and dead.

    Returns >= 0; give it a NEGATIVE weight (see the module docstring).
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        from isaaclab_microduck.robot.microduck_cfg import HEAD_JOINT_NAMES

        asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.joint_ids = [asset.joint_names.index(name) for name in HEAD_JOINT_NAMES]
        self.default_joint_pos = as_torch(asset.data.default_joint_pos)
        self._ema = torch.zeros(env.num_envs, len(self.joint_ids), device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._ema.zero_()
        else:
            self._ema[env_ids] = 0.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "head_pose",
        tau_s: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        measured = as_torch(asset.data.joint_pos)[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids]
        error = measured - command

        alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
        self._ema = (1.0 - alpha) * self._ema + alpha * error
        return torch.mean(torch.abs(self._ema), dim=1)


def body_pose_tracking_6d(
    env: "ManagerBasedRLEnv",
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.05,
    z_std: float = 0.02,
    angle_std: float = 0.2618,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the 6D body-pose command [x, y, z, roll, pitch, yaw] from nominal.

    Carried at weight 0 in the velocity task purely to keep the observation slot
    and its input neurons alive; the standup family raises the weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 6)

    height = as_torch(asset.data.root_link_pos_w)[:, 2] - as_torch(env.scene.env_origins)[:, 2]
    z_error = torch.square(height - (nominal_height + command[:, 2])) / z_std**2

    projected_gravity_b = quat_apply_inverse(as_torch(asset.data.root_link_quat_w), as_torch(asset.data.GRAVITY_VEC_W))
    angle_error = torch.sum(torch.square(projected_gravity_b[:, :2] - command[:, 3:5]), dim=1) / angle_std**2

    # xy is a velocity-frame quantity in the walking task; the slot is kept alive
    # but not scored here (the standup family scores it against a spawn anchor).
    del xy_std
    return torch.exp(-(z_error + angle_error))


##
# Regularizers. All return >= 0 and take a NEGATIVE weight.
##


def body_angular_velocity_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum-of-squares trunk angular velocity.

    A MOTION-BLOCKER, not a smoothness term: it penalizes what a dynamic motion
    physically requires. Keep it LOW here (and near zero for dynamic tasks).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        ang_vel = as_torch(asset.data.body_link_ang_vel_w)[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        ang_vel = as_torch(asset.data.root_link_ang_vel_w)
    return torch.sum(torch.square(ang_vel), dim=1)


def feet_slip(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal foot speed WHILE the foot is loaded.

    Only charged when a command is actually being given: at zero command the
    robot should be still anyway, and charging slip there just taxes standing.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = as_torch(contact_sensor.data.net_forces_w_history)[:, 0, sensor_cfg.body_ids]
    in_contact = torch.norm(forces, dim=-1) > 1.0

    foot_vel = as_torch(asset.data.body_link_lin_vel_w)[:, asset_cfg.body_ids, :2]
    speed = torch.norm(foot_vel, dim=-1)

    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :3], dim=1) > command_threshold).float()
    return torch.sum(speed * in_contact.float(), dim=1) * commanded


def self_collision_cost(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Count of self-contacts above a force threshold."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = as_torch(contact_sensor.data.force_matrix_w)
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    magnitude = torch.norm(forces, dim=-1)
    return (magnitude > threshold).float().flatten(1).sum(dim=1)


def feet_clearance(
    env: "ManagerBasedRLEnv",
    target_height: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cost on how far a SWINGING foot's peak height sits from `target_height`.

    Foot height is taken from the body z above the env origin. That is exact on
    flat terrain, which is what this task uses; the rough-terrain variant needs a
    per-foot terrain-height ray sensor, since "height above the world" and
    "height above the ground under the foot" stop agreeing there.

    Only charged while a command is being given, and only for feet that are off
    the ground — otherwise it would tax standing still.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = as_torch(contact_sensor.data.net_forces_w_history)[:, 0, sensor_cfg.body_ids]
    airborne = (torch.norm(forces, dim=-1) < 1.0).float()

    foot_z = as_torch(asset.data.body_link_pos_w)[:, asset_cfg.body_ids, 2] - as_torch(env.scene.env_origins)[:, 2].unsqueeze(1)
    error = torch.square(foot_z - target_height)

    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :3], dim=1) > command_threshold).float()
    return torch.sum(error * airborne, dim=1) * commanded


def feet_swing_height(
    env: "ManagerBasedRLEnv",
    target_height: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cost on a swinging foot staying BELOW `target_height`.

    One-sided, unlike `feet_clearance`: lifting higher than the target is not
    charged here. Together they ask for a definite step without pricing an
    occasional high clear.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = as_torch(contact_sensor.data.net_forces_w_history)[:, 0, sensor_cfg.body_ids]
    airborne = (torch.norm(forces, dim=-1) < 1.0).float()

    foot_z = as_torch(asset.data.body_link_pos_w)[:, asset_cfg.body_ids, 2] - as_torch(env.scene.env_origins)[:, 2].unsqueeze(1)
    shortfall = torch.clamp(target_height - foot_z, min=0.0)

    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :3], dim=1) > command_threshold).float()
    return torch.sum(shortfall * airborne, dim=1) * commanded


def forward_speed_linear(
    env: "ManagerBasedRLEnv",
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame forward speed toward the command, LINEAR and capped. Dense.

    Exists because the Gaussian `track_linear_velocity` has no gradient far from
    its target. With std = sqrt(0.1), a robot moving 0.29 m/s under a 1.2 m/s
    command scores exp(-8.3) ~ 0.00025: it is paid nothing, and told nothing about
    which direction is better. Two full 6000-iteration runs stalled in exactly that
    flat region, at ~20% of commanded speed.

    Linear in speed, so every extra bit of progress pays more on every step, from a
    standing start. This is the same shape as `ball_forward_velocity`, which is
    what bootstrapped the kick from nothing -- Gaussians are for precision once
    you are close, not for finding the behaviour in the first place.

    Capped at the commanded speed so overshoot pays nothing extra, and clamped at 0
    so moving backwards under a forward command earns zero rather than a penalty --
    a mis-step must not scare the policy away from moving at all.

    Normalised by |command| so the term's scale does not change as the speed
    curriculum widens; at zero command it returns 0 (nothing to track).
    """
    asset: "Articulation" = env.scene[asset_cfg.name]
    speed = as_torch(asset.data.root_lin_vel_b)[:, 0]
    cmd = env.command_manager.get_command(command_name)[:, 0]

    # Accumulate for the EPISODE-MEAN speed metric as a side effect. This lives
    # here, rather than in a dedicated zero-weight term, because Isaac Lab's
    # RewardManager skips terms with weight 0 outright
    # (`reward_manager.py`: "if term_cfg.weight == 0.0: continue"), so such a term
    # is never called. This function already has exactly the two quantities the
    # metric needs and runs every step.
    from .observations import _speed_accum

    _ssum, _csum, _steps = _speed_accum(env)
    _ssum += torch.nan_to_num(speed, nan=0.0)
    _csum += cmd
    _steps += 1.0

    moving = cmd.abs() > 0.05
    progress = torch.where(cmd >= 0, speed, -speed)  # signed toward the command
    capped = torch.clamp(progress, min=0.0) / cmd.abs().clamp_min(1e-3)
    return torch.where(moving, torch.clamp(capped, max=1.0), torch.zeros_like(capped))


def yaw_rate_error_l1(
    env: "ManagerBasedRLEnv",
    command_name: str = "twist",
    max_error: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|yaw rate - commanded| in rad/s, clipped. Cost >= 0, so NEGATIVE weight.

    DENSE, for the same reason `forward_speed_linear` is. The Gaussian
    `track_angular_velocity` uses std = sqrt(0.5), so a robot yawing at 1.19 rad/s
    under a zero command scores exp(-2.83) ~ 0.06 -- it is paid almost nothing and,
    worse, told almost nothing about which direction is better. A measured policy sat
    exactly there: it walked at the commanded 0.4 m/s but arced instead of holding a
    line (straightness 0.74, mean |yaw rate| 1.19 rad/s against a commanded 0.0).

    Linear in the error, so every bit of heading correction pays from anywhere in the
    range. Clipped at `max_error` so a tumble cannot dominate the episode -- a 25 cm
    robot tumbles at 3.5-5.5 rad/s naturally, and an unclipped term would price that
    far above anything the policy can act on.
    """
    asset: "Articulation" = env.scene[asset_cfg.name]
    yaw_rate = as_torch(asset.data.root_ang_vel_b)[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = torch.abs(torch.nan_to_num(yaw_rate, nan=0.0) - cmd)
    return torch.clamp(err, max=max_error)
