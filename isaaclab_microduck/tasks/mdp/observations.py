"""Observation terms for the Microduck tasks.

Only the terms with no Isaac Lab equivalent live here. Everything else uses
`isaaclab.envs.mdp` directly, with `SceneEntityCfg("robot", joint_names=...)` to
restrict joint terms to the 14 servos — essential on the roller and backlash
models, where passive joints interleave and the raw articulation is wider than
the action space.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab_microduck.utils.arrays import as_torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul

# Deferred: resolving `Articulation` / `RigidObject` eagerly forces Isaac Lab's
# lazy loader to import simulation_context -> scene_data_provider -> pxr (pip's
# USD) at CFG-IMPORT time, which happens BEFORE Kit starts. Kit then loads its
# own USD copy and aborts in libusd_tf.so static init. These names are used only
# in annotations, and PEP 526 never evaluates local annotations at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def _random_misalignment_quat(num_envs: int, max_angle_deg: float, device) -> torch.Tensor:
    """Per-env random-axis rotation of up to `max_angle_deg`, as a wxyz quaternion.

    ZERO-CENTERED by construction: this trains tolerance to misalignment
    MAGNITUDE. It cannot compensate a systematic mounting bias — the real board's
    steady pitch offset is a runtime calibration, corrected at the source, not
    something domain randomization can absorb.
    """
    axis = torch.randn(num_envs, 3, device=device)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-6)
    angle = torch.rand(num_envs, device=device) * math.radians(max_angle_deg)
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(1), axis * torch.sin(half).unsqueeze(1)], dim=1)


class _ImuMisalignedBase(ManagerTermBase):
    """Holds one per-env IMU misalignment quaternion, resampled on reset."""

    def __init__(self, cfg: "ObservationTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._max_angle_deg = float(cfg.params.get("max_angle_deg", 0.0))
        self._quat = _random_misalignment_quat(env.num_envs, self._max_angle_deg, env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._max_angle_deg <= 0.0:
            return
        if env_ids is None:
            self._quat = _random_misalignment_quat(self.num_envs, self._max_angle_deg, self.device)
        else:
            self._quat[env_ids] = _random_misalignment_quat(len(env_ids), self._max_angle_deg, self.device)


class projected_gravity_imu_misaligned(_ImuMisalignedBase):
    """Projected gravity as seen through a slightly misaligned IMU.

    ACTOR ONLY — the critic keeps the true value. The policy should learn to
    tolerate a mounting error it cannot observe; the value function should not be
    handicapped by it.
    """

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        max_angle_deg: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del max_angle_deg
        asset: Articulation = env.scene[asset_cfg.name]
        quat = quat_mul(as_torch(asset.data.root_link_quat_w), self._quat)
        return quat_apply_inverse(quat, as_torch(asset.data.GRAVITY_VEC_W))


class base_ang_vel_imu_misaligned(_ImuMisalignedBase):
    """Base angular velocity through the same misaligned IMU frame. Actor only."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        max_angle_deg: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del max_angle_deg
        asset: Articulation = env.scene[asset_cfg.name]
        return quat_apply_inverse(self._quat, as_torch(asset.data.root_link_ang_vel_b))


def _speed_accum(env):
    """Per-env running sum of forward speed and step count."""
    import torch

    if not hasattr(env, "_speed_sum"):
        env._speed_sum = torch.zeros(env.num_envs, device=env.device)
        env._speed_cmd_sum = torch.zeros(env.num_envs, device=env.device)
        env._speed_steps = torch.zeros(env.num_envs, device=env.device)
    return env._speed_sum, env._speed_cmd_sum, env._speed_steps


def accumulate_speed(env: "ManagerBasedRLEnv", command_name: str = "twist") -> "torch.Tensor":
    """Accumulate per-step speed. Register as a ZERO-WEIGHT reward term.

    A reward term is used only because it is the hook that runs EVERY step; the
    return is all zeros so it contributes nothing. The episode mean is finalised in
    :func:`locomotion_metrics` on reset.

    This exists because the first version of the metric sampled velocity ONLY on
    the reset hook -- that is, at the instant of termination, which for a falling
    robot is the least representative moment of the episode. It read ~0 while the
    dense reward showed real progress, and the two could not both be right.
    """
    import torch

    ssum, csum, steps = _speed_accum(env)
    asset = env.scene["robot"]
    v = torch.nan_to_num(as_torch(asset.data.root_lin_vel_b)[:, 0], nan=0.0)
    c = env.command_manager.get_command(command_name)[:, 0]
    ssum += v
    csum += c
    steps += 1.0
    return torch.zeros_like(v)


def locomotion_metrics(env: "ManagerBasedRLEnv", env_ids=None) -> float:
    """Publish EPISODE-MEAN measured motion, in m/s -- not reward.

    Reward terms were misread as evidence of locomotion for two entire runs:
    `Episode_Reward/track_linear_velocity` read ~1.2 while the robot travelled
    0.8 mm in 6 s, because a stationary upright robot still collects the height,
    pose, upright and head-tracking terms.

    Averaged over the whole episode via :func:`accumulate_speed`. Reading velocity
    at reset alone samples the moment of termination, which is dominated by falls.
    """
    import torch

    ssum, csum, steps = _speed_accum(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return 0.0

    n = steps[env_ids].clamp_min(1.0)
    mean_v = ssum[env_ids] / n
    mean_c = csum[env_ids] / n
    moving = mean_c.abs() > 0.05

    log = env.extras.setdefault("log", {})
    log["Metrics/base_speed_x"] = mean_v.mean().item()
    log["Metrics/base_speed_abs"] = mean_v.abs().mean().item()
    if moving.any():
        log["Metrics/speed_tracking_ratio"] = (
            (mean_v[moving] / mean_c[moving]).clamp(-2, 2).mean().item()
        )
    ssum[env_ids] = 0.0
    csum[env_ids] = 0.0
    steps[env_ids] = 0.0
    return mean_v.abs().mean().item()
