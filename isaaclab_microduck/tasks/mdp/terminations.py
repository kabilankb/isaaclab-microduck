"""Termination terms for the Microduck tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_microduck.utils.arrays import as_torch
from isaaclab.managers import SceneEntityCfg

# Deferred: resolving `Articulation` / `RigidObject` eagerly forces Isaac Lab's
# lazy loader to import simulation_context -> scene_data_provider -> pxr (pip's
# USD) at CFG-IMPORT time, which happens BEFORE Kit starts. Kit then loads its
# own USD copy and aborts in libusd_tf.so static init. These names are used only
# in annotations, and PEP 526 never evaluates local annotations at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def robot_state_is_nan(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate an env whose state has gone non-finite.

    A single NaN reaching the learner kills the whole run through rsl_rl's NaN
    check, taking every other environment with it. Catching it as a termination
    costs one episode instead. Checks root state and joint state; sensor-derived
    observations are handled separately by sanitizing the critic terms.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    bad = ~torch.isfinite(as_torch(asset.data.root_link_pos_w)).all(dim=1)
    bad |= ~torch.isfinite(as_torch(asset.data.root_link_quat_w)).all(dim=1)
    bad |= ~torch.isfinite(as_torch(asset.data.root_link_lin_vel_b)).all(dim=1)
    bad |= ~torch.isfinite(as_torch(asset.data.root_link_ang_vel_b)).all(dim=1)
    bad |= ~torch.isfinite(as_torch(asset.data.joint_pos)).all(dim=1)
    bad |= ~torch.isfinite(as_torch(asset.data.joint_vel)).all(dim=1)
    return bad
