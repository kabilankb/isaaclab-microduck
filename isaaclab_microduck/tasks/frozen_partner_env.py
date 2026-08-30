"""`ManagerBasedRLEnv` that drives a second robot with a frozen policy.

Shared by every two-duck task (BallRally, RunParallel). Duck A is the learner and
duck B ("partner") replays an exported policy, which keeps the env SINGLE-AGENT
from the manager's point of view and leaves the 61D / 14-D contract untouched. A
single net driving both ducks would double the observation and make the learned
policy undeployable in the runtime, permanently.

Why `step()` and not a manager term: the partner is not part of the agent's action
space, so an action term would steal action dimensions and break the 14-D contract,
and an event fires after the physics substeps rather than before.

The partner policy MUST be exported TorchScript (`exported/policy.pt` from play.py
or scripts/export.py) with the observation normalizer baked in. Never a raw
checkpoint: an unnormalized partner behaves like a different robot, and the bug is
invisible in sim because in-sim play applies the normalizer anyway.
"""

from __future__ import annotations

import os

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_microduck.utils.arrays import as_torch

#: Actor observation width, fixed across the whole Microduck policy family.
OBS_DIM = 61

#: Width of the command block: twist(3) + head_pose(4) + body_pose(6).
COMMAND_DIM = 13


class FrozenPartnerEnv(ManagerBasedRLEnv):
    """Env whose ``partner`` articulation is driven by a frozen policy.

    Subclasses set :attr:`partner_asset_name` if the scene calls it something else.
    The cfg must carry ``partner_policy_path``.
    """

    partner_asset_name: str = "partner"

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._partner_policy = self._load_partner(cfg.partner_policy_path)
        self._partner_actions = torch.zeros(self.num_envs, 14, device=self.device)

    def _load_partner(self, path: str):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Partner policy not found: {path!r}. Train the source task, run play.py "
                "once to produce 'exported/policy.pt', then point the cfg's "
                "partner_policy_path at it."
            )
        policy = torch.jit.load(path, map_location=self.device).eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        return policy

    def _partner_command(self) -> torch.Tensor:
        """The partner's 13D command block. Zero = idle; override to drive it.

        RunParallel overrides this to feed the pacer a forward twist -- a partner
        fed all zeros stands still, which looks like "the policy ignores the
        command" rather than a bug.
        """
        return torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)

    def _partner_obs(self) -> torch.Tensor:
        """The partner's own 61D view, in the shared contract's exact order:

        base_ang_vel(3) projected_gravity(3) joint_pos(14) joint_vel(14)
        actions(14) twist(3) head_pose(4) body_pose(6)
        """
        robot = self.scene[self.partner_asset_name]
        quat = as_torch(robot.data.root_link_quat_w)
        gravity = quat_apply_inverse(quat, as_torch(robot.data.GRAVITY_VEC_W))
        ang_vel = quat_apply_inverse(quat, as_torch(robot.data.root_ang_vel_w))
        joint_pos = as_torch(robot.data.joint_pos) - as_torch(robot.data.default_joint_pos)
        joint_vel = as_torch(robot.data.joint_vel)
        return torch.cat(
            [ang_vel, gravity, joint_pos, joint_vel, self._partner_actions, self._partner_command()],
            dim=1,
        )

    def _drive_partner(self) -> None:
        obs = self._partner_obs()
        if obs.shape[1] != OBS_DIM:  # pragma: no cover - guards a silent contract break
            raise RuntimeError(f"partner obs is {obs.shape[1]}D, expected {OBS_DIM}D")
        with torch.no_grad():
            self._partner_actions = self._partner_policy(obs).clone()

        robot = self.scene[self.partner_asset_name]
        scale = self.cfg.actions.joint_pos.scale
        robot.set_joint_position_target(
            as_torch(robot.data.default_joint_pos) + scale * self._partner_actions
        )

    def step(self, action: torch.Tensor):
        # Before the physics substeps, so the partner's targets are held across
        # decimation exactly like the learner's.
        self._drive_partner()
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._partner_actions[env_ids] = 0.0
