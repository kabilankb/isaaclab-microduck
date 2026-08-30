"""Command-term RUNTIME classes, kept OUT of the cfg import path.

These subclass Isaac Lab's command implementations, and importing those pulls
`Articulation` -> `simulation_context` -> `scene_data_provider` -> `pxr` (pip's
USD). Task cfgs are imported during CLI preset collection, BEFORE Kit starts, so
doing that from a cfg module put pip's USD in the process first; Kit then loaded
its own separately-built USD and aborted in `libusd_tf.so` static init
("free(): invalid pointer"), which is why `--visualizer kit` segfaulted on every
Microduck task while working on Isaac Lab's own.

The cfg classes in `commands.py` therefore reference these by STRING, exactly as
Isaac Lab's own `commands_cfg.py` does, so the import happens at env creation --
after Kit is up. Import this module from a cfg and the crash comes back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import MicroduckVelocityCommandCfg, UniformPoseCommandCfg

class MicroduckVelocityCommand(UniformVelocityCommand):
    """Twist command with an explicit turn-in-place bucket."""

    cfg: "MicroduckVelocityCommandCfg"

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)

        prob = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if prob <= 0.0 or len(env_ids) == 0:
            return

        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < prob]
        if len(turn_ids) == 0:
            return

        # Zero the linear command and force a MEANINGFUL yaw: a magnitude drawn
        # from [0.4*max, max] rather than uniform, since small yaw commands are
        # already well covered by the base sampling and do not teach spinning.
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(lo), abs(hi))
        sign = torch.where(
            torch.rand(len(turn_ids), device=self.device) < 0.5, -1.0, 1.0
        )
        magnitude = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * max_rate, max_rate)
        self.vel_command_b[turn_ids, 2] = sign * magnitude

        # These envs must actually turn: un-mark them as standing, which would
        # otherwise zero the command straight back out.
        self.is_standing_env[turn_ids] = False


class UniformPoseCommand(CommandTerm):
    """Generic N-dim uniform command, held between resamples.

    Deliberately lightweight (no metrics, no debug viz) because the Microduck
    policies carry several of these at once.
    """

    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    def __str__(self) -> str:
        return (
            f"UniformPoseCommand: {self.dim}D, ranges={self.cfg.ranges}, "
            f"zero_command_prob={self.cfg.zero_command_prob}"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids) -> None:
        count = len(env_ids)
        if count == 0:
            return
        r = torch.empty(count, device=self.device)
        for i, (low, high) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(low, high)

        # Explicit zero-command bucket — see the module docstring.
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(count, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0
