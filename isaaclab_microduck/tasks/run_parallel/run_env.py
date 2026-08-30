"""RunParallel env: the pacer duck is driven with a FORWARD twist command.

Unlike BallRally -- whose partner is a standing policy fed the idle command -- the
pacer here must actually run, so its 13D command block carries a non-zero forward
velocity. A locomotion policy fed all zeros stands still, which reads as "the
partner ignores the command" rather than as the bug it is.

The pacer's speed is held FIXED per episode rather than tracking the learner: a
pacer that chased the learner would make the formation reward trivially satisfiable
by standing still, since the target would come to you.
"""

from __future__ import annotations

import torch

from isaaclab_microduck.tasks.frozen_partner_env import COMMAND_DIM, FrozenPartnerEnv


class MicroduckRunParallelEnv(FrozenPartnerEnv):
    """Two ducks running side by side; duck A learns, duck B paces."""

    def _partner_command(self) -> torch.Tensor:
        cmd = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        # Slot 0 is twist lin_vel_x -- the same slot the runtime writes for walking.
        cmd[:, 0] = self.cfg.pacer_speed
        return cmd
