"""BallRally env: duck B replays a frozen BallKick policy.

All the partner-driving machinery lives in `tasks/frozen_partner_env.py`, shared
with RunParallel. BallRally needs no override: its partner is a standing-kick
policy driven with the zero (idle) command block, which is what it was trained
against.

WHAT THE PARTNER ACTUALLY IS: a ball-blind standing-kick policy that swings when a
ball arrives at its right foot. It cannot chase, aim, or trap. Treat it as a wall
that sometimes kicks back, not a second player.
"""

from __future__ import annotations

from isaaclab_microduck.tasks.frozen_partner_env import OBS_DIM, FrozenPartnerEnv  # noqa: F401


class MicroduckBallRallyEnv(FrozenPartnerEnv):
    """Two ducks, one ball; duck A learns, duck B replays a frozen policy."""
