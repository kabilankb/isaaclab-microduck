"""Command terms for the Microduck tasks.

Ported from `mjlab_microduck.tasks.mdp`. Together these produce the **13D command
block** every Microduck policy carries, in this exact order:

    [twist(3), head_pose(4), body_pose(6)]

An env that does not use a slot ZERO-PADS it (keep the term, sample tiny ranges)
rather than deleting it, so policies stay hot-swappable in the runtime.

Two sampling rules here exist because uniform sampling alone does not produce
them, and both were learned the hard way:

* **Zero commands must be sampled explicitly.** Uniform sampling essentially never
  produces the all-zero command, which is exactly the deployment idle state.
* **Turn-in-place needs its own bucket.** Independent uniform sampling makes
  "lin about 0, |ang| large" roughly 2% of experience, so spinning never trained.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

# commands_cfg ONLY -- it declares class_type as a lazily-resolved string and is
# free of pxr, unlike the `isaaclab.envs.mdp.commands` package root which drags in
# the runtime classes. See command_impl.py for why that matters.
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .command_impl import MicroduckVelocityCommand, UniformPoseCommand

_IMPL = "isaaclab_microduck.tasks.mdp.command_impl"

@configclass
class MicroduckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Twist command cfg. `rel_standing_envs` (inherited) is the zero-command bucket."""

    class_type: type["MicroduckVelocityCommand"] | str = f"{_IMPL}:MicroduckVelocityCommand"

    rel_turn_in_place_envs: float = 0.0
    """Fraction of envs commanded to turn in place each resample (0 disables).

    Kept non-zero from step 0: a command region that is never sampled cannot be
    learned later, and turning is a deployment requirement.
    """


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Per-dimension uniform ranges; the tuple length defines the command dim."""

    class_type: type["UniformPoseCommand"] | str = f"{_IMPL}:UniformPoseCommand"

    ranges: tuple[tuple[float, float], ...] = ()
    """(low, high) per dimension.

    Every slot keeps a small NON-ZERO range from step 0 even when its reward
    weight is 0: a command input that is never non-zero has dead weights forever,
    and a later curriculum cannot revive them.
    """

    zero_command_prob: float = 0.0
    """Probability a resample yields the exact all-zero command."""
