"""Locations of the Microduck MJCF sources and their converted USD assets.

The MJCFs are VENDORED into this package (`assets/mjcf/`) so the repo stands alone.
They originate in the mjlab stack (`src/mjlab_microduck/robot/microduck/`), which
remains the upstream source of truth: regenerate from Onshape there, then sync here.

Set ``MICRODUCK_MJCF_DIR`` to an mjlab checkout to read them in place instead, which
is what you want when developing both stacks together -- it keeps one source of truth
and avoids a silently stale copy.

USD output is a build artifact — gitignored, rebuilt by `scripts/convert_assets.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Directory holding the MJCF exports. Override with ``MICRODUCK_MJCF_DIR`` to read
#: them from an mjlab checkout instead of the vendored copy.
MJCF_DIR = Path(os.environ.get("MICRODUCK_MJCF_DIR", _PACKAGE_ROOT / "assets" / "mjcf"))

#: Converted USD output root. Override with ``MICRODUCK_USD_DIR``.
USD_DIR = Path(os.environ.get("MICRODUCK_USD_DIR", _PACKAGE_ROOT / "assets" / "usd"))

#: Logical model name -> MJCF filename.
#:
#: These mirror `mjlab_microduck.robot.microduck_constants`. Model choice is not
#: cosmetic: a `-Backlash-` task variant MUST use the backlash twin of its base
#: task's model, or the backlash A/B is confounded.
ROBOT_MJCF: dict[str, str] = {
    # Minimal-collision walking model (feet only) — the velocity recipe.
    "walk": "robot_walk.xml",
    # Full-collision model — standup / sitstand / ground-pick / ball-kick.
    "allcollisions": "robot_allcollisions.xml",
    # Full collision + passive wheel hinges (`passive_*wheel`) — roller tasks.
    "rollers": "robot_allcollisions_rollers.xml",
    # Backlash twins: every servo joint gains an unactuated
    # `passive_<joint>_backlash` hinge in series (±1° play).
    "walk_backlash": "robot_walk_backlash.xml",
    "allcollisions_backlash": "robot_allcollisions_backlash.xml",
    "rollers_backlash": "robot_allcollisions_rollers_backlash.xml",
    # 70 mm / 15 g ball prop for the BallKick task.
    "ball": "ball.xml",
}


def mjcf_path(model: str) -> Path:
    """Absolute path to a model's MJCF source."""
    if model not in ROBOT_MJCF:
        raise KeyError(f"Unknown model '{model}'. Known: {sorted(ROBOT_MJCF)}")
    path = MJCF_DIR / ROBOT_MJCF[model]
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found: {path}")
    return path


def usd_path(model: str) -> Path:
    """Path the converter writes a model's USD interface file to.

    The Isaac Sim MJCF importer always lays a model out as
    ``<usd_dir>/<stem>/<stem>.usda`` plus a ``payloads/`` directory.
    """
    stem = Path(ROBOT_MJCF[model]).stem
    return USD_DIR / stem / f"{stem}.usda"
