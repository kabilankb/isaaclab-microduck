"""Shared launcher: run an Isaac Lab script with our tasks registered.

The Microduck tasks live in an EXTERNAL package (this one), not inside the Isaac
Lab repo -- the in-repo "internal task" path is only for upstreaming. Isaac Lab
supports exactly this with ``--external_callback``, so these wrappers add that
flag and hand everything else through untouched, rather than forking or copying
Isaac Lab's training scripts.

Three details this handles that a naive copy of ``train.py`` gets wrong:

* Isaac Lab's ``train.py`` / ``play.py`` import a SIBLING ``cli_args`` module, so
  that directory must be importable. We put it on ``PYTHONPATH`` rather than
  chdir'ing into it -- see the next point for why.
* Isaac Lab resolves its log directory RELATIVE TO THE CWD, so chdir'ing to the
  script's own directory buries runs in
  ``$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/logs/``. We chdir to the
  package root instead, which keeps runs in ``isaaclab_microduck/logs/rsl_rl/``,
  next to the mjlab stack's ``logs/`` and easy to compare against. Override with
  ``MICRODUCK_ISAACLAB_LOG_ROOT``.
* ``--external_callback`` is resolved with ``separator="."``, so the value is a
  dotted path ``module.path.attribute``, not the ``module:attribute`` form used
  elsewhere in Isaac Lab.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Registration hook, as Isaac Lab's ``--external_callback`` wants it (dotted).
EXTERNAL_CALLBACK = "isaaclab_microduck.tasks.register_tasks"

#: Default source-install location; override with ``ISAACLAB_PATH``.
_DEFAULT_ISAACLAB_PATH = Path("/home/chronos/IsaacLab")


def isaaclab_path() -> Path:
    """Locate the Isaac Lab checkout, preferring the ``ISAACLAB_PATH`` env var."""
    path = Path(os.environ.get("ISAACLAB_PATH", _DEFAULT_ISAACLAB_PATH))
    if not (path / "scripts").is_dir():
        raise SystemExit(
            f"Isaac Lab checkout not found at '{path}'. Set ISAACLAB_PATH to the "
            "directory containing isaaclab.sh."
        )
    return path


def log_root() -> Path:
    """Directory runs are written under (Isaac Lab resolves ``logs/`` from the CWD).

    Defaults to this package's root, so runs land in
    ``isaaclab_microduck/logs/rsl_rl/<experiment_name>/<timestamp>/`` -- inside the
    repo, alongside the mjlab stack's ``logs/``. Override with
    ``MICRODUCK_ISAACLAB_LOG_ROOT``.
    """
    path = Path(os.environ.get("MICRODUCK_ISAACLAB_LOG_ROOT", Path(__file__).resolve().parents[1]))
    path.mkdir(parents=True, exist_ok=True)
    return path


def exec_isaaclab_script(relative_script: str, argv: list[str]) -> None:
    """Replace this process with an Isaac Lab script, tasks registered.

    Args:
        relative_script: Script path relative to the Isaac Lab checkout, e.g.
            ``scripts/reinforcement_learning/rsl_rl/train.py``.
        argv: Arguments to forward (typically ``sys.argv[1:]``).
    """
    script = isaaclab_path() / relative_script
    if not script.is_file():
        raise SystemExit(f"Isaac Lab script not found: {script}")

    # Don't add the flag twice if the caller already passed their own.
    args = list(argv)
    if not any(a == "--external_callback" or a.startswith("--external_callback=") for a in args):
        args += ["--external_callback", EXTERNAL_CALLBACK]

    # The script imports its sibling `cli_args`: make that importable WITHOUT
    # chdir'ing there, so the run's log directory isn't resolved inside the Isaac
    # Lab checkout.
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [str(script.parent)] + ([env_pythonpath] if env_pythonpath else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    os.chdir(log_root())
    os.execv(sys.executable, [sys.executable, str(script), *args])
