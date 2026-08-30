"""P1 gate: the external package registers its tasks with gymnasium.

CPU-only and simulator-free — importing ``isaaclab_microduck.tasks`` needs nothing
beyond gymnasium, which is what keeps this runnable in CI.
"""

import gymnasium as gym
import pytest

import isaaclab_microduck.tasks  # noqa: F401  -- import registers the tasks


def _microduck_task_ids() -> list[str]:
    return sorted(k for k in gym.registry if "MicroDuck" in k)


def test_at_least_one_task_is_registered():
    assert _microduck_task_ids(), "importing isaaclab_microduck.tasks registered nothing"


def test_task_ids_follow_the_isaac_convention():
    # Isaac Lab's convention is Isaac-<Name>-v0; mjlab's ids map onto it one for one
    # (Mjlab-Velocity-Flat-MicroDuck -> Isaac-Velocity-Flat-MicroDuck-v0), which is
    # what keeps the two stacks easy to line up when comparing runs.
    for task_id in _microduck_task_ids():
        assert task_id.startswith("Isaac-"), task_id
        assert task_id.endswith("-v0"), task_id


@pytest.mark.parametrize("task_id", _microduck_task_ids())
def test_env_cfg_entry_point_is_declared(task_id):
    assert gym.registry[task_id].kwargs.get("env_cfg_entry_point"), (
        f"{task_id} has no env_cfg_entry_point"
    )


def test_external_callback_is_resolvable():
    """The dotted path Isaac Lab's --external_callback resolves must stay valid.

    Isaac Lab resolves it with string_to_callable(..., separator="."), i.e.
    ``module.path.attribute`` — not the ``module:attribute`` form used elsewhere.
    """
    import importlib

    from isaaclab_microduck.tasks import register_tasks

    from scripts._launcher import EXTERNAL_CALLBACK  # noqa: PLC0415

    mod_name, attr_name = EXTERNAL_CALLBACK.rsplit(".", 1)
    resolved = getattr(importlib.import_module(mod_name), attr_name)
    assert resolved is register_tasks
    # Returning None means "consumed no CLI args"; train.py relies on that.
    assert register_tasks() is None
