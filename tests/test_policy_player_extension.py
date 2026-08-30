"""The Isaac Sim extension's runner half must work without Kit.

`runner.py` deliberately imports nothing from `omni.*` so it can be tested here; only
the UI layer needs Kit running. These tests cover the failure modes a user will
actually hit -- pointing at a raw checkpoint, at a directory with no export, or at a
policy from a different robot.

CPU-only.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_EXT = Path(__file__).resolve().parents[1] / "exts" / "microduck.policy.player"
_RUNNER_PY = _EXT / "microduck" / "policy" / "player" / "runner.py"


def _load_runner():
    """Import runner.py DIRECTLY from its path, bypassing the package __init__.

    The package __init__ imports the UI module, which imports `omni.ext` -- present
    only inside Kit. Loading the file directly is what lets the Kit-independent half
    be tested in ordinary CPU CI, which is the whole reason it is a separate module.
    """
    spec = importlib.util.spec_from_file_location("_microduck_runner", _RUNNER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_r = _load_runner()
ACTION_DIM = _r.ACTION_DIM
OBS_DIM = _r.OBS_DIM
PolicyRunner = _r.PolicyRunner
PolicyRunnerError = _r.PolicyRunnerError
find_exported_policy = _r.find_exported_policy
list_play_tasks = _r.list_play_tasks


def test_runner_has_no_kit_dependency():
    # The whole point of the split. If this regresses, the extension can only be
    # tested by launching Isaac Sim, which no CI here does.
    src = _RUNNER_PY.read_text()
    assert "import omni" not in src and "from omni" not in src


def test_ui_layer_is_the_only_part_that_needs_kit():
    src = (_EXT / "microduck/policy/player/extension.py").read_text()
    assert "import omni.ext" in src


def test_obs_contract_matches_the_policy_family():
    assert OBS_DIM == 61 and ACTION_DIM == 14


def test_raw_checkpoint_is_rejected_with_a_useful_message():
    # A raw model_*.pt has NO baked normalizer; loading one produces a policy that
    # behaves like a different robot, and the bug is invisible in sim.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "model_5999.pt"
        ckpt.write_bytes(b"")
        with pytest.raises(PolicyRunnerError, match="raw checkpoint"):
            find_exported_policy(ckpt)


def test_missing_export_names_the_fix():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(PolicyRunnerError, match="scripts/play.py"):
            find_exported_policy(d)


def test_export_is_found_from_the_run_directory():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        exported = Path(d) / "exported"
        exported.mkdir()
        (exported / "policy.pt").write_bytes(b"")
        assert find_exported_policy(d).name == "policy.pt"


def test_only_play_variants_are_offered():
    # The non-Play tasks sample random commands, which makes a working locomotion
    # policy look broken -- roughly 1 in 10 envs is told to stand still.
    tasks = list_play_tasks()
    assert tasks, "no Microduck Play tasks registered"
    assert all(t.endswith("-Play-v0") for t in tasks)


def test_runner_starts_unloaded_and_step_is_a_noop():
    r = PolicyRunner()
    assert not r.is_loaded
    r.step()          # must not raise when nothing is loaded
    r.reset()
    r.unload()
    assert r.steps == 0
