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
PolicyDriver = _r.PolicyDriver
PolicyRunnerError = _r.PolicyRunnerError
find_exported_policy = _r.find_exported_policy
find_live_env = _r.find_live_env


def test_runner_has_no_kit_dependency():
    # If this regresses, the only way left to test the extension is launching Isaac Sim.
    src = _RUNNER_PY.read_text()
    assert "import omni" not in src and "from omni" not in src


def test_ui_layer_is_the_only_part_that_needs_kit():
    src = (_EXT / "microduck/policy/player/extension.py").read_text()
    assert "import omni.ext" in src


def test_obs_contract_matches_the_policy_family():
    assert OBS_DIM == 61 and ACTION_DIM == 14


def test_create_refuses_when_a_context_already_exists():
    """Both modes exist, and Create must guard the singleton.

    In a PLAIN Isaac Sim session isaaclab's SimulationContext has no instance, so
    creating an env here is fine -- that is what makes task selection possible. Inside
    a play.py session one already exists, and a second env would silently inherit it
    and discard the task's Newton config, so Create must refuse rather than produce a
    wrong-physics env.
    """
    src = _RUNNER_PY.read_text()
    assert "SimulationContext.instance() is not None" in src
    assert "Use Attach instead." in src


def test_both_modes_are_available():
    d = PolicyDriver()
    assert hasattr(d, "create") and hasattr(d, "attach")


def test_release_only_closes_an_env_we_created():
    # Closing a play.py-owned env would kill the user's session.
    src = _RUNNER_PY.read_text()
    assert "_owns_env" in src


def test_raw_checkpoint_is_rejected_with_a_useful_message():
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


def test_attach_without_a_running_env_explains_how_to_start_one():
    d = PolicyDriver()
    assert not d.is_attached
    with pytest.raises(PolicyRunnerError, match="play.py"):
        d.attach()


def test_step_is_a_noop_when_not_ready():
    d = PolicyDriver()
    d.step()          # must not raise
    d.reset()
    d.detach()
    assert d.steps == 0


def test_no_live_env_in_a_bare_process():
    assert find_live_env() is None
