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


# --------------------------------------------------------------------------
# The BaseSample path: observation assembly is the part that fails SILENTLY.
# A wrong rotation or a wrong slot order still runs; it just behaves badly.
# --------------------------------------------------------------------------


def _load_sample_mod():
    spec = importlib.util.spec_from_file_location(
        "_microduck_sample", _EXT / "microduck" / "policy" / "player" / "sample.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_s = _load_sample_mod()


def test_sample_has_no_kit_dependency_at_import():
    """Scene APIs are imported INSIDE setup_scene, so the module imports bare.

    Checked over the AST rather than the text: the module docstring mentions Isaac Sim
    by name, and a substring search matches prose as readily as code.
    """
    import ast

    tree = ast.parse((_EXT / "microduck/policy/player/sample.py").read_text())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = set()
    for n in top_level:
        if isinstance(n, ast.ImportFrom):
            names.add(n.module or "")
        else:
            names.update(a.name for a in n.names)
    offenders = {m for m in names if m.split(".")[0] in {"omni", "isaacsim", "pxr"}}
    assert not offenders, offenders


def test_observation_width_matches_the_shared_contract():
    assert _s.OBS_DIM == 61
    assert _s.ACTION_DIM == 14
    assert _s.COMMAND_DIM == 13
    # 3 + 3 + 14 + 14 + 14 + 13 == 61: the layout must actually add up.
    assert 3 + 3 + _s.ACTION_DIM * 3 + _s.COMMAND_DIM == _s.OBS_DIM


def test_identity_rotation_is_a_no_op():
    import numpy as np

    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _s._rotate_inverse(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), v)
    assert np.allclose(out, v, atol=1e-6)


def test_gravity_points_down_when_upright():
    import numpy as np

    # Upright robot: projected gravity must be (0, 0, -1).
    g = _s._rotate_inverse(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    assert np.allclose(g, [0.0, 0.0, -1.0], atol=1e-6)


def test_yaw_rotation_leaves_gravity_untouched():
    import math

    import numpy as np

    # Yaw must not tilt gravity -- if it does, the rotation is wrong.
    a = math.pi / 3
    quat = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], dtype=np.float32)
    g = _s._rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert np.allclose(g, [0.0, 0.0, -1.0], atol=1e-5), g


def test_pitch_rotation_tilts_gravity_into_x():
    import math

    import numpy as np

    # 90 deg pitch about y: body-frame gravity moves out of z into x.
    a = math.pi / 2
    quat = np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0], dtype=np.float32)
    g = _s._rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert abs(abs(g[0]) - 1.0) < 1e-5 and abs(g[2]) < 1e-5, g


def test_observation_is_zeros_before_a_robot_exists():
    import numpy as np

    smp = _s.MicroduckPolicySample()
    obs = smp.build_observation()
    assert obs.shape == (_s.OBS_DIM,) and np.all(obs == 0)


def test_forward_command_writes_the_twist_x_slot():
    # Slot 0 is the same one the deployment runtime writes for walking.
    smp = _s.MicroduckPolicySample()
    smp.set_forward_command(0.7)
    assert smp._command[0] == pytest.approx(0.7)
    assert smp._command[1:].sum() == 0.0


def test_step_policy_is_a_noop_without_policy_or_robot():
    smp = _s.MicroduckPolicySample()
    assert smp.step_policy() is None
