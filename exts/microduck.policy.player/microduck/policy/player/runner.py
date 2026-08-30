"""Attach to a running Microduck env and hot-swap the policy driving it.

WHY THIS ATTACHES INSTEAD OF CREATING THE ENV
---------------------------------------------
An earlier version of this extension called `gym.make()` itself. That cannot work.
`SimulationContext` is a SINGLETON:

    # isaaclab/sim/simulation_context.py
    if cls._instance is not None:
        return cls._instance          # the cfg you passed is DISCARDED

and `ManagerBasedEnv.__init__` builds its sim with `SimulationContext(self.cfg.sim)`.
Inside a running Kit session a context already exists, so the env binds to the app's
context and the task's Newton MJWarp configuration is silently ignored -- the env comes
up on the wrong physics backend, or not at all. Isaac Lab environments must own the
app lifecycle (`AppLauncher -> env -> loop`), which is what `scripts/play.py` does.

So: launch the session with

    python scripts/play.py --task=<TASK>-Play-v0 --checkpoint <ckpt> --visualizer kit

and enable this extension inside that Kit window. It finds the live env and lets you
swap the policy without restarting -- which is also a rehearsal for what the real
runtime does, hot-swapping walk / stand / trick ONNX files against one shared 61D
observation buffer.

Imports no `omni.*`, so it is unit-testable outside Kit.
"""

from __future__ import annotations

from pathlib import Path

#: Actor observation width, fixed across the whole Microduck policy family.
OBS_DIM = 61

#: Servo count == action width.
ACTION_DIM = 14


class PolicyRunnerError(RuntimeError):
    """Raised with a message intended to be shown directly in the UI."""


def find_exported_policy(run_dir: str | Path) -> Path:
    """Resolve a run directory (or a direct file) to its exported TorchScript.

    Accepts either, because both are things a user will drag into a file picker.
    """
    p = Path(run_dir)
    if p.is_file():
        if p.name.startswith("model_"):
            raise PolicyRunnerError(
                f"'{p.name}' is a raw checkpoint. Use exported/policy.pt -- it has the "
                "observation normalizer baked in. Run scripts/play.py once to create it."
            )
        return p
    candidate = p / "exported" / "policy.pt"
    if candidate.is_file():
        return candidate
    raise PolicyRunnerError(f"No exported/policy.pt under '{p}'. Run scripts/play.py once first.")


def find_live_env():
    """Return the `ManagerBasedRLEnv` running in this process, or None.

    Found by scanning live objects rather than through a registry: Isaac Lab's own
    `play.py` creates the env, and this package has no hook into that script. A
    debugging panel scanning for the object it attaches to is an acceptable trade;
    inventing a global that only our own scripts set would not find envs created by
    the upstream script, which is the case that matters.
    """
    import gc

    try:
        from isaaclab.envs import ManagerBasedRLEnv
    except ImportError:  # pragma: no cover - Isaac Lab absent
        return None

    envs = [o for o in gc.get_objects() if isinstance(o, ManagerBasedRLEnv)]
    return envs[0] if envs else None


def load_policy(policy_path: str | Path, device: str = "cuda:0"):
    """Load exported TorchScript and verify it speaks the Microduck contract."""
    import torch

    path = find_exported_policy(policy_path)
    try:
        policy = torch.jit.load(str(path), map_location=device).eval()
        probe = policy(torch.zeros(1, OBS_DIM, device=device))
    except Exception as exc:
        raise PolicyRunnerError(f"Could not load '{path.name}': {exc}") from exc

    if probe.shape[-1] != ACTION_DIM:
        raise PolicyRunnerError(
            f"Policy outputs {probe.shape[-1]} actions, expected {ACTION_DIM}. "
            "This is not a Microduck policy."
        )
    return policy


class PolicyDriver:
    """Drives an ALREADY-RUNNING env with a swappable policy."""

    def __init__(self) -> None:
        self.env = None
        self.policy = None
        self._obs = None
        self.steps = 0

    @property
    def is_attached(self) -> bool:
        return self.env is not None

    @property
    def is_ready(self) -> bool:
        return self.env is not None and self.policy is not None

    def attach(self) -> str:
        """Find the live env. Returns a human-readable description."""
        env = find_live_env()
        if env is None:
            raise PolicyRunnerError(
                "No running environment found. Start one with:\n"
                "  python scripts/play.py --task=<TASK>-Play-v0 "
                "--checkpoint <ckpt> --visualizer kit\n"
                "then enable this extension inside that Kit window."
            )
        self.env = env
        self._obs = None
        self.steps = 0
        return f"{type(env).__name__}, {env.num_envs} envs, device {env.device}"

    def set_policy(self, policy_path: str | Path) -> None:
        if self.env is None:
            raise PolicyRunnerError("Attach to a running environment first.")
        self.policy = load_policy(policy_path, device=str(self.env.device))

    def step(self) -> None:
        """Advance one control step. No-op unless attached AND a policy is loaded."""
        import torch

        if not self.is_ready:
            return
        if self._obs is None:
            self._obs, _ = self.env.reset()
        obs = self._obs["policy"] if isinstance(self._obs, dict) else self._obs
        with torch.no_grad():
            action = self.policy(obs)
        self._obs, *_ = self.env.step(action)
        self.steps += 1

    def reset(self) -> None:
        if self.env is not None:
            self._obs, _ = self.env.reset()
            self.steps = 0

    def detach(self) -> None:
        self.env = None
        self.policy = None
        self._obs = None
        self.steps = 0
