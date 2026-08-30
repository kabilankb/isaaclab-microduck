"""Create or attach to a Microduck env and drive it with an exported policy.

TWO MODES, because which one works depends on where this extension is loaded:

* **Create** -- in a plain Isaac Sim session, `isaaclab.sim.SimulationContext` has no
  instance yet (it is Isaac Lab's OWN singleton, not a subclass of Isaac Sim's), so an
  env built here creates its own context with the task's Newton MJWarp config. This is
  the mode that lets you pick a task from the UI.
* **Attach** -- inside a session started by `scripts/play.py`, an env already exists.
  Creating a second one would hit the singleton:

      # isaaclab/sim/simulation_context.py
      if cls._instance is not None:
          return cls._instance      # the cfg you pass is DISCARDED

  so the task's Newton config would be silently ignored. Attach to the live one instead.

Try Create first; fall back to Attach if a session is already running.

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


def list_play_tasks() -> list[str]:
    """Registered `-Play-v0` task ids.

    Only the Play twins: they disable domain randomization and observation noise, and
    for locomotion they pin a real forward command instead of sampling one. A randomly
    sampled command makes a working policy look broken, because roughly 1 env in 10 is
    told to stand still.
    """
    import gymnasium as gym

    import isaaclab_microduck.tasks  # noqa: F401  -- registers on import

    return sorted(k for k in gym.registry if "MicroDuck" in k and k.endswith("-Play-v0"))


def create_env(task_id: str, num_envs: int = 4, device: str = "cuda:0"):
    """Build a task env in THIS process. Only valid when no Isaac Lab env exists yet.

    Raises `PolicyRunnerError` with the underlying failure, rather than swallowing it:
    whether this works inside a given Kit app is exactly the thing worth seeing.
    """
    import gymnasium as gym

    import isaaclab_microduck.tasks  # noqa: F401
    from isaaclab.sim import SimulationContext
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    if SimulationContext.instance() is not None:
        raise PolicyRunnerError(
            "An Isaac Lab simulation context already exists in this process, so a new "
            "env would silently inherit it and ignore this task's Newton config. "
            "Use Attach instead."
        )
    try:
        cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        return gym.make(task_id, cfg=cfg).unwrapped
    except Exception as exc:
        raise PolicyRunnerError(f"Could not create '{task_id}': {exc}") from exc


class PolicyDriver:
    """Drives an ALREADY-RUNNING env with a swappable policy."""

    def __init__(self) -> None:
        self.env = None
        self.policy = None
        self._obs = None
        self._owns_env = False
        self.steps = 0

    @property
    def is_attached(self) -> bool:
        return self.env is not None

    @property
    def is_ready(self) -> bool:
        return self.env is not None and self.policy is not None

    def create(self, task_id: str, num_envs: int = 4) -> str:
        """Build the env here and own it."""
        self.detach()
        self.env = create_env(task_id, num_envs=num_envs)
        self._owns_env = True
        self._obs = None
        self.steps = 0
        return f"{task_id}, {self.env.num_envs} envs, device {self.env.device}"

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
        self._owns_env = False
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
        # Only close what we created; closing a play.py-owned env would kill its session.
        if self._owns_env and self.env is not None:
            try:
                self.env.close()
            except Exception:  # pragma: no cover - teardown is best-effort
                pass
        self._owns_env = False
        self.env = None
        self.policy = None
        self._obs = None
        self.steps = 0
