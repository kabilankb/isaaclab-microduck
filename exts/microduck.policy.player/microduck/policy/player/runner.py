"""Load an exported Microduck policy and step a task with it.

Deliberately free of any `omni.*` import so it can be unit-tested outside Kit; the
extension's UI layer is the only part that needs Kit running.

WHY TORCHSCRIPT, NOT A CHECKPOINT. This loads `exported/policy.pt`, not `model_*.pt`.
The exported TorchScript has the **observation normalizer baked in**; a raw checkpoint
does not, and an unnormalized policy behaves like a different robot. That failure is
invisible in-sim, because in-sim play applies the normalizer anyway -- so it only shows
up on hardware. `scripts/play.py` writes the export on every run.
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

    Accepts either the run directory or the `.pt` itself, because both are things a
    user will reasonably drag into a file picker.
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


def list_play_tasks() -> list[str]:
    """Registered `-Play-v0` task ids.

    Only the Play twins: they disable domain randomization and observation noise, and
    for locomotion they pin a real forward command instead of sampling one. A randomly
    sampled command makes a working policy look broken, because some envs are told to
    stand still.
    """
    import gymnasium as gym

    import isaaclab_microduck.tasks  # noqa: F401  -- registers on import

    return sorted(k for k in gym.registry if "MicroDuck" in k and k.endswith("-Play-v0"))


class PolicyRunner:
    """Owns the env and the policy for one playback session."""

    def __init__(self) -> None:
        self._env = None
        self._policy = None
        self._obs = None
        self.steps = 0

    @property
    def is_loaded(self) -> bool:
        return self._env is not None and self._policy is not None

    def load(self, task_id: str, policy_path: str | Path, num_envs: int = 4, device: str = "cuda:0") -> None:
        """Build the env and load the policy. Raises `PolicyRunnerError` on any failure."""
        import gymnasium as gym
        import torch

        import isaaclab_microduck.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        path = find_exported_policy(policy_path)
        self.unload()
        try:
            cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
            self._env = gym.make(task_id, cfg=cfg).unwrapped
        except Exception as exc:  # pragma: no cover - needs a simulator
            raise PolicyRunnerError(f"Could not build '{task_id}': {exc}") from exc

        try:
            policy = torch.jit.load(str(path), map_location=device).eval()
            probe = policy(torch.zeros(1, OBS_DIM, device=device))
        except Exception as exc:
            self.unload()
            raise PolicyRunnerError(f"Could not load policy '{path.name}': {exc}") from exc

        if probe.shape[-1] != ACTION_DIM:
            self.unload()
            raise PolicyRunnerError(
                f"Policy outputs {probe.shape[-1]} actions, expected {ACTION_DIM}. "
                "This is not a Microduck policy."
            )
        self._policy = policy
        self._obs, _ = self._env.reset()
        self.steps = 0

    def step(self) -> None:
        """Advance one control step. No-op when nothing is loaded."""
        import torch

        if not self.is_loaded:
            return
        obs = self._obs["policy"] if isinstance(self._obs, dict) else self._obs
        with torch.no_grad():
            action = self._policy(obs)
        self._obs, *_ = self._env.step(action)
        self.steps += 1

    def reset(self) -> None:
        if self.is_loaded:
            self._obs, _ = self._env.reset()
            self.steps = 0

    def unload(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # pragma: no cover - teardown is best-effort
                pass
        self._env = None
        self._policy = None
        self._obs = None
        self.steps = 0
