"""Microduck policy playback as an Isaac Sim `BaseSample`.

WHY BaseSample AND NOT AN ISAAC LAB ENV
---------------------------------------
An earlier version of this extension tried to build a `ManagerBasedRLEnv` from inside
Kit. Isaac Lab environments expect to own the application lifecycle
(`AppLauncher -> env -> loop`), and inside a running Isaac Sim session that is the
wrong shape: the env wants to construct its own `SimulationContext` and drive stepping.

`BaseSample` is Isaac Sim's own contract for exactly this -- it owns an
`isaacsim.core.api.World`, and the Examples Browser gives it Load / Reset / Clear.
So the scene is built with Isaac Sim APIs and the exported policy is stepped against
it directly. The policy does not care: it is TorchScript taking 61 observations and
returning 14 joint targets, with the observation normalizer already baked in.

That means the 61D observation is assembled HERE, in the same order every Microduck
policy was trained against:

    base_ang_vel(3) projected_gravity(3) joint_pos(14) joint_vel(14)
    actions(14) twist(3) head_pose(4) body_pose(6)

Get that order wrong and the policy still runs, silently, badly -- which is why the
layout is spelled out rather than assumed.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 61
ACTION_DIM = 14

#: Command block width: twist(3) + head_pose(4) + body_pose(6).
COMMAND_DIM = 13


class MicroduckPolicySample:
    """Scene + policy playback. Instantiated by the extension; driven by BaseSample."""

    def __init__(self) -> None:
        self._policy = None
        self._robot = None
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._command = np.zeros(COMMAND_DIM, dtype=np.float32)
        self._default_joint_pos = np.zeros(ACTION_DIM, dtype=np.float32)
        self.status = "No policy loaded."
        self.steps = 0

    # -- scene ------------------------------------------------------------

    def setup_scene(self, world, model: str = "allcollisions") -> None:
        """Add a ground plane and one Microduck to `world`.

        `world` is an `isaacsim.core.api.World` supplied by BaseSample. The USD is the
        same build artifact the training stack uses -- if it is missing, the message
        says to run convert_assets.py rather than failing on a path.
        """
        from isaacsim.core.api.objects.ground_plane import GroundPlane
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage

        from isaaclab_microduck.assets.paths import usd_path
        from isaaclab_microduck.robot.microduck_cfg import HOME_JOINT_POS, SPAWN_HEIGHT

        usd = usd_path(model)
        if not usd.is_file():
            raise RuntimeError(
                f"USD for '{model}' not built: {usd}. Run scripts/convert_assets.py --force"
            )

        GroundPlane(prim_path="/World/groundPlane", size=10.0)
        prim_path = "/World/Microduck"
        add_reference_to_stage(usd_path=str(usd), prim_path=prim_path)
        self._robot = world.scene.add(
            Robot(prim_path=prim_path, name="microduck", position=np.array([0.0, 0.0, SPAWN_HEIGHT]))
        )
        self._home = dict(HOME_JOINT_POS)
        self.status = f"Scene ready ({model}). Load a policy."

    def setup_post_load(self) -> None:
        """Cache HOME in the articulation's own joint order.

        Joint ORDER is not guaranteed to match the config's declaration order, and a
        mismatch here silently sends every servo the wrong target -- so it is resolved
        against the live articulation rather than assumed.
        """
        if self._robot is None:
            return
        names = list(self._robot.dof_names)[:ACTION_DIM]
        self._default_joint_pos = np.array(
            [self._home.get(n, 0.0) for n in names], dtype=np.float32
        )
        self._robot.set_joint_positions(self._default_joint_pos)
        self.status = "Robot at HOME. Load a policy and press Play."

    # -- policy -----------------------------------------------------------

    def load_policy(self, path: str) -> None:
        """Load exported TorchScript. Raises RuntimeError with a UI-ready message."""
        import torch

        from .runner import find_exported_policy

        resolved = find_exported_policy(path)
        policy = torch.jit.load(str(resolved), map_location="cpu").eval()
        probe = policy(torch.zeros(1, OBS_DIM))
        if probe.shape[-1] != ACTION_DIM:
            raise RuntimeError(
                f"Policy outputs {probe.shape[-1]} actions, expected {ACTION_DIM}. "
                "Not a Microduck policy."
            )
        self._policy = policy
        self.status = f"Loaded {resolved.name}."

    def set_forward_command(self, vx: float) -> None:
        """Set the twist command's linear-x slot.

        Slot 0 is the same one the deployment runtime writes for walking. Feeding all
        zeros means "stand", which reads as "the policy ignores the command".
        """
        self._command[0] = float(vx)

    # -- observation ------------------------------------------------------

    def build_observation(self) -> np.ndarray:
        """Assemble the 61D actor observation from the live articulation."""
        if self._robot is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        quat = np.asarray(self._robot.get_world_pose()[1], dtype=np.float32)  # (w,x,y,z)
        ang_vel = np.asarray(self._robot.get_angular_velocity(), dtype=np.float32)
        joint_pos = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)[:ACTION_DIM]
        joint_vel = np.asarray(self._robot.get_joint_velocities(), dtype=np.float32)[:ACTION_DIM]

        obs = np.concatenate(
            [
                _rotate_inverse(quat, ang_vel),
                _rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)),
                joint_pos - self._default_joint_pos,
                joint_vel,
                self._prev_action,
                self._command,
            ]
        ).astype(np.float32)

        if obs.shape[0] != OBS_DIM:  # pragma: no cover - guards a silent contract break
            raise RuntimeError(f"observation is {obs.shape[0]}D, expected {OBS_DIM}D")
        return obs

    def step_policy(self) -> np.ndarray | None:
        """One control step: observe, infer, write joint targets."""
        import torch

        if self._policy is None or self._robot is None:
            return None
        obs = torch.from_numpy(self.build_observation()).unsqueeze(0)
        with torch.no_grad():
            action = self._policy(obs).squeeze(0).numpy()
        self._prev_action = action.astype(np.float32)
        self.steps += 1
        return self._default_joint_pos + action  # scale 1.0, offset from HOME


def _rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` into the body frame given a (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz
    q = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(q, vec)
    return (vec - w * t + np.cross(q, t)).astype(np.float32)
