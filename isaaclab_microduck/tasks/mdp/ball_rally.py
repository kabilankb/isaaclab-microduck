"""BallRally MDP terms: two ducks, one ball, passed back and forth.

The LEARNER is duck A. Duck B is a frozen `ball_kick` policy (see
`rally_env.MicroduckBallRallyEnv`), so this is a single-agent env from the
manager's point of view and the 61D actor contract is untouched.

Both ducks stay BALL-BLIND, exactly as in `ball_kick`: ball state is critic-only.
The learner therefore cannot home on the ball, and robustness comes from the
placement spread rather than from a trajectory tuned to one ball position.

The central reward-design problem here is that a rally pays PER PASS, and a
per-pass bonus is a jackpot: arriving at the goal state early and repeatedly is
worth arbitrary violence. Every scoring term below is therefore LATCHED -- a pass
can only score once, and only re-arms after the ball has come back. Farming the
same crossing twice is worth zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_microduck.utils.arrays import as_torch

# Deferred: resolving `Articulation` / `RigidObject` eagerly forces Isaac Lab's
# lazy loader to import simulation_context -> scene_data_provider -> pxr (pip's
# USD) at CFG-IMPORT time, which happens BEFORE Kit starts. Kit then loads its
# own USD copy and aborts in libusd_tf.so static init. These names are used only
# in annotations, and PEP 526 never evaluates local annotations at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _rally(env: "ManagerBasedRLEnv") -> dict:
    """Lazily-created per-env rally bookkeeping.

    `armed` is the latch that stops a single crossing being farmed: it is True
    while a pass TOWARD the partner can still score, and only re-arms once the
    ball has returned past the halfway line.
    """
    if not hasattr(env, "_rally_state"):
        n, dev = env.num_envs, env.device
        env._rally_state = {
            "armed": torch.ones(n, dtype=torch.bool, device=dev),
            "passes": torch.zeros(n, device=dev),
            "prev_gap": torch.zeros(n, device=dev),
            "has_prev": torch.zeros(n, dtype=torch.bool, device=dev),
        }
    return env._rally_state


def _positions(env, ball_cfg, learner_cfg, partner_cfg):
    ball: RigidObject = env.scene[ball_cfg.name]
    learner: Articulation = env.scene[learner_cfg.name]
    partner: Articulation = env.scene[partner_cfg.name]
    return (
        as_torch(ball.data.root_link_pos_w)[:, :2],
        as_torch(learner.data.root_link_pos_w)[:, :2],
        as_torch(partner.data.root_link_pos_w)[:, :2],
    )


def ball_progress_to_partner(
    env: "ManagerBasedRLEnv",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    learner_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    partner_cfg: SceneEntityCfg = SceneEntityCfg("partner"),
) -> torch.Tensor:
    """POTENTIAL-BASED shaping: pay the per-step DECREASE in ball->partner gap.

    Potential-based rather than a distance bonus on purpose. A term that pays for
    the ball merely BEING near the partner is farmable by parking it there; paying
    only the change means holding position pays exactly zero and the integral over
    any closed round trip is zero. Moving the ball away costs the same amount it
    paid to move it closer, so there is no cycle to exploit.

    Returns 0 on the first step after a reset, when there is no previous gap.
    """
    state = _rally(env)
    ball_xy, _, partner_xy = _positions(env, ball_cfg, learner_cfg, partner_cfg)
    gap = torch.linalg.norm(partner_xy - ball_xy, dim=1)

    delta = torch.where(state["has_prev"], state["prev_gap"] - gap, torch.zeros_like(gap))
    state["prev_gap"] = gap
    state["has_prev"] = torch.ones_like(state["has_prev"])
    return torch.nan_to_num(delta, nan=0.0)


def pass_completed(
    env: "ManagerBasedRLEnv",
    reach_radius: float = 0.20,
    rearm_fraction: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    learner_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    partner_cfg: SceneEntityCfg = SceneEntityCfg("partner"),
) -> torch.Tensor:
    """1.0 on the step the ball first reaches the partner, else 0. LATCHED.

    The latch is what keeps this from being a jackpot. Without it a ball resting
    inside `reach_radius` would pay every step, and the optimum would be to shove
    the ball into the partner's feet and leave it there. Here the term fires once,
    disarms, and only re-arms when the ball has travelled back past
    `rearm_fraction` of the way to the learner -- i.e. only a genuine RETURN makes
    another pass scoreable, which is exactly the rally we want.
    """
    state = _rally(env)
    ball_xy, learner_xy, partner_xy = _positions(env, ball_cfg, learner_cfg, partner_cfg)

    to_partner = torch.linalg.norm(partner_xy - ball_xy, dim=1)
    to_learner = torch.linalg.norm(learner_xy - ball_xy, dim=1)
    separation = torch.linalg.norm(partner_xy - learner_xy, dim=1).clamp_min(1e-6)

    arrived = to_partner < reach_radius
    scored = arrived & state["armed"]

    state["passes"] = state["passes"] + scored.float()
    state["armed"] = state["armed"] & ~arrived
    # Re-arm once the ball is back on the learner's side of the halfway line.
    state["armed"] = state["armed"] | (to_learner < rearm_fraction * separation)
    return scored.float()


def rally_length(env: "ManagerBasedRLEnv", env_ids=None) -> float:
    """Mean completed passes per episode -> `Metrics/rally_passes`.

    THIS is the task's score. Reward totals cannot stand in for it: they rise on
    the standing stack alone even when no pass ever happens.

    Runs as a curriculum term because that hook fires on reset BEFORE the reset
    events clear the counters, so the finished episode's tally is still intact.
    """
    state = _rally(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return 0.0

    mean_passes = state["passes"][env_ids].mean().item()
    log = env.extras.setdefault("log", {})
    log["Metrics/rally_passes"] = mean_passes
    log["Metrics/rally_success_rate"] = (state["passes"][env_ids] >= 1.0).float().mean().item()
    return mean_passes


def reset_rally_state(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Clear the latch and counters. MUST run in the reset events."""
    if env_ids is None or len(env_ids) == 0:
        return
    state = _rally(env)
    state["armed"][env_ids] = True
    state["passes"][env_ids] = 0.0
    state["prev_gap"][env_ids] = 0.0
    state["has_prev"][env_ids] = False


def ball_out_of_play(
    env: "ManagerBasedRLEnv",
    max_distance: float = 2.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    learner_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the ball is hopelessly far from the learner.

    Without this a ball blasted off the mat leaves the episode running with
    nothing to do, and the policy still collects the standing stack for the full
    time-out -- paying it to end the rally.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    learner: Articulation = env.scene[learner_cfg.name]
    gap = torch.linalg.norm(
        as_torch(ball.data.root_link_pos_w)[:, :2] - as_torch(learner.data.root_link_pos_w)[:, :2], dim=1
    )
    return gap > max_distance
