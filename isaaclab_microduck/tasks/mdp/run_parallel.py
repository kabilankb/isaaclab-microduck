"""RunParallel MDP terms: two ducks running side by side.

The LEARNER is duck A; duck B replays a frozen locomotion policy and acts as the
pacer. Single-agent from the manager's point of view, so the 61D actor contract is
untouched, and the learner is PARTNER-BLIND -- partner state is critic-only, exactly
as the ball is in ball_kick. That is deliberate: the real robot has no teammate
sensing, so a policy that homed on the partner would not transfer.

Design problem specific to this task: "stay abreast" is a POSITION relationship, and
paying for it directly is farmable -- the cheapest way to match a pacer's position is
to stop moving whenever you drift ahead, which produces a lurching non-gait. So the
formation term here pays only the LONGITUDINAL error against the pacer, is bounded,
and is deliberately weaker than the forward-speed tracking it must not override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_microduck.utils.arrays import as_torch

# Deferred: eagerly resolving these forces Isaac Lab's lazy loader to import
# simulation_context -> scene_data_provider -> pxr at CFG-IMPORT time, which breaks
# `--visualizer kit`. See tasks/mdp/command_impl.py.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _pair_xy(env, learner_cfg, partner_cfg):
    learner: "Articulation" = env.scene[learner_cfg.name]
    partner: "Articulation" = env.scene[partner_cfg.name]
    return (
        as_torch(learner.data.root_link_pos_w)[:, :2],
        as_torch(partner.data.root_link_pos_w)[:, :2],
    )


def abreast_error(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    learner_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    partner_cfg: SceneEntityCfg = SceneEntityCfg("partner"),
) -> torch.Tensor:
    """Gaussian on LONGITUDINAL offset from the pacer. Lateral offset is ignored.

    Only the along-track component is priced. The lane separation is a fixed spawn
    offset the policy should not be paid to close -- rewarding lateral proximity
    would invite the two robots to converge and collide.

    `std` is set to the offset we still care about (25 cm ~ one robot length), not
    the maximum: a std tight enough to punish normal gait phase differences would
    tax running itself, which is the trap AGENTS.md records for head tracking.
    """
    learner_xy, partner_xy = _pair_xy(env, learner_cfg, partner_cfg)
    # Along-track axis is the ENV's forward (+x); lanes are separated in y.
    along = learner_xy[:, 0] - partner_xy[:, 0]
    return torch.exp(-torch.square(torch.nan_to_num(along, nan=0.0)) / std**2)


def pair_separation_penalty(
    env: "ManagerBasedRLEnv",
    min_separation: float = 0.25,
    learner_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    partner_cfg: SceneEntityCfg = SceneEntityCfg("partner"),
) -> torch.Tensor:
    """Cost (>= 0) for closing inside `min_separation` laterally. NEGATIVE weight.

    Anti-collision. Hinged rather than a Gaussian well so it is exactly zero at
    normal lane spacing and only bites when the pair actually converges -- a term
    that pays everywhere would compete with the gait for gradient.
    """
    learner_xy, partner_xy = _pair_xy(env, learner_cfg, partner_cfg)
    lateral = torch.abs(learner_xy[:, 1] - partner_xy[:, 1])
    return torch.clamp(min_separation - torch.nan_to_num(lateral, nan=min_separation), min=0.0)


def pair_metrics(env: "ManagerBasedRLEnv", env_ids=None) -> float:
    """Publish `Metrics/abreast_error_m` and `Metrics/pair_separation_m`.

    The reward totals cannot stand in for these: they rise on the locomotion stack
    alone while the pair drifts apart. Runs as a curriculum term, the per-reset hook
    that fires before reset events move the robots.
    """
    learner_xy, partner_xy = _pair_xy(env, SceneEntityCfg("robot"), SceneEntityCfg("partner"))
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return 0.0

    along = torch.abs(learner_xy[env_ids, 0] - partner_xy[env_ids, 0])
    lateral = torch.abs(learner_xy[env_ids, 1] - partner_xy[env_ids, 1])
    log = env.extras.setdefault("log", {})
    log["Metrics/abreast_error_m"] = along.mean().item()
    log["Metrics/pair_separation_m"] = lateral.mean().item()
    return along.mean().item()
