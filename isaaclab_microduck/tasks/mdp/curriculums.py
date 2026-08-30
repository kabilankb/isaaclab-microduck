"""Curriculum terms for the Microduck tasks.

Steps are ENV steps: `iteration * NUM_STEPS_PER_ENV` (24 here).

These are STEP FUNCTIONS, not interpolations — a ramp must be discretized into
stages. Every one of them mutates term configs through the MANAGERS
(`env.reward_manager.get_term_cfg(...)`), never `env.cfg.*`: the managers
deepcopy their configuration at init, so writing to `env.cfg` is a silent no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stage_value(stages: list[dict], step: int, key: str):
    """Value of the last stage whose `step` has been reached."""
    value = stages[0][key]
    for stage in stages:
        if step >= stage["step"]:
            value = stage[key]
    return value


def reward_weight(env: "ManagerBasedRLEnv", env_ids, reward_name: str, weight_stages: list[dict]) -> float:
    """Step-schedule a reward weight.

    Used to introduce taxes AFTER the skill exists: any attempt-tax active while a
    hard skill is still being explored makes "do nothing" win.
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    weight = _stage_value(weight_stages, int(env.common_step_counter), "weight")
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(reward_name, term_cfg)
    return weight


def pose_command_range_curriculum(
    env: "ManagerBasedRLEnv", env_ids, command_name: str, range_stages: list[dict]
) -> float:
    """Widen a pose command's per-dimension ranges in stages."""
    del env_ids
    ranges = _stage_value(range_stages, int(env.common_step_counter), "ranges")
    env.command_manager.get_term(command_name).cfg.ranges = ranges
    return float(max(abs(low) for low, _ in ranges))


def standing_envs_curriculum(
    env: "ManagerBasedRLEnv", env_ids, command_name: str, standing_stages: list[dict]
) -> float:
    """Raise the fraction of envs given the exact zero command.

    Zero-command behaviour has to be trained explicitly — it is the deployment
    idle state, and uniform sampling essentially never produces it.
    """
    del env_ids
    fraction = _stage_value(standing_stages, int(env.common_step_counter), "rel_standing_envs")
    env.command_manager.get_term(command_name).cfg.rel_standing_envs = fraction
    return fraction


def twist_range_curriculum(
    env: "ManagerBasedRLEnv", env_ids, command_name: str, range_stages: list[dict]
) -> float:
    """Widen the twist command's linear-x range in stages.

    Speed MUST be ramped, not set wide from step 0. The velocity cfg records what
    happens otherwise: widening the ranges once "outpaced the robot's capability
    and tracked a post-iteration-1000 decline in both reward and episode length".
    Starting at the known-good range and stretching it only after the gait holds
    keeps every stage inside what the policy can currently do.

    Mutates through the MANAGER, never `env.cfg`: managers deepcopy their config at
    init, so writing to `env.cfg` is a silent no-op.

    Returns the current max forward speed, so it is visible in the run log.
    """
    del env_ids
    lin_x = _stage_value(range_stages, int(env.common_step_counter), "lin_vel_x")
    term = env.command_manager.get_term(command_name)
    term.cfg.ranges.lin_vel_x = tuple(lin_x)
    return float(max(abs(v) for v in lin_x))
