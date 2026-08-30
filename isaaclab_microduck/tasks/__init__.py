"""Gym registration for every Microduck Isaac Lab task.

Task IDs mirror the mjlab ones (``Mjlab-Velocity-Flat-MicroDuck`` ->
``Isaac-Velocity-Flat-MicroDuck-v0``) so the two stacks stay easy to line up.

Registration happens on import. Isaac Lab's training scripts reach this module
through their ``--external_callback`` hook, which resolves a dotted path with
:func:`isaaclab.utils.string.string_to_callable` and expects back the list of
CLI arguments the callback consumed (``None`` = consumed nothing):

    --external_callback isaaclab_microduck.tasks.register_tasks
"""

import gymnasium as gym

##
# Task registry. IDs mirror the mjlab ones:
#   Mjlab-Velocity-Flat-MicroDuck -> Isaac-Velocity-Flat-MicroDuck-v0
##

gym.register(
    id="Isaac-Velocity-Flat-MicroDuck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity.velocity_env_cfg:MicroduckVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckVelocityPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-MicroDuck-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity.velocity_env_cfg:MicroduckVelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckVelocityPPORunnerCfg"
        ),
    },
)

# BallKick — kick a 70 mm / 15 g ball forward with the right foot from a standing
# start. Flat terrain only (a ball on rough terrain is another task). The actor is
# BALL-BLIND: ball state is critic-only, so the 61D contract is unchanged.
gym.register(
    id="Isaac-BallKick-Flat-MicroDuck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_kick.ball_kick_env_cfg:MicroduckBallKickFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckBallKickPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BallKick-Flat-MicroDuck-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_kick.ball_kick_env_cfg:MicroduckBallKickFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckBallKickPPORunnerCfg"
        ),
    },
)


# BallRally - two ducks, one ball, passed back and forth. Duck A learns; duck B
# replays a frozen BallKick policy, so this stays a SINGLE-AGENT env and the 61D
# actor contract is unchanged. A custom entry point drives the partner.
gym.register(
    id="Isaac-BallRally-Flat-MicroDuck-v0",
    entry_point="isaaclab_microduck.tasks.ball_rally.rally_env:MicroduckBallRallyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_rally.rally_env_cfg:MicroduckBallRallyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckBallRallyPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BallRally-Flat-MicroDuck-Play-v0",
    entry_point="isaaclab_microduck.tasks.ball_rally.rally_env:MicroduckBallRallyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_rally.rally_env_cfg:MicroduckBallRallyFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckBallRallyPPORunnerCfg"
        ),
    },
)


# RunParallel - two ducks running side by side. Duck A learns; duck B replays a
# frozen LOCOMOTION policy as a fixed-speed pacer. Single-agent env, 61D contract
# unchanged; the learner is partner-blind.
gym.register(
    id="Isaac-RunParallel-Flat-MicroDuck-v0",
    entry_point="isaaclab_microduck.tasks.run_parallel.run_env:MicroduckRunParallelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.run_parallel.run_env_cfg:MicroduckRunParallelFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckRunParallelPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-RunParallel-Flat-MicroDuck-Play-v0",
    entry_point="isaaclab_microduck.tasks.run_parallel.run_env:MicroduckRunParallelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.run_parallel.run_env_cfg:MicroduckRunParallelFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckRunParallelPPORunnerCfg"
        ),
    },
)


# Running - the velocity recipe with speed ramped by curriculum and a flight phase
# demanded. The walking policy tops out at 0.4 m/s because that is its whole
# command range; this task changes the demand, not the algorithm.
gym.register(
    id="Isaac-Running-Flat-MicroDuck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running.running_env_cfg:MicroduckRunningFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckRunningPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Running-Flat-MicroDuck-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running.running_env_cfg:MicroduckRunningFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_microduck.agents.rsl_rl_ppo_cfg:MicroduckRunningPPORunnerCfg"
        ),
    },
)


def register_tasks() -> None:
    """Entry point for Isaac Lab's ``--external_callback``.

    Importing this module is what actually registers the tasks; this function
    exists so the training scripts have something callable to name. It consumes
    no CLI arguments, hence the ``None`` return (``list_intersection`` treats
    that as "consumed nothing").
    """
    return None
