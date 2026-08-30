"""PPO runner configs, ported from `MicroduckRlCfg` in the mjlab stack."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def _model(*, actor: bool) -> RslRlMLPModelCfg:
    """512/256/128 ELU with observation normalization ON.

    The normalizer being on is why export MUST bake it into the ONNX: in-sim play
    applies it anyway, so a hand-converted checkpoint looks fine in the viewer and
    fails on hardware.

    Only the ACTOR carries a distribution — it is the stochastic policy. Giving the
    critic one, or omitting it on the actor, fails at runner construction with
    `'NoneType' object has no attribute 'log_prob'`.
    """
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=(
            RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar")
            if actor
            else None
        ),
    )


_ALGORITHM = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class MicroduckVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Gaits are curriculum-heavy: budget 4000-6000 iterations at 4096 envs."""

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 250
    experiment_name = "microduck_velocity"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor: RslRlMLPModelCfg = _model(actor=True)
    critic: RslRlMLPModelCfg = _model(actor=False)
    algorithm: RslRlPpoAlgorithmCfg = _ALGORITHM


@configclass
class MicroduckBallKickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """A simple episodic trick: budget ~1000 iterations at 4096 envs."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 250
    experiment_name = "microduck_ball_kick"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor: RslRlMLPModelCfg = _model(actor=True)
    critic: RslRlMLPModelCfg = _model(actor=False)
    algorithm: RslRlPpoAlgorithmCfg = _ALGORITHM


@configclass
class MicroduckBallRallyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """A simple episodic trick: budget ~1000 iterations at 4096 envs."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 250
    experiment_name = "microduck_ball_rally"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor: RslRlMLPModelCfg = _model(actor=True)
    critic: RslRlMLPModelCfg = _model(actor=False)
    algorithm: RslRlPpoAlgorithmCfg = _ALGORITHM


@configclass
class MicroduckRunParallelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Gaits are curriculum-heavy: budget 4000-6000 iterations at 4096 envs."""

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 250
    experiment_name = "microduck_run_parallel"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor: RslRlMLPModelCfg = _model(actor=True)
    critic: RslRlMLPModelCfg = _model(actor=False)
    algorithm: RslRlPpoAlgorithmCfg = _ALGORITHM


@configclass
class MicroduckRunningPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Gaits are curriculum-heavy: budget 4000-6000 iterations at 4096 envs."""

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 250
    experiment_name = "microduck_running"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor: RslRlMLPModelCfg = _model(actor=True)
    critic: RslRlMLPModelCfg = _model(actor=False)
    algorithm: RslRlPpoAlgorithmCfg = _ALGORITHM
