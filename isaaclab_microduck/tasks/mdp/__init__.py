"""Microduck MDP terms, layered on top of `isaaclab.envs.mdp`.

Import the Isaac Lab terms directly where they exist; everything here is either
absent upstream or has a different kernel than the mjlab recipe requires.
"""

import isaaclab.envs.mdp as _isaaclab_mdp


def __getattr__(name: str):
    """Forward unknown names to `isaaclab.envs.mdp`, LAZILY.

    This deliberately replaces `from isaaclab.envs.mdp import *`. That star-import
    forces every lazily-loaded attribute in Isaac Lab's mdp namespace to evaluate
    at import time, which pulls in `isaaclab.scene_data.scene_data_provider` and
    through it `pxr` -- pip's USD.

    That mattered far beyond tidiness. Task cfgs are imported during CLI preset
    collection, BEFORE Kit starts, so the star-import put pip's USD in the process
    first; Kit then loaded its own separately-built USD copy and aborted in
    `libusd_tf.so` static init with "free(): invalid pointer". The visible symptom
    was that `--visualizer kit` segfaulted on every Microduck task while working
    fine on Isaac Lab's own (which do not star-import).

    PEP 562 module `__getattr__` keeps every `mdp.*` name working while leaving
    Isaac Lab's lazy loading intact. Do NOT restore the star-import.
    """
    return getattr(_isaaclab_mdp, name)

# Locomotion-specific terms live in the task package, not the core mdp library.
# `feet_air_time_positive_biped` (not the quadruped `feet_air_time`) is the right
# one here: it gates on SINGLE STANCE, so it rewards alternating steps rather than
# both feet leaving the ground together, which on a biped is a hop, not a gait.
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (  # noqa: F401
    feet_air_time,
    feet_air_time_positive_biped,
)

from .ball_kick import (  # noqa: F401
    ball_forward_velocity,
    kick_success,
    ball_pos_in_base,
    ball_speed_overshoot_penalty,
    ball_vel_in_base,
    height_target_gaussian,
    pose_target_match,
    reset_ball_in_front_of_foot,
    single_foot_grounded_reward,
    zero_command_padding,
)
from .ball_rally import (  # noqa: F401
    ball_out_of_play,
    ball_progress_to_partner,
    pass_completed,
    rally_length,
    reset_rally_state,
)
# Only the CFG classes here. The runtime command classes live in `command_impl`
# and are referenced by string, so importing a task cfg does not pull pxr in
# before Kit starts -- see command_impl.py.
from .run_parallel import (  # noqa: F401
    abreast_error,
    pair_metrics,
    pair_separation_penalty,
)
from .commands import (  # noqa: F401
    MicroduckVelocityCommandCfg,
    UniformPoseCommandCfg,
)
from .curriculums import (  # noqa: F401
    pose_command_range_curriculum,
    reward_weight,
    standing_envs_curriculum,
    twist_range_curriculum,
)
from .observations import (  # noqa: F401
    accumulate_speed,
    locomotion_metrics,
    base_ang_vel_imu_misaligned,
    projected_gravity_imu_misaligned,
)
from .rewards import (  # noqa: F401
    yaw_rate_error_l1,
    forward_speed_linear,
    feet_clearance,
    feet_swing_height,
    body_angular_velocity_penalty,
    body_pose_tracking_6d,
    feet_slip,
    head_pose_bias_penalty,
    head_pose_tracking,
    self_collision_cost,
    track_angular_velocity,
    track_linear_velocity,
    upright,
    variable_posture,
)
from .terminations import robot_state_is_nan  # noqa: F401
