#!/usr/bin/env python
"""Train a Microduck task with Isaac Lab's RSL-RL entry point.

    python scripts/train.py --task=Isaac-Velocity-Flat-MicroDuck-v0 \
        --num_envs=64 --max_iterations=5 --headless

All flags are Isaac Lab's own; this only registers the Microduck tasks first.
Checkpoints land under the EXPERIMENT NAME, not the task ID:
``$ISAACLAB_PATH/logs/rsl_rl/<experiment_name>/<timestamp>/``.
"""

import sys

from _launcher import exec_isaaclab_script

if __name__ == "__main__":
    exec_isaaclab_script("scripts/reinforcement_learning/rsl_rl/train.py", sys.argv[1:])
