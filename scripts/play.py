#!/usr/bin/env python
"""Evaluate a Microduck checkpoint with Isaac Lab's RSL-RL play entry point.

    python scripts/play.py --task=Isaac-Velocity-Flat-MicroDuck-v0 \
        --checkpoint <path/to/model_XXXX.pt> --num_envs=16 physics=newton_mjwarp

For RSL-RL, ``play.py`` also exports the policy to TorchScript and ONNX under the
checkpoint's ``exported/`` directory. That export is NOT the sim2real hand-off:
use ``scripts/export.py`` (P7), which guarantees the observation normalizer is
baked in and the ``passive_*`` joints are filtered out of the metadata.
"""

import sys

from _launcher import exec_isaaclab_script

if __name__ == "__main__":
    exec_isaaclab_script("scripts/reinforcement_learning/rsl_rl/play.py", sys.argv[1:])
