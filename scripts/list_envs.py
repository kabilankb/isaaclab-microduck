#!/usr/bin/env python
"""List the Microduck tasks this package registers.

Deliberately lightweight: it imports only ``isaaclab_microduck.tasks`` (which
needs nothing beyond gymnasium), so it answers "did registration work?" without
launching the simulator. For the full 200+ task table including Isaac Lab's own
environments, use ``$ISAACLAB_PATH/scripts/environments/list_envs.py``.
"""

import gymnasium as gym

import isaaclab_microduck.tasks  # noqa: F401  -- import registers the tasks

if __name__ == "__main__":
    ids = sorted(k for k in gym.registry if "MicroDuck" in k)
    if not ids:
        raise SystemExit("No Microduck tasks registered.")
    width = max(len(i) for i in ids)
    print(f"{len(ids)} Microduck task(s) registered:\n")
    for task_id in ids:
        cfg = gym.registry[task_id].kwargs.get("env_cfg_entry_point", "-")
        print(f"  {task_id:<{width}}  {cfg}")
