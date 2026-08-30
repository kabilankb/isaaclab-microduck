# P1 — Package scaffold (2026-08-28)

`isaaclab_microduck/` is an **external Isaac Lab project**: it pip-installs against the
Isaac Lab checkout and registers its tasks with `gym.register`. Isaac Lab is never
forked or edited — the in-repo "internal task" path is only for upstreaming, and it is
auto-disabled whenever Isaac Lab is pip-installed.

## Install

```bash
PY=/home/chronos/miniconda3/envs/env_isaaclab/bin/python
cd isaaclab_microduck && $PY -m pip install -e . --no-deps
```

`--no-deps` is deliberate — Isaac Lab, Isaac Sim, torch, warp and rsl-rl are already
provisioned in `env_isaaclab` (mostly as editable installs from `$ISAACLAB_PATH`), and
re-resolving them fights that install. `pyproject.toml` records the pinned versions.

## How task registration reaches Isaac Lab

Isaac Lab's `train.py` exposes `--external_callback`, resolved with
`string_to_callable(..., separator=".")` — a **dotted** path, not the `module:attribute`
form used everywhere else in Isaac Lab. Importing the named module is what registers the
tasks; the callback returns `None`, meaning "consumed no CLI arguments".

`scripts/train.py` and `scripts/play.py` append
`--external_callback isaaclab_microduck.tasks.register_tasks` and then `execv` Isaac
Lab's own script, so every upstream flag keeps working.

## Two launcher traps, both hit and fixed here

1. **`cli_args` is a sibling import.** Isaac Lab's `train.py`/`play.py` do `import cli_args`
   from their own directory. The launcher puts that directory on `PYTHONPATH`.
2. **The log directory is resolved from the CWD.** The obvious fix for (1) — chdir into
   the script's directory — silently buried the first run in
   `$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/logs/`. The launcher chdirs to
   the **package root** instead, so runs land in
   `isaaclab_microduck/logs/rsl_rl/<experiment_name>/<timestamp>/`, next to the mjlab
   stack's `logs/`. Override with `MICRODUCK_ISAACLAB_LOG_ROOT`.

Remember the experiment-name trap: the subdirectory is the **experiment name**, never the
task ID. A smoke test asserting a task-ID-shaped path fails even when training succeeded.

## Verified gates

```bash
$PY scripts/list_envs.py      # -> 1 Microduck task registered
$PY -m pytest tests/          # -> 4 passed
$PY scripts/train.py --task=Isaac-Scaffold-MicroDuck-v0 --num_envs=16 \
      --max_iterations=5 physics=newton_mjwarp --headless
# -> exit 0; logs in isaaclab_microduck/logs/rsl_rl/cartpole/<timestamp>/ with model_*.pt
```

`Isaac-Scaffold-MicroDuck-v0` is a **placeholder** that reuses Isaac Lab's own cartpole
cfg purely to prove external registration end to end. **Delete it in P5** when the
velocity task lands.
