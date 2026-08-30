# isaaclab_microduck

Isaac Lab 3.0 (Newton MJWarp) reinforcement-learning environments for **Microduck** — an
~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos.

Policies train at 50 Hz, export to ONNX, and are deployed by the runtime in
[`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck). This is the
Isaac Lab port of the mjlab stack that lives alongside it; **mjlab remains the
behavioural baseline** every ported task is A/B'd against.

## Tasks

| Task id | What it is |
|---|---|
| `Isaac-Velocity-Flat-MicroDuck-v0` | commanded-twist locomotion (the shared base recipe) |
| `Isaac-Running-Flat-MicroDuck-v0` | locomotion with a speed curriculum and a demanded flight phase |
| `Isaac-BallKick-Flat-MicroDuck-v0` | kick a 70 mm / 15 g ball forward from a standing start |
| `Isaac-BallRally-Flat-MicroDuck-v0` | two ducks, one ball, passed back and forth |
| `Isaac-RunParallel-Flat-MicroDuck-v0` | two ducks running side by side with a frozen pacer |

Every task has a `-Play-v0` twin for viewing. `python scripts/list_envs.py` is the live
registry — take task ids from it, never from memory.

## Design invariants

These are not style preferences; each exists because breaking it produced a policy that
worked in the viewer and failed on hardware.

- **The actor observation is 61D across the whole policy family**, so the runtime can
  hot-swap ONNX files against one buffer: 48 proprioception + a 13D command block
  `[twist(3), head_pose(4), body_pose(6)]`. A task that drives no head or body command
  ZERO-PADS those slots rather than removing them.
- **Two-robot tasks use a frozen partner, not a shared network.** A net driving both
  ducks would double the observation and make the policy undeployable. See
  `tasks/frozen_partner_env.py`.
- **Measure behaviour, not reward.** Reward totals rise on the standing stack while the
  task never happens — `Metrics/kick_success_rate`, `Metrics/rally_passes` and
  `Metrics/speed_tracking_ratio` exist because that failure mode cost whole runs here.
- **Unactuated joints are named `passive_*`** and every selector uses `^(?!passive_).*`.

## Status

BallKick and BallRally are trained and measured. Locomotion is **not working**: the
policies converge to standing still, and the cause is still under investigation — see
`docs/`. Treat everything here as a **sim milestone, not a deployable
policy**: the BAM actuator, observation delays and encoder bias are not yet ported.

## Setup

Full instructions, pinned versions and known traps: **[SETUP.md](SETUP.md)**.

## Environment

This package runs in the conda env `env_isaaclab` (Python 3.12, torch 2.10.0+cu128),
**not** the repo's uv `.venv` (torch 2.9.1). Never cross-activate them.

```bash
PY=/home/chronos/miniconda3/envs/env_isaaclab/bin/python
$PY -m pip install -e . --no-deps      # Isaac Lab/Sim/torch already provisioned
```

`--no-deps` is deliberate: Isaac Lab and Isaac Sim are editable source installs from
`$ISAACLAB_PATH`, and letting pip re-resolve them fights that install. `pyproject.toml`
documents the pinned versions this port was developed against.

## Commands

**Everything at once** — run this before any long training job:

```bash
./scripts/smoke.sh          # full chain (needs the GPU, a few minutes)
./scripts/smoke.sh --cpu    # CPU-only gates (seconds, no simulator)
```

Individual steps:

```bash
# Assets (P2) — rebuild after any MJCF change
$PY scripts/dump_mjcf_reference.py           # MuJoCo ground truth -> assets/reference/*.json
$PY scripts/convert_assets.py --force        # MJCF -> USD (needs Isaac Sim)
$PY scripts/check_asset_parity.py --model walk   # parity + settle gate (needs a GPU)

$PY scripts/list_envs.py                     # what this package registers (no sim launch)
$PY scripts/train.py --task=<TASK> --num_envs=64 --max_iterations=5 \
      physics=newton_mjwarp --headless       # SMOKE TEST — always run first
$PY scripts/play.py  --task=<TASK> --checkpoint <model_XXXX.pt> physics=newton_mjwarp
$PY -m pytest tests/
```

`train.py` / `play.py` are thin wrappers: they register the Microduck tasks through
Isaac Lab's supported `--external_callback` hook and exec Isaac Lab's own scripts, so
every upstream flag works unchanged. Isaac Lab is **not** forked or edited.

**To look at the robot itself** (no Microduck task exists until P5):

```bash
DISPLAY=:1 $PY scripts/view_robot.py --visualizer newton
DISPLAY=:1 $PY scripts/view_robot.py --model rollers --num-envs 4 --drop 0.25 --visualizer newton
```

**Keyboard teleop** of the joint targets (run in a real terminal — keys come from the
TTY, not the viewer window, so it works under any visualizer and over SSH):

```bash
DISPLAY=:1 $PY scripts/view_robot.py --visualizer newton --teleop
```

`i/k j/l u/o n/m` head · `w/s` squat · `a/d` lean · `[ ]` select joint · arrows nudge ·
`r` reset · `q` quit. This drives joint targets, **not** a policy — teleoping a walking
robot needs the velocity task (P5) plus a trained checkpoint.

**To watch training in a GUI**, drop `--headless` and add a visualizer (`kit`, `newton`,
`rerun`, `viser`):

```bash
DISPLAY=:1 $PY scripts/train.py --task=<TASK> --num_envs=16 --max_iterations=400 \
      physics=newton_mjwarp --visualizer newton
```

Keep `--num_envs` small when rendering; train for real headless at 4096.

`physics=newton_mjwarp` selects the Newton MuJoCo-Warp solver — the same solver family
mjlab uses, which is why the port targets it.

## Traps

- Checkpoints land under the **experiment name**, not the task ID:
  `$ISAACLAB_PATH/logs/rsl_rl/<experiment_name>/<timestamp>/`.
- Task IDs change between Isaac Lab releases — list them, never recall them.
- `play.py`'s automatic ONNX export is not the sim2real hand-off; use `scripts/export.py`,
  which bakes the observation normalizer in.
- **`ImplicitActuator` does not work with this robot** — the converted USD has no physics
  drives, so the robot goes completely limp and it looks exactly like broken collision.
  Use explicit actuators (BAM is one). See `docs/isaaclab_port/02_assets.md`.
- The three `*_backlash` models currently convert **without their backlash hinges** (USD
  allows one joint per body pair; MuJoCo does not). `check_asset_parity.py` fails on them
  on purpose — do not train a backlash task until that is fixed.
