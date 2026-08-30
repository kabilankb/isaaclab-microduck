# isaaclab_microduck

Isaac Lab 3.0 (Newton MJWarp) reinforcement-learning environments for **Microduck** — an
~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos.
Policies train at 50 Hz, export to ONNX, and are deployed by the runtime in
[`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck). This is the
Isaac Lab port of the mjlab stack that lives alongside it; **mjlab remains the
behavioural baseline** every ported task is A/B'd against.


https://github.com/user-attachments/assets/cf405f24-e72d-4768-9017-d1ea9c43c6e4


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

## Physics: where Newton enters

Newton is a **dependency, not vendored code** — it arrives with Isaac Lab
(`isaaclab_newton`) and is installed per [SETUP.md](SETUP.md). It touches this package
in exactly four places, and it is worth knowing all four before changing any of them:

| Where | What it does |
|---|---|
| `tasks/velocity/velocity_env_cfg.py` | **the one that matters** — `SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=10, ls_iterations=20)))`. Every task inherits this cfg, so this single line is what makes the whole package run on Newton. |
| `utils/arrays.py` | `as_torch()`. Under Newton, `ArticulationData` / `RigidObjectData` / sensor data are **not** plain torch tensors, so every MDP term that reads sim state goes through this bridge. |
| `scripts/convert_assets.py` | Newton builds its physics model from USD (`ModelBuilder.add_usd`), so the MJCFs must be converted first. This is why USD exists here at all. |
| `scripts/check_asset_parity.py` | builds its own Newton `SimulationCfg` to A/B the converted asset against MuJoCo ground truth. |

Timing is 200 Hz physics / 50 Hz control (`decimation=4`), matching the mjlab recipes.
Running the same MuJoCo-Warp solver is the point: it makes this a like-for-like port of
behaviours proven in mjlab rather than a re-tune against a different contact model.

### Do not pass `physics=newton_mjwarp`

The cfgs configure Newton **directly**, so the CLI preset is redundant — and Isaac Lab's
preset selector rejects unknown preset names outright:

```
ValueError: Unknown preset(s): newton_mjwarp
```

That kills the run before iteration 0. Isaac Lab's own docs and older Microduck notes
show this flag; it does not apply here.

### Newton vs the Newton *viewer*

Two different things that share a name, and confusing them costs debugging time:

- **Newton the physics backend** — always on, headless or not.
- **`--visualizer newton`** — Newton's own OpenGL window. Optional, and it draws from
  Newton's internal shape transforms rather than from USD. A USD-side bug can therefore
  look fine there and broken in Kit/RTX, which is exactly how the nested-rigid-body bug
  in `convert_assets.py` stayed hidden (see `docs/02_assets.md`).

## Usage

All commands run under the Isaac Lab interpreter. Set it once:

```bash
PY=$(which python)           # inside the activated env_isaaclab
cd isaaclab-microduck
```

### 1. List the environments

The live registry — take task ids from here, never from memory. Launches no simulator.

```bash
$PY scripts/list_envs.py
```

```
Isaac-Velocity-Flat-MicroDuck-v0        Isaac-Velocity-Flat-MicroDuck-Play-v0
Isaac-Running-Flat-MicroDuck-v0         Isaac-Running-Flat-MicroDuck-Play-v0
Isaac-BallKick-Flat-MicroDuck-v0        Isaac-BallKick-Flat-MicroDuck-Play-v0
Isaac-BallRally-Flat-MicroDuck-v0       Isaac-BallRally-Flat-MicroDuck-Play-v0
Isaac-RunParallel-Flat-MicroDuck-v0     Isaac-RunParallel-Flat-MicroDuck-Play-v0
```

### 2. Train

**Always smoke-test first.** 5 iterations at 64 envs takes seconds and catches most
config errors — a missing term, a wrong joint selector, a NaN — for almost nothing.

```bash
$PY scripts/train.py --task=Isaac-BallKick-Flat-MicroDuck-v0 \
      --num_envs=64 --max_iterations=5 --headless
```

Then the real run. Episodic tricks converge in ~1000-2000 iterations; gaits and
curriculum-heavy tasks want 4000-6000.

```bash
$PY scripts/train.py --task=Isaac-BallKick-Flat-MicroDuck-v0 \
      --num_envs=4096 --max_iterations=6000 --headless
```

Resume from a checkpoint:

```bash
$PY scripts/train.py --task=<TASK> --num_envs=4096 --max_iterations=6000 --headless \
      --resume --load_run 2026-08-29_11-14-48 --checkpoint model_5999.pt
```

**Use 4096 envs for anything you intend to believe.** Small-env runs are for watching:
the same config that converged at 4096 failed completely at 32 here, so treat
small-batch numbers as unusable for conclusions.

### 3. Play a trained policy

Use the `-Play-v0` twin — it disables domain randomization and observation noise, and
for locomotion it pins a real forward command instead of sampling one (a randomly
sampled command makes a working policy look broken, because some envs are told to
stand still).

```bash
$PY scripts/play.py --task=Isaac-BallKick-Flat-MicroDuck-Play-v0 \
      --checkpoint $PWD/logs/rsl_rl/microduck_ball_kick/<RUN>/model_5999.pt \
      --num_envs=4 --visualizer kit,newton,rerun
```

### 4. Inference / export

`play.py` writes TorchScript and ONNX to `<run>/exported/` on every run, with the
**observation normalizer baked in**:

```
logs/rsl_rl/<experiment>/<run>/exported/policy.pt         # TorchScript
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx       # ONNX graph
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx.data  # ONNX EXTERNAL WEIGHTS
```

**Copy `policy.onnx.data` alongside `policy.onnx`.** The weights are stored
externally (the `.onnx` is ~14 KB, the `.data` ~790 KB), so shipping the graph alone
silently produces a policy with no weights.

Load it standalone — 61 observations in, 14 joint targets out:

```python
import torch
policy = torch.jit.load("exported/policy.pt").eval()
actions = policy(torch.zeros(1, 61))        # -> (1, 14)
```

`onnxruntime` is declared in `pyproject.toml` but the `--no-deps` install does not
install it; `pip install onnxruntime` separately if you want to validate the ONNX.

**Never hand-convert a raw checkpoint.** Observation normalization is on, and an
unnormalized policy behaves like a different robot — a bug that is invisible in sim,
because in-sim play applies the normalizer anyway.

This is also how the two-duck tasks drive their second robot: `BallRally` and
`RunParallel` load an exported `policy.pt` as a frozen partner
(`tasks/frozen_partner_env.py`), which is what keeps them single-agent and the actor
observation at 61D.

### 5. Visualizers

```bash
--headless                            # no rendering; use this to train
--visualizer rerun                    # browser viewer, most reliable here, time scrubbing
--visualizer newton                   # Newton's own OpenGL window, light
--visualizer kit                      # full Omniverse RTX, heaviest, slow to boot
--visualizer kit,newton,rerun         # comma-separated, no spaces
--max_visible_envs 9                  # cap what is drawn in a large run
```

Rerun prints a `http://127.0.0.1:9090/...` URL. **Two Rerun-enabled runs clash**: the
second binds nothing but still prints a URL, so verify with `ss -ltn | grep 9090`.

### 6. Look at the robot without a policy

```bash
$PY scripts/view_robot.py --visualizer newton                 # holds HOME, prints z and tilt
$PY scripts/view_robot.py --model rollers --num-envs 4 --drop 0.25 --visualizer newton
$PY scripts/view_robot.py --visualizer newton --teleop        # keyboard joint teleop
```

Teleop reads the **TTY**, not the viewer window, so run it in a real terminal; it works
headless and over SSH. Actuation there is the conversion-check PD, not BAM.

### 7. Everything at once

```bash
./scripts/smoke.sh --cpu    # seconds, no GPU: cfg tests, registration, reference dump
./scripts/smoke.sh          # minutes: + asset parity + a real training run
$PY -m pytest tests/        # 149 cfg-invariant tests, CPU only
```

## Where things land

Runs are keyed by **experiment name, not task id** — `Isaac-BallKick-Flat-MicroDuck-v0`
writes to `logs/rsl_rl/microduck_ball_kick/<timestamp>/`. Any automation must assert on
the experiment-name path.

```
logs/rsl_rl/<experiment>/<timestamp>/
    model_*.pt          checkpoints (every 250 iterations)
    exported/           TorchScript + ONNX, normalizer baked in
    params/             the resolved cfg for this run
```

Override the root with `MICRODUCK_ISAACLAB_LOG_ROOT`.


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
