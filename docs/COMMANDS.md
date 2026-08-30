# Isaac Lab port — commands

Two independent Python environments live in this repo. **Never cross-activate them.**

| stack | interpreter | notes |
|---|---|---|
| mjlab (baseline) | `.venv` via `uv run` | torch 2.9.1+cu128, unchanged by the port |
| Isaac Lab (port) | `/home/chronos/miniconda3/envs/env_isaaclab/bin/python` | torch 2.10.0+cu128 |

```bash
PY=/home/chronos/miniconda3/envs/env_isaaclab/bin/python
cd /home/chronos/microduck_rl/isaaclab_microduck
```

## Launch and test — the short version

```bash
./scripts/smoke.sh --cpu     # seconds, no GPU: cfg tests, registration, reference dump
./scripts/smoke.sh           # full chain, a few minutes: + asset parity + a training run
```

`smoke.sh` runs every gate in dependency order and stops at the first real failure. The
three `*_backlash` models are reported separately because they are **expected to fail**
today (see `02_assets.md`); if one starts passing, the script says so.

Override the interpreter with `MICRODUCK_ISAACLAB_PYTHON`.

## Individual steps

```bash
# Fast gates (CPU only, no simulator)
$PY -m pytest tests/ -q                      # 56 cfg-invariant tests
$PY scripts/list_envs.py                     # what this package registers

# Assets — rerun after ANY change to the MJCFs or meshes
$PY scripts/dump_mjcf_reference.py           # MuJoCo ground truth -> assets/reference/*.json
$PY scripts/convert_assets.py --force        # MJCF -> USD (needs Isaac Sim)
$PY scripts/check_asset_parity.py --model walk   # parity + settle gate (needs a GPU)

# Training / evaluation (all Isaac Lab flags pass straight through)
$PY scripts/train.py --task=<TASK> --num_envs=64 --max_iterations=5 \
      physics=newton_mjwarp --headless       # SMOKE TEST — always run first
$PY scripts/train.py --task=<TASK> --num_envs=4096 --max_iterations=6000 \
      physics=newton_mjwarp --headless
$PY scripts/play.py  --task=<TASK> --checkpoint <model_XXXX.pt> physics=newton_mjwarp
```

Today the only registered task is the P1 placeholder `Isaac-Scaffold-MicroDuck-v0`
(Isaac Lab's cartpole under a Microduck id, proving external registration). Real
Microduck tasks arrive in P5. Always take task ids from `list_envs.py`, never memory.

## Seeing the robot itself

There is no Microduck TASK until P5, so a training run today shows Isaac Lab's cartpole
under a Microduck id. To look at the actual duck:

```bash
DISPLAY=:1 $PY scripts/view_robot.py --visualizer newton
```

It spawns the converted USD on a ground plane, holds HOME, and prints trunk height and
tilt once a simulated second until you close the window.

```bash
--model rollers      # or allcollisions, walk_backlash, ...
--num-envs 4         # several copies side by side
--drop 0.25          # spawn high and watch it land
--joint-noise 0.05   # perturb the start pose
--seconds 10         # stop automatically instead of running until closed
```

Actuation is the P2 conversion-check PD, **not** BAM — this shows the asset is right,
not how the real servos behave. Standing here should match `check_asset_parity.py`:
116.4 mm trunk z on the walk model, 139.6 mm on rollers.

## Keyboard teleop

**Run it in your own terminal** — keys are read from the TTY, so it does nothing when
backgrounded or piped (it says so and carries on):

```bash
DISPLAY=:1 $PY scripts/view_robot.py --visualizer newton --teleop
```

Keep the **terminal** focused and watch the viewer window.

```
  head      i / k   neck_pitch down / up        j / l   head_yaw left / right
            u / o   head_pitch down / up        n / m   head_roll left / right
  legs      w / s   stand / squat (hip+knee+ankle)
            a / d   lean left / right (hip_roll)
  joint     [ / ]   select previous / next joint
            up/down nudge the selected joint
  other     r       reset to HOME    h  help    q  quit
```

`--step 0.05` changes the per-press increment (default 0.02 rad ≈ 1.15°). Targets are
clamped to each joint's real limit.

**Why terminal keys and not Isaac Lab's `Se2Keyboard`:** Isaac Lab's keyboard devices go
through `carb` / `omni.appwindow`, so they only receive events under the **Kit**
visualizer. Reading the TTY works under `--visualizer newton`, headless, and over SSH —
and it is what `scripts/infer_policy.py` already does in the mjlab stack, so the two
stacks feel the same. The reader is ported from there into
`isaaclab_microduck/utils/terminal_input.py`.

**What this is not.** This drives **joint targets**, not a policy. Teleoping a *walking*
robot means sending twist commands to a trained locomotion policy, which needs the
velocity task (P5) and a trained checkpoint — neither exists yet in either stack (the
mjlab velocity run was stopped at ~iteration 70, so the only checkpoint on disk is
untrained). Once P5 lands, the same key reader drives the 13D command block instead.

## Watching training in a GUI

Drop `--headless` and pick a visualizer. Verified on this machine (`DISPLAY=:1`), a
Newton viewer window opens and training keeps running at ~0.85 s/iter:

```bash
DISPLAY=:1 $PY scripts/train.py --task=Isaac-Scaffold-MicroDuck-v0 \
      --num_envs=16 --max_iterations=400 \
      physics=newton_mjwarp --visualizer newton
```

`--visualizer` (alias `--viz`) takes a comma-separated list, no spaces. Valid names:
**`kit`** (full Isaac Sim UI — heaviest, RTX rendering), **`newton`** (Newton's own
OpenGL viewer — light, works kit-less, the one verified here), **`rerun`**, **`viser`**
(browser-based), **`none`**.

```bash
--visualizer kit            # full Isaac Sim editor UI
--visualizer newton,rerun   # several at once
--max_visible_envs 16       # cap how many envs are drawn in a big run
```

Rules of thumb:

- **Keep `--num_envs` small when watching** (16-64). Rendering is not free, and the point
  of a GUI run is to see behaviour, not throughput. Train for real with `--headless` at
  4096 envs.
- To watch a **trained** policy rather than a training run, use `play.py` with the same
  visualizer flags and a `--checkpoint` — that is the closer analogue of the mjlab stack's
  `play`, which is also where its live viewers live.
- Over SSH, set `DISPLAY` to the machine's local X display (`:1` here). Without a display,
  omit the visualizer and record video post-hoc instead.

## Where things land

- Runs: `isaaclab_microduck/logs/rsl_rl/<experiment_name>/<timestamp>/` — the
  **experiment name**, never the task id. Override the root with
  `MICRODUCK_ISAACLAB_LOG_ROOT`.
- USD: `isaaclab_microduck/isaaclab_microduck/assets/usd/` (gitignored build artifact)
- Reference dumps: `.../assets/reference/*.json` (committed; a diff means the robot changed)

## Checking the mjlab baseline still works

```bash
cd /home/chronos/microduck_rl
uv run --with pytest pytest tests/           # 154 passed, 1 skipped
```

## Two exit-code traps on this stack

- Isaac Sim's `simulation_app.close()` terminates the process with status 0, so a script
  must exit **before** teardown or its failures vanish. `check_asset_parity.py` does.
- `uv sync` has been seen to report success with a broken venv (see
  `docs/local_sim_setup_2026-08-28.md`).

Verify the artifact, not the status.
