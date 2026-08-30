# Microduck Policy Player

Isaac Sim (Kit) extension: pick a trained Microduck checkpoint and watch it run in the
viewport, without dropping to a terminal.

## Enable

Point Kit at this repo's `exts/` folder, then enable **Microduck Policy Player**:

```
Window > Extensions > (gear) > Extension Search Paths > +
    /path/to/isaaclab-microduck/exts
```

The extension imports `isaaclab_microduck`, so that package must be installed into the
same interpreter Kit is running (`pip install -e . --no-deps`; see `SETUP.md`).

## Use

1. **Task** — only `-Play-v0` variants are listed. That is deliberate: they disable
   domain randomization and observation noise, and for locomotion they pin a real
   forward command instead of sampling one. With a sampled command roughly 1 in 10
   envs is told to stand still, which makes a working policy look broken.
2. **Exported policy** — point at either a run directory or `exported/policy.pt`.
3. **Environments** — 1/4/9/16. Playback only; see the warning below.
4. **Load**, then **Play**.

## It wants `exported/policy.pt`, not `model_*.pt`

The exported TorchScript has the **observation normalizer baked in**. A raw checkpoint
does not, and an unnormalized policy behaves like a different robot — a failure that is
invisible in simulation, because in-sim play applies the normalizer anyway, so it only
appears on hardware. The extension rejects `model_*.pt` with that message rather than
loading something that looks fine and is wrong.

`scripts/play.py` writes the export on every run:

```
logs/rsl_rl/<experiment>/<run>/exported/policy.pt
```

## Do not read playback as evaluation

The env counts here are for watching. The same config that converged at 4096 envs
failed completely at 32 on this project, so small-env behaviour is not evidence about a
policy. Use a headless run and the `Metrics/*` values for that.

## Layout

| File | Role |
|---|---|
| `runner.py` | env + policy. Imports **no** `omni.*`, so it is unit-tested in ordinary CPU CI (`tests/test_policy_player_extension.py`). |
| `extension.py` | Kit UI only. Steps from the Kit update stream — a `while` loop here would freeze the viewport it is meant to render. |

Keep that split. If `runner.py` grows a Kit dependency, the only way left to test this
extension is launching Isaac Sim by hand.
