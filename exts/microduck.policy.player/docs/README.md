# Microduck Policy Player

Isaac Sim (Kit) extension: **attach to a running Microduck session and hot-swap the
policy driving it**, without restarting the simulator.

## It attaches — it does not create the environment

This is a hard constraint, not a design preference. `SimulationContext` is a singleton:

```python
# isaaclab/sim/simulation_context.py
if cls._instance is not None:
    return cls._instance          # the cfg you passed is DISCARDED
```

and `ManagerBasedEnv.__init__` builds its sim with `SimulationContext(self.cfg.sim)`.
Inside a running Kit session a context already exists, so an env created there binds to
the app's context and the task's **Newton MJWarp configuration is silently ignored**.
Isaac Lab environments must own the app lifecycle (`AppLauncher -> env -> loop`).

An earlier version of this extension called `gym.make()` and could not work for exactly
that reason.

## Use

**1. Start the session** (this creates the env correctly, and starts Kit):

```bash
python scripts/play.py --task=Isaac-BallKick-Flat-MicroDuck-Play-v0 \
    --checkpoint $PWD/logs/rsl_rl/microduck_ball_kick/<RUN>/model_5999.pt \
    --num_envs=4 --visualizer kit
```

**2. Enable the extension** inside that Kit window:

```
Window > Extensions > (gear) > Extension Search Paths > +
    /path/to/isaaclab-microduck/exts
```

Extension id: **`microduck.policy.player`** (shown as *Microduck Policy Player*).

**3. In the panel:** press **Attach**, enter a run directory or an
`exported/policy.pt`, press **Load**, then **Play**.

## Why hot-swapping is the useful part

Swapping policies against one already-running env is a rehearsal for what the real
runtime does: it hot-swaps walk / stand / trick ONNX files against a single shared 61D
observation buffer. Comparing two checkpoints on the same physics, without a restart
between them, is also the honest way to A/B them.

## It wants `exported/policy.pt`, not `model_*.pt`

The exported TorchScript has the **observation normalizer baked in**. A raw checkpoint
does not, and an unnormalized policy behaves like a different robot — invisible in
simulation, because in-sim play applies the normalizer anyway, so it only appears on
hardware. The extension rejects `model_*.pt` with that message rather than loading
something that looks fine and is wrong.

## Layout

| File | Role |
|---|---|
| `runner.py` | attach + policy. Imports **no** `omni.*`, so it is unit-tested in ordinary CPU CI (`tests/test_policy_player_extension.py`, 10 tests). |
| `extension.py` | Kit UI only. Steps from the Kit update stream — a `while` loop here would freeze the viewport it is meant to render. |

Keep that split. If `runner.py` grows a Kit dependency, the only way left to test this
extension is launching Isaac Sim by hand.

## Known limits

- `find_live_env()` locates the env by scanning live objects. Isaac Lab's `play.py`
  creates it and this package has no hook into that script; a global that only our own
  scripts set would miss exactly the case that matters. If several envs exist, the
  first is used.
- The UI has not been exercised inside Kit end-to-end — the runner half is tested, the
  panel is not. Expect to iterate on it.
