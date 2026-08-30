# P5 — Ball tasks: BallKick fixes, BallRally, and the Kit crash (2026-08-29)

Two reward bugs found by diffing the port against the mjlab recipe, a real success
metric for BallKick, a new two-duck BallRally task, and the root cause of the
`--visualizer kit` segfault (which was **our** import hygiene, not the install).

```bash
uv run list-envs                      # mjlab side, unchanged
$PY scripts/list_envs.py              # 6 Isaac tasks: Velocity / BallKick / BallRally (+ Play)
$PY -m pytest tests/ -q               # 110 passed
```

Everything below was measured on this machine, not carried over from mjlab.

---

## 1. BallKick had no curriculum at all

The port copied mjlab's reward WEIGHTS correctly but not its schedules. mjlab ramps
`action_rate_l2` from `-0.1` to `-1.0` between iterations 500 and 1500; the port
declared no `curriculum` field, so the weight sat frozen at the stage-0 value for
the whole run. Per AGENTS.md, smoothness is a motion-blocker during skill discovery
and a jitter-damper afterwards — freezing it at stage 0 never applies the discipline.

Fixed by porting mjlab's stages verbatim into `BallKickCurriculumCfg`. A curriculum
class that is never assigned to the cfg is a silent no-op, so
`tests/test_ball_kick_curriculum.py` asserts the wiring as well as the values.

**Still not ported** (mjlab has them, we do not): the `com_range`, `head_com_range`
and `push_magnitude` curricula. `push_robot` therefore runs at full ±0.3 from
iteration 0, where mjlab deliberately ramps it in from zero at iteration 500 — the
same timing bug class, still open.

## 2. The kick reward was 4× the recipe's own documented rule

This is the one that mattered. `microduck_ball_kick_env_cfg.py` documents:

```
line  93:  "weight ~= 3/target ... if you change the target, rescale the weights with it"
line  95:  BALL_TARGET_SPEED = 1.0
line 224:  "peaking at BALL_TARGET_SPEED (0.25 m/s - a gentle tap)"
line 226:  "Weight 12.0 = 3.0/target"        <- only true when target = 0.25
```

The target was raised `0.25 -> 1.0` and the weights were never rescaled. **This is a
bug in the mjlab baseline too**, inherited verbatim by the port.

Consequence: the kick paid **12/step** against an 8/step standing stack
(support 2 + upright 2 + height 1 + legs 2 + neck 1), so leaning over to strike
harder was simply the better trade. PPO took it, and `Episode_Reward/upright`
flatlined at **1.5% of its weight** while every other term climbed — the classic
compromise basin.

Fixed by expressing both weights against the target so they cannot drift apart
again:

```python
ball_forward_velocity   weight = 3.0 / TARGET_BALL_SPEED     # was 12.0
ball_speed_overshoot    weight = -1.0 / TARGET_BALL_SPEED    # was -4.0
```

`tests/test_ball_kick_reward_balance.py` locks the at-target payoff, the 3:1
asymmetry (net kick reward crosses zero at 4× target), and that the kick never
outweighs the whole standing stack.

### Result (4096 envs, 6000 iters, 1h55m, zero NaN)

| | before | after |
|---|---|---|
| `upright` | 0.03 of 2.0 (1.5%) | **1.83 of 2.0 (92%)** |
| `fell_over` | 4% | **1%** |
| `ball_forward_velocity` | mean 5.30, **sd 3.02**, range 0–11.98 | mean 2.02, **sd 0.05** |
| episode length | 234 / 250 | **250 / 250** |

The 50× drop in ball-speed variance is the headline: the kick went from thrashing
to calibrated. `ball_speed_overshoot` settles at −0.07 against a −1.0 weight, i.e.
the ball averages ~0.07 m/s above the 1.0 m/s target — the "gentle, controlled tap"
the recipe asks for.

### Batch size is doing heavy lifting

The same fixed cfg at **32 envs** did NOT converge: at matched iteration 3000 it had
4× less reward, 20× worse `upright` (0.08 vs 1.81) and 23× more falls, and ball speed
was still hunting (sd 0.24). Small-batch runs on these tasks are for WATCHING, not
for drawing conclusions.

## 3. `Metrics/success_rate` does not measure kicking

It comes from Isaac Lab's `UniformVelocityCommand` and scores twist tracking
(`vel_xy_success_threshold=0.5`, `vel_yaw_success_threshold=0.4`) against the ±0.01
command this task carries only for 61D obs parity. It read `0.0000` through a run
whose ball speed rose 100×, and flickered to `1.00` for a single iteration when the
robot happened to hold still. **It is actively misleading here.**

`mdp.kick_success` is the real criterion — ball travel ≥ `min_distance` projected on
the kick direction latched at reset, AND trunk within `max_tilt_deg` of vertical at
episode end (30°, far tighter than the 70° `fell_over` termination: that limit leaves
a wide band of "leaning but alive" a SCORE must not pass even though a TERMINATION
should).

It runs as a **curriculum term** because that is the only per-reset hook firing
BEFORE the reset events re-place the ball (`_reset_idx` calls
`curriculum_manager.compute()` at line 369, `event_manager.apply(mode="reset")` at
375). A test asserts that ordering in Isaac Lab's source, so an upstream swap fails
loudly instead of silently reading zero.

| | untrained (iter 8) | trained (`model_5999.pt`) |
|---|---|---|
| `Metrics/kick_distance` | 0.06 m | **0.99 – 1.81 m** |
| `Metrics/kick_success_rate` | 0.00 | **0.85 – 0.94** |
| `Metrics/success_rate` | 0.0000 | 0.0000 |

Quote those as RANGES: six consecutive evaluations of the same checkpoint spanned
0.85–0.94 success. `min_distance = 0.30 m` sits in the empty gap between random
bumping (0.06 m) and trained kicks (~1.8 m), so its exact value is not load-bearing
at this quality level — but it should be set from a distribution, not a mean, before
anyone leans on it. `Metrics/kick_distance` is logged threshold-free for exactly that.

## 4. BallRally — two ducks, one ball

`Isaac-BallRally-Flat-MicroDuck-v0`. Duck A learns; duck B replays a frozen BallKick
policy (`exported/policy.pt`, TorchScript with the obs normalizer **baked in** —
never point it at a raw checkpoint).

**Why a frozen partner and not one net driving both ducks:** AGENTS.md's hard
invariant is that the actor obs stays 61D across the whole policy family so policies
are hot-swappable in the runtime. A shared net would have doubled the observation and
made this task's policy undeployable, permanently. A test asserts the actor term list
matches BallKick's exactly.

The partner is driven by a `ManagerBasedRLEnv` subclass overriding `step()` — the
only hook in the right place. An action term would steal action dimensions and break
the 14-D contract; an event fires after physics rather than before.

**Anti-jackpot design**, because a rally pays per pass:

- `pass_completed` is **latched** — one crossing scores once, disarms, and re-arms
  only after the ball returns past halfway. Unlatched, a ball resting at the
  partner's feet would pay every step and the optimum would be to shove it there.
- `ball_progress_to_partner` is **potential-based** (Δgap): holding pays exactly
  zero and any closed round trip integrates to zero.
- `ball_lost` terminates at 2.0 m so a dead rally cannot keep collecting the
  standing stack.

`PARTNER_DISTANCE = 0.8 m` is set off the MEASURED kick (~1.8 m travel) — inside one
kick, beyond a nudge. Episode is 15 s; 5 s cannot contain an exchange.

### Result (4096 envs, in progress at iteration ~4200)

| window | passes/episode | max | `upright` | `fell_over` |
|---|---|---|---|---|
| 0–500 | 0.083 | 1.25 | 0.07 | 27% |
| 1000–1500 | 0.470 | 7.40 | 1.77 | 4.6% |
| 2500–3000 | **1.583** | **18.78** | 1.91 | **0.4%** |

Mean passes up ~19×, best iterations reaching 18 passes — a sustained rally. The
prediction that a frozen standing-kick partner caps this at 1–2 passes was **too
pessimistic**; the learner found placements the partner reliably returns.

Two caveats. The variance did NOT damp as the mean rose (max 18.8 vs mean 1.58), so
these are occasional long rallies, not consistent ones. And `rally_success_rate`
(~0.017) lags `rally_passes` badly, which says a small subset of episodes carries
most of the passes.

Any single iteration is meaningless here — roughly a quarter read exactly zero. Read
windows, not snapshots. Two independent measures agree
(`Episode_Reward/pass_completed ÷ 0.01333` reproduces `Metrics/rally_passes`), which
is the cross-check worth repeating.

**Next step for real rallies:** refresh the partner from a BallRally checkpoint —
self-play. The design supports it by swapping one path.

## 5. `--visualizer kit` segfaulted because of OUR imports

Symptom: `free(): invalid pointer` inside `libusd_tf.so` static init during
`omni.usd.libs` startup, exit 134/139, on every Microduck task. Isaac Lab's own tasks
were fine, including **cartpole under `physics=newton_mjwarp`** — so it was neither
"Kit is broken here" nor "Kit and Newton cannot coexist". Both of those were wrong
guesses before the import spy was run.

Real cause: task cfgs are imported during CLI **preset collection, BEFORE Kit
starts**. Three of our imports pulled pip's `usd_core` USD (`pxr`) into the process
at that point; Kit then loaded its own separately-built USD copy, and the second
one's static initializers freed a `std::string` from the first one's allocator.

| # | trigger | fix |
|---|---|---|
| 1 | `from isaaclab.envs.mdp import *` in `tasks/mdp/__init__.py` — a star-import forces every lazily-loaded attribute to evaluate, reaching `scene_data_provider` → `pxr` | PEP 562 module `__getattr__` forwarding lazily |
| 2 | `from isaaclab.assets import Articulation, RigidObject` in 5 mdp modules | moved into `TYPE_CHECKING` — annotation-only, and PEP 526 never evaluates local annotations |
| 3 | `commands.py` subclassing `UniformVelocityCommand` at module level | runtime classes split into `command_impl.py`, referenced by lazily-resolved string, exactly as Isaac Lab's own `commands_cfg.py` does |

Verified with an import spy: `pxr loaded after cfg import: False`. All three
visualizers now run together:

```bash
DISPLAY=:1 $PY scripts/train.py --task=Isaac-BallRally-Flat-MicroDuck-v0 \
    --num_envs=16 --max_iterations=3 --visualizer kit,newton,rerun     # exit 0
```

**This is now an import-hygiene contract, not a one-off.** Any future star-import of
Isaac Lab's mdp, or a runtime asset/command class imported at cfg-module level,
reintroduces the crash. The docstrings in all three files say so. A test asserting
`pxr` stays unloaded after a cfg import is the obvious guard and is NOT yet written.

## 6. Visualizer notes

| visualizer | status |
|---|---|
| `kit` (Omniverse) | works after §5. RTX UI takes minutes to boot; heaviest |
| `rerun` | most reliable. Webviewer at `http://127.0.0.1:9090/?url=rerun%2Bhttp://127.0.0.1:9876/proxy`; time scrubbing makes it the best tool for inspecting the instant of contact |
| `newton` | **intermittent** — renders at 9/16/32 envs sometimes, and the window comes up 1×1 other times with the identical command. Unexplained; an env-count ceiling was hypothesised and then falsified |

**Two Rerun-enabled runs cannot coexist**: the second binds nothing but still prints
a URL, so the link looks valid while no server is listening. Check with
`ss -ltn | grep 9090` before trusting it.

## Open work

- Port the `com_range` / `head_com_range` / `push_magnitude` curricula (§1).
- Add the `pxr`-stays-unloaded regression test (§5).
- Self-play partner refresh for BallRally (§4).
- Still absent vs the mjlab recipe: BAM actuator, observation delays, encoder bias,
  `angular_momentum`, `self_collisions`. **These are sim milestones, not deployable
  policies.**
- **Nobody has watched a kick frame by frame.** Every claim here is a number. The
  video check AGENTS.md asks for — which geom strikes the ball, toe versus shin —
  has not been done.
