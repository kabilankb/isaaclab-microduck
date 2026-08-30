# P2 — Robot assets: MJCF → USD (2026-08-28)

Newton builds its physics model from USD (`ModelBuilder.add_usd`), so USD is the only
asset path into Isaac Lab — the MJCFs cannot be loaded directly. All 7 models
(walk / allcollisions / rollers, their 3 backlash twins, and the ball prop) are
converted by `scripts/convert_assets.py` through Isaac Sim's MJCF importer.

The MJCFs are read **in place** from `src/mjlab_microduck/robot/microduck/`; nothing is
copied, so both stacks stay on one source of truth. The USD output is a gitignored
build artifact — regenerate it, never hand-edit it.

```bash
python scripts/dump_mjcf_reference.py          # MuJoCo ground truth -> assets/reference/*.json
python scripts/convert_assets.py --force       # MJCF -> USD
python scripts/check_asset_parity.py --model walk
```

## The headline result: the conversion is faithful

Standing trunk height, 3 s holding HOME from ±0.05 rad noisy inits, identical explicit
PD gains (kp 5.0) on both sides:

| | trunk z | tilt |
|---|---|---|
| MuJoCo (`scene_walk.xml`) | 116.4 mm | 0.46° |
| Isaac Lab (Newton MJWarp, converted USD) | 116.4 mm | 1.94° mean / 3.31° max over 16 envs |

Mass matches to 0.00 %, joint limits to 1.2e-07 rad, armature to 3.1e-11.

**Standing trunk z is 116.4 mm.** It is a measured quantity — do not carry it across
model revisions, and re-measure under a trained policy rather than reusing this number
as a target.

## What survives conversion — better than expected

The plan assumed MuJoCo contact semantics would be lost. They are not. Isaac Sim 5.0+
uses the `mujoco-usd-converter` backend, which writes a layered USD including
`payloads/Physics/mujoco.usda`, and applies `MjcCollisionAPI` + `NewtonCollisionAPI` to
each collider carrying **`mjc:condim`, `mjc:priority`, `mjc:solref`, `mjc:solimp`,
`mjc:solmix`, `mjc:margin`, `mjc:gap`**, plus per-joint `mjc:armature`, `mjc:damping`,
`mjc:frictionloss`, `mjc:ctrlRange`, `mjc:forceRange`.

Dumping the MuJoCo model Newton finally builds (`MJWarpSolverCfg.save_to_mjcf`) confirms
it end to end: the soles arrive with `friction="1 0.005 0.0001" solref="0.02"
solimp="0.9 0.95 0.001"`, straight from the MJCF.

`save_to_mjcf` is the single most useful debugging tool found in this phase — it shows
exactly the MuJoCo model the solver runs, and it can be loaded in plain MuJoCo to
separate "the model is wrong" from "the solver is behaving differently".

Collision is correctly narrowed to the 5 colliding geoms (2 soles, 2 legs, power
support) out of 75 — `contype == 0 and conaffinity == 0` geoms are imported visual-only,
matching MuJoCo.

Still to re-author Isaac-Lab-side (not inherited): the runtime `CollisionCfg` from
`microduck_constants.FULL_COLLISION` — feet `condim` 3 / `priority` 1 / `friction` 1.0,
other collision geoms `condim` 1. mjlab applies those by editing the MjSpec at load
time, so they are not in the XML and cannot come through the converter.

## Four traps found, each costing real time

1. **`ImplicitActuator` does not work with this asset — and it looks exactly like broken
   collision.** The MJCF importer warns, for all 14 joints, *"Gain and bias prm arrays
   are not in the expected format ... physics drive stiffness and damping will not be
   created"*, and the MuJoCo model Newton builds has `nu = 0`. An implicit actuator
   delegates the PD law to those non-existent drives, so the robot is completely limp:
   it collapses, bodies end up below the ground plane, and every instinct says
   "collision didn't convert".
   It had converted fine. The decisive checks were (a) a control cube, which rested at
   exactly 100 mm, and (b) loading Newton's own generated MJCF in plain MuJoCo, which
   collapsed *identically* to Isaac Lab — same model, same result, so the difference was
   never the conversion.
   **The port uses explicit actuators throughout, which is what BAM is anyway**, so this
   never binds. The placeholder is `IdealPDActuatorCfg`.

2. **The MJCF's own servo gain cannot hold the robot in either stack.** `kp = 0.55` is an
   identified XL330 parameter for BAM to fit against, not a usable controller: plain
   MuJoCo collapses to ~178° tilt under it. So the P2 gate uses an explicitly
   non-physical check gain (5.0) applied identically to both stacks, and **the real
   equilibrium gate belongs to P3**, once BAM exists.

3. **Isaac Lab reorders the joints.** MuJoCo's canonical order is left leg → head →
   right leg; Isaac Lab's articulation order is **right leg → head → left leg**. This is
   the concrete reason AGENTS.md's "never hardcode joint indices" rule must hold in the
   port too — resolve every index by name at runtime.

4. **The importer renames a body that collides with a joint name.** Microduck has both a
   `neck_pitch` hinge and a `neck_pitch` body; the body becomes `neck_pitch_1` in USD.
   `reference.usd_body_name()` encodes the rule and the parity check verifies it per
   model, rather than trusting it.

5. **Isaac Sim's teardown swallows the exit code.** `simulation_app.close()` terminates
   the process with status 0, so anything after it never runs — the parity check reported
   **exit 0 while failing its own checks**, which would have let CI wave a broken asset
   through. Neither `raise SystemExit(1)` nor `os._exit(1)` placed *after* `close()`
   survives; both were tried. The fix is to flush and `os._exit()` **before** teardown.
   Verified: `walk_backlash` (failing) exits 1, `walk` (passing) exits 0.
   This is the same shape as the `uv sync` exit-code lie already recorded in
   `docs/local_sim_setup_2026-08-28.md` — on this stack, verify the artifact, never the
   status.

Also worth knowing: re-running the converter over a populated output directory writes a
**sibling `<stem>_1/`** and reports success, leaving the stale asset in place for
everything else to load. `convert_assets.py` now clears the target on `--force` and
raises if the importer wrote anywhere unexpected.

## A pre-existing mjlab bug this phase surfaced

`test_home_pose_covers_every_servo_exactly_once` fails against the mjlab HOME frame on
every backlash model, and the cause is in the **mjlab stack**, not the port.

`HOME_FRAME` in `src/mjlab_microduck/robot/microduck_constants.py` keys joints by
patterns like `.*hip_pitch.*`. mjlab resolves these with `re.match`, whose leading `.*`
happily consumes a `passive_` prefix — so on the backlash models the pattern also matches
`passive_left_hip_pitch_backlash`. Verified directly against mjlab's own `resolve_expr`:

```
backlash joints given a NON-ZERO init position by HOME_FRAME: 10/14
  passive_left_hip_pitch_backlash  init=-0.4579 rad  limit=[-0.0175, 0.0175]  -> OUT OF RANGE
  passive_left_ankle_backlash      init=+0.4530 rad  limit=[-0.0175, 0.0175]  -> OUT OF RANGE
  ...  9 of the 10 land outside the hinge limits
```

Every episode of every `-Backlash-` task therefore starts with its ±1° gear-play hinges
up to 26× outside their range, and the solver snaps them back at each reset. Worth
fixing in the mjlab stack (anchor the patterns, or key by exact joint name).

The port sidesteps it: `HOME_JOINT_POS` uses **exact joint names**, and the test above
holds the line.

## Files

- `scripts/dump_mjcf_reference.py` → `isaaclab_microduck/assets/reference/*.json`
  (committed: small, and a diff means the robot changed)
- `scripts/convert_assets.py` → `isaaclab_microduck/assets/usd/**` (gitignored)
- `scripts/check_asset_parity.py` — parity + settle, the GPU half of the gate
- `isaaclab_microduck/robot/reference.py` — reads the dumps; joint order, limits, sites,
  contact params, body-name mapping
- `isaaclab_microduck/robot/microduck_cfg.py` — `ArticulationCfg` per model
- `tests/test_robot_cfg.py` — the CPU half of the gate

## Parity sweep — all six robot models

Explicit PD check gain (kp 5.0), 3 s hold at HOME, ±0.05 rad init noise, 16 envs:

| model | joints | bodies | mass | standing trunk z | tilt (mean / max) | result |
|---|---|---|---|---|---|---|
| walk | 14 ✓ | 15 ✓ | 737.24 g ✓ | 116.4 mm | 2.21° / 4.28° | **pass** |
| allcollisions | 14 ✓ | 15 ✓ | 737.24 g ✓ | 116.4 mm | 1.97° / 3.40° | **pass** |
| rollers | 18 ✓ | 19 ✓ | 736.83 g ✓ | 139.6 mm | 1.80° / 3.18° | **pass** |
| walk_backlash | 14 of 28 ✗ | 15 ✓ | 737.24 g ✓ | 116.4 mm | 2.04° / 3.57° | **fail** (see below) |
| allcollisions_backlash | 14 of 28 ✗ | 15 ✓ | 737.24 g ✓ | 116.4 mm | 2.01° / 3.05° | **fail** |
| rollers_backlash | 18 of 32 ✗ | 19 ✓ | 736.83 g ✓ | 139.6 mm | 1.49° / 4.13° | **fail** |

Standing heights: **116.4 mm** on the walk/allcollisions family, **139.6 mm** on the
roller models (the wheels lift the trunk). Measured, not assumed.

## Open gap: backlash models lose all 14 backlash hinges

The three `*_backlash` models convert and stand, but **every `passive_*_backlash` hinge
is missing from the USD** — silently, with no importer warning. The parity check catches
it; nothing else would.

Cause: `add_backlash.py` injects the play hinge as a **second joint on the same body** —

```xml
<body name="yaw2roll" ...>
  <joint name="left_hip_yaw" type="hinge" class="chosen_actuator"/>
  <joint name="passive_left_hip_yaw_backlash" type="hinge" class="backlash"/>
```

MuJoCo allows several joints per body; a USD/PhysX articulation allows exactly one joint
between a given pair of bodies, so the importer keeps the servo hinge and drops the play
hinge.

**Fix (not yet implemented):** preprocess the MJCF before conversion, inserting a
massless intermediate body between parent and child to carry the backlash hinge — the
standard single-joint-per-link encoding, physically equivalent and USD-representable.
That belongs with the backlash task variants, which are last in the P6 order, and it is
the same "do MjSpec surgery before conversion" pattern the plan already anticipates for
`spec_fn`.

Until then: the three `*_backlash` models are **rigid twins of their base model**, not
backlash models. Do not use them for a backlash A/B — the parity check fails loudly on
purpose, and `check_asset_parity.py` must pass before any backlash task is trained.
