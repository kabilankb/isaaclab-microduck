# P3 — BAM actuator: findings before implementation (2026-08-28)

Notes gathered while reading `bam/mjlab.py` against the Isaac Lab / Newton APIs. These
decide the architecture of the port, so they are recorded before any code is written.

## BAM needs generalized forces that Isaac Lab's Newton backend does not expose

`BamActuator.compute()` is not a pure function of (position, velocity, target). Its
load-dependent friction terms — which is the whole point of the m3–m6 models, and
Microduck runs **m6** — need the external load on the gearbox:

```python
external_torque = -qfrc_bias + qfrc_constraint - qfrc_friction
```

plus `qfrc_actuator` (the torque applied on the previous solve) as the motor-side load,
and an `efc` scan to subtract the friction constraint BAM itself injected last step.

On the Isaac Lab side:

- `ArticulationData.gravity_compensation_forces` — **raises `NotImplementedError`** on
  the Newton backend ("Newton has no gravity-compensation primitive"), tracked upstream
  as newton#2497 / #2529 / #2625.
- There is no `qfrc_bias`, `qfrc_constraint`, or constraint-row (`efc`) equivalent
  anywhere in the Newton `ArticulationData`.

So a BAM port written against Isaac Lab's articulation API cannot reproduce the m6
friction model. It would silently degrade to something close to m1 (Coulomb only), which
is exactly the sim2real fidelity the BAM actuator exists to provide.

## The way through: target the MuJoCo-Warp model underneath

Isaac Lab's Newton MJWarp backend **is** mujoco_warp, and it keeps the mjwarp objects
reachable:

```python
NewtonMJWarpManager._solver.mjw_model   # mujoco_warp Model
NewtonMJWarpManager._solver.mjw_data    # mujoco_warp Data
```

Those are the same objects mjlab's `BamActuator` already reads (`qfrc_bias`,
`qfrc_constraint`, `qfrc_actuator`, `efc.type/id/force`, `nefc`) and writes
(`dof_frictionloss`, `dof_damping`). So the port is close to line-for-line rather than a
reimplementation, and it keeps the property that MuJoCo's own constraint solver performs
the static-friction clipping (BAM Algorithm 1) instead of the actuator adding a passive
friction torque.

**Architecture decision: the BAM actuator subclasses Isaac Lab's `ActuatorBase` for its
lifecycle and Isaac Lab-facing API, but reads and writes the mjwarp `Model`/`Data`
directly for the physics.** This binds the port to `physics=newton_mjwarp` — already the
chosen backend, and the reason it was chosen.

Fallback if that access ever breaks: `write_joint_friction_coefficient_to_sim_*` maps to
Newton's `Model.joint_friction`, documented as a dry-friction torque limit in N·m — the
right target for the friction budget — but it does not solve the missing `qfrc_*` inputs.

## Two things already confirmed working

- `bam.model.load_model(motor_name="xl330", model="m6")` imports and loads standalone in
  `env_isaaclab`, with no mjlab in `sys.modules` (P0).
- Explicit actuators work on this asset; implicit ones do not (P2, trap 1). BAM is
  explicit, so that limitation never binds.

## Per-env field expansion

mjlab requires `sim.expand_model_fields(("dof_frictionloss", "dof_damping"))` before BAM
can write per-environment friction — a non-expanded field aliases one `(1, nv)` buffer
with stride 0 on the world axis, so per-env writes would be silently wrong. mjlab's BAM
raises a clear error for exactly this. **The Isaac Lab port needs the equivalent check**:
verify the mjwarp field is per-world before writing, and fail loudly if it is not.
