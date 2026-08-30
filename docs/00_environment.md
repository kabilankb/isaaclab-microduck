# P0 — Isaac Lab environment & pipeline validation (2026-08-28)

Everything below was executed and verified on this workstation.

## Versions (pin these)

| Component | Version |
|---|---|
| GPU / driver | NVIDIA RTX PRO 5000 Blackwell, 24 GB — driver 595.84, CUDA 13.2, x86_64 |
| Isaac Lab | 3.0.0 (`v3.0.0-beta2.patch1-18-g72cb3826d`), source install at `/home/chronos/IsaacLab` |
| Isaac Sim | 6.0.1.0 (pip) |
| Python env | conda `env_isaaclab`, Python 3.12.14 (`/home/chronos/miniconda3/envs/env_isaaclab/bin/python`) |
| newton | 1.2.1 |
| mujoco-warp | 3.8.0.3 |
| warp-lang | 1.13.0 |
| torch | 2.10.0+cu128 |
| rsl-rl-lib | 5.0.1 |
| skrl | 2.1.0 |
| better-actuator-models (bam) | 1.0.1, git `Rhoban/bam@mjlab_frictionloss` — installed 2026-08-28 |

Isaac Lab 3.0.0-beta2 is a **pre-release** and pairs here with Isaac Sim 6.0.1, ahead
of the `isaac-lab` skill's documented 4.5/5.0/5.1 window. It works; re-verify the
pairing before any upgrade.

**Two Python environments live in this repo — never cross-activate them:**
- `.venv` (uv, py3.12, torch 2.9.1+cu128) → the mjlab stack. Unchanged by this port.
- conda `env_isaaclab` (py3.12, torch 2.10.0+cu128) → the Isaac Lab stack.

## Verified gates

### 1. Task registry loads — 224 tasks

```bash
/home/chronos/miniconda3/envs/env_isaaclab/bin/python \
  /home/chronos/IsaacLab/scripts/environments/list_envs.py
```

Never write task IDs from memory; this is the source of truth.

### 2. Prebuilt task trains on the Newton MJWarp solver — PASSED

```bash
cd /home/chronos/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Cartpole-Direct-v0 --num_envs=16 --max_iterations=10 \
  physics=newton_mjwarp --headless
```

exit 0, 10/10 iterations, ~860 steps/s, mean reward 7.55 → 8.25, training time 4.18 s.

### 3. BAM imports standalone — PASSED

```bash
python -c "import bam.model; print(bam.model.load_model(motor_name='xl330', model='m6'))"
```

`bam` base deps are only `numpy` + `colorama`; the zmq/protobuf trouble documented in
this repo's `pyproject.toml` lives in the `identification` extra, which we do not install.
`bam.model` / `bam.actuator` import **without** mjlab — confirmed (`"mjlab" in sys.modules`
is `False`).

Note the `mjlab` extra pins `warp-lang<1.13` and `mujoco-warp<3.8`, both of which conflict
with this env (1.13.0 / 3.8.0.3). That extra is deliberately not installed, and it is a
second reason `bam.mjlab.BamActuator` cannot be reused here — the port reimplements the
voltage law against `bam.model` (P3).

## Traps confirmed on this machine

- **Checkpoints land under the EXPERIMENT NAME, not the task ID.**
  `Isaac-Cartpole-Direct-v0` → `logs/rsl_rl/cartpole_direct/2026-08-28_19-20-40/`.
  Any smoke test must assert on the experiment-name path.
- **`scripts/reinforcement_learning/rsl_rl/train.py` is deprecated** in 3.0 in favour of
  `./isaaclab.sh train --rl_library rsl_rl --task <TASK>`. Both work today; prefer the
  `isaaclab.sh train` form in docs and CI.
- **rsl-rl 5.0.1 wants an explicit `obs_groups` dict** with `actor` and `critic` keys.
  It falls back to a `policy` group with a deprecation warning. Our cfgs must declare
  `obs_groups` explicitly (mjlab already uses actor/critic groups).
- Benign headless warnings on a clean run: materialx, `usd_config`, Newton
  "shape color replacement" FutureWarning, and cartpole's own inertia-tensor warnings.

## GPU sharing

An mjlab run (`Mjlab-BallKick-Flat-MicroDuck`, 4096 envs, PID 4499) was training on the
same card during this validation, using ~6.5 GB and ~90 % util. Isaac Lab smoke runs
coexist fine; expect contention on throughput numbers and do not size a large Isaac Lab
run against the free VRAM without checking `nvidia-smi` first.
