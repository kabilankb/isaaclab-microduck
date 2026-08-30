# Setup — Isaac Lab 3.0 + Newton for `isaaclab_microduck`

Portable version of the environment this package was developed against. Nothing here
assumes a particular machine; the only hard requirements are an NVIDIA GPU and
Python 3.12.

## 1. Requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Linux x86_64 | Isaac Sim 6.0.1 ships no aarch64 wheel |
| GPU | NVIDIA, >= 8 GB VRAM | 4096 envs of the two-duck tasks peaked at ~6.5 GB |
| Driver | >= 535 | needs CUDA 12.8+ runtime support |
| RAM | >= 32 GB | Isaac Sim alone is heavy before any envs exist |
| Disk | ~40 GB | Isaac Sim + extension cache dominate; USD build adds ~1 GB |
| Python | **3.12 only** | Isaac Lab 3.0 / Isaac Sim 6.0.1 are 3.12-exclusive |

### Reference system

Everything in this repo was developed and measured on this machine. Numbers quoted
elsewhere (throughput, VRAM, iteration times) come from here — treat them as a
reference point, not a spec.

| | |
|---|---|
| Machine | Dell Pro Max 16 Plus MB16250 (laptop), firmware 3.1.1 |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic, x86_64 |
| CPU | Intel Core Ultra 9 285HX — 24 cores / 24 threads |
| RAM | 125 GiB |
| GPU | NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, 24 GB |
| Driver / CUDA | 595.84 / CUDA 13.2 |
| Disk | 1.9 TB NVMe |

It is worth stating plainly that this is a **laptop**, not a workstation or a cluster
node. The 4096-env runs quoted below are what a single mobile Blackwell GPU sustains;
thermal headroom, not VRAM, is the usual limit on a long run here.

Measured on that machine: **~1.3 s/iteration at 4096 envs headless** for the
single-robot tasks (~106k env-steps/s), ~1.6-2.4 s/iteration for the two-robot tasks,
and **3.3-6.5 GB VRAM**. A 6000-iteration run takes 1.5-3 hours.

### Verified software versions

Read live from the working environment, not copied from release notes:

| Package | Version |
|---|---|
| Python | 3.12.14 |
| isaacsim | 6.0.1.0 |
| isaaclab | 6.1.17 (repo `v3.0.0-beta2.patch1-18-g72cb3826d`) |
| isaaclab_tasks / isaaclab_rl / isaaclab_newton | 1.10.9 / 0.5.7 / 0.13.6 |
| newton | 1.2.1 |
| mujoco-warp | 3.8.0.3 |
| warp-lang | 1.13.0 |
| mujoco | 3.8.0 |
| torch | 2.10.0+cu128 (CUDA build 12.8) |
| rsl-rl-lib | 5.0.1 |
| better-actuator-models (imports as `bam`) | 1.0.1 |
| gymnasium / numpy | 1.2.1 / 2.3.1 |

Two things worth noting about that table. **torch is built against CUDA 12.8 while the
driver reports CUDA 13.2** — that is fine, the driver is backward compatible, and it is
the pairing that was actually tested. And **Isaac Lab 3.0.0-beta2 is a pre-release**
paired with Isaac Sim 6.0.1; it works, but re-verify the pairing before upgrading
either, since the two version lines move together.

## 2. Create the environment

```bash
conda create -n env_isaaclab python=3.12 -y
conda activate env_isaaclab

pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
```

## 3. Isaac Lab (source install)

A source install is required — the scripts call into `$ISAACLAB_PATH/scripts/`.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v3.0.0-beta2
./isaaclab.sh --install
export ISAACLAB_PATH=$PWD          # persist this in your shell profile
```

## 4. Newton + BAM

```bash
pip install newton==1.2.1 mujoco-warp==3.8.0.3 warp-lang==1.13.0
pip install "better-actuator-models @ git+https://github.com/Rhoban/bam@mjlab_frictionloss"
```

Install BAM **base only**. The `identification` extra pulls a broken `zmq==0.0.0` stub
and `protobuf<4`; the `mjlab` extra pins `warp-lang<1.13` / `mujoco-warp<3.8`, both of
which conflict with this environment. The distribution is `better-actuator-models`; the
import stays `bam`.

## 5. This package

```bash
git clone <this-repo> && cd <this-repo>/isaaclab_microduck
pip install -e . --no-deps
```

`--no-deps` is deliberate: Isaac Lab and Isaac Sim are editable source installs, and
letting pip re-resolve them fights that install. `pyproject.toml` documents the pins
rather than resolving them.

## 6. Build the robot assets

USD is a **build artifact** — gitignored, regenerated, never hand-edited. Newton builds
its physics model from USD, so the MJCFs cannot be loaded directly.

```bash
python scripts/dump_mjcf_reference.py     # MuJoCo ground truth -> assets/reference/*.json
python scripts/convert_assets.py --force  # MJCF -> USD (needs Isaac Sim)
python scripts/check_asset_parity.py --model walk
```

Parity should report **trunk z 116.4 mm** (walk) and **139.6 mm** (rollers), tilt < 2°.
Those are measured values for this robot; a mismatch means the conversion changed.

## 7. Verify

```bash
./scripts/smoke.sh --cpu    # seconds, no GPU: cfg tests, registration, reference dump
./scripts/smoke.sh          # full chain, minutes: + asset parity + a training run
python scripts/list_envs.py # what this package registers
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `ISAACLAB_PATH` | Isaac Lab source checkout | `/home/chronos/IsaacLab` |
| `MICRODUCK_ISAACLAB_PYTHON` | interpreter used by `smoke.sh` | the conda env's python |
| `MICRODUCK_ISAACLAB_LOG_ROOT` | where runs are written | the package root |

## Traps

- **Checkpoints land under the EXPERIMENT NAME, not the task id** —
  `Isaac-BallKick-Flat-MicroDuck-v0` writes to `logs/rsl_rl/microduck_ball_kick/`.
  Any automation must assert on the experiment-name path.
- **rsl-rl 5.0.1 needs an explicit `obs_groups`** dict with `actor` and `critic` keys.
  It silently falls back to a `policy` group otherwise. Our cfgs declare it, and the
  critic group must be named `privileged`.
- **Do not star-import `isaaclab.envs.mdp`.** It forces Isaac Lab's lazy loader to
  import `scene_data_provider` -> `pxr` (pip's USD) at cfg-import time, before Kit
  starts; Kit then loads its own USD copy and aborts in `libusd_tf.so` static init.
  That is what made `--visualizer kit` segfault on every task. Same for importing
  `Articulation` / runtime command classes at cfg-module level.
- **Isaac Sim's `simulation_app.close()` exits with status 0**, so a script must exit
  BEFORE teardown or its failures vanish. Verify the artifact, not the exit code.
- Benign headless warnings on a clean run: materialx, `usd_config`, Newton "shape color
  replacement" FutureWarning, and USD `material:binding:physics` scope warnings.

## GPU sharing

Two 4096-env runs will not co-exist comfortably on 24 GB. Check `nvidia-smi` before
sizing a run. Two Rerun-enabled processes also clash: the second binds nothing but
still prints a URL, so the link looks valid while no server is listening — confirm
with `ss -ltn | grep 9090`.
