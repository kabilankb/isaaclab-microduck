#!/usr/bin/env bash
# One-command smoke test for the Isaac Lab port.
#
#   ./scripts/smoke.sh          # everything (needs the GPU; a few minutes)
#   ./scripts/smoke.sh --cpu    # CPU-only checks (seconds, no simulator)
#
# Runs the same gates the port is developed against, in dependency order, and
# stops at the first failure. Run this before any long training job.

set -uo pipefail

PY="${MICRODUCK_ISAACLAB_PYTHON:-/home/chronos/miniconda3/envs/env_isaaclab/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
CPU_ONLY=0
[ "${1:-}" = "--cpu" ] && CPU_ONLY=1

FAILED=0
step() {
  local name="$1"; shift
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "▶ $name"
  echo "──────────────────────────────────────────────────────────────"
  if "$@"; then
    echo "✔ $name"
  else
    echo "✘ $name (exit $?)"
    FAILED=1
    return 1
  fi
}

[ -x "$PY" ] || { echo "Python not found: $PY (set MICRODUCK_ISAACLAB_PYTHON)"; exit 1; }
echo "interpreter: $PY"
"$PY" -c "import isaaclab, isaaclab_microduck, bam; print('imports OK')" || exit 1

step "cfg tests (CPU)"            "$PY" -m pytest "$ROOT/tests" -q            || exit 1
step "task registration"          "$PY" "$HERE/list_envs.py"                  || exit 1
step "MuJoCo reference dump"      "$PY" "$HERE/dump_mjcf_reference.py"        || exit 1

if [ "$CPU_ONLY" = "1" ]; then
  echo; echo "CPU-only run: skipped asset parity and the training smoke."
  exit $FAILED
fi

# Asset parity. The three *_backlash models are EXPECTED to fail today: USD keeps
# one joint per body pair, so their backlash hinges are dropped on conversion
# (see docs/isaaclab_port/02_assets.md). They are listed separately rather than
# skipped, so the day they start passing is visible.
for model in walk allcollisions rollers; do
  step "asset parity: $model" "$PY" "$HERE/check_asset_parity.py" --model "$model" || true
done
for model in walk_backlash allcollisions_backlash rollers_backlash; do
  echo
  echo "▶ asset parity: $model (known-failing: backlash hinges dropped)"
  "$PY" "$HERE/check_asset_parity.py" --model "$model" >/dev/null 2>&1 \
    && echo "!! $model now PASSES — the backlash gap is fixed, update the docs" \
    || echo "   still failing as expected"
done

# Do NOT pass `physics=newton_mjwarp`: the env cfgs configure Newton directly, and
# Isaac Lab's preset selector rejects unknown presets before iteration 0.
step "training smoke (16 envs, 5 iters)" \
  "$PY" "$HERE/train.py" --task=Isaac-Velocity-Flat-MicroDuck-v0 \
  --num_envs=16 --max_iterations=5 --headless || true

echo
echo "──────────────────────────────────────────────────────────────"
[ "$FAILED" = "0" ] && echo "ALL GATES PASSED" || echo "SOME GATES FAILED (see above)"
exit $FAILED
