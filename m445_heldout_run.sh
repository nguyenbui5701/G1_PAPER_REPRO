#!/usr/bin/env bash
# M4.45 — Predeclared fresh-reset-seed checkpoint comparison (diagnostic only)
# Put in /home/labrobot/nguyenbui/humanoid/G1_PAPER_REPRO
# Run: bash m445_heldout_run.sh first   (seeds 7001,7075)
# Then: bash m445_heldout_run.sh second (seeds 7149,7186)
set -Eeuo pipefail

SCOPE="${1:-first}"
case "$SCOPE" in
  first)  SEEDS=(7001 7075) ;;
  second) SEEDS=(7149 7186) ;;
  *) echo 'Usage: bash m445_heldout_run.sh first|second' >&2; exit 2 ;;
esac

BASE=/home/labrobot/nguyenbui/humanoid
M37="$BASE/G1_M3_7_G1_MJX_GPU_ONE_STEP_V1"
PY="$M37/.venv_m37_jax0435_mjx327/bin/python"
REPO="$M37/source/mujoco_playground_2025"
E="$BASE/G1_M3_11E_PPO_CLOUDPICKLE_OVERLAY_V1/overlays/cloudpickle_20261107_111524_3962615"
D="$BASE/G1_M3_11D_PPO_GYM_ISOLATED_REPAIR_V1/overlays/wheels_20261107_111000_3952924"
C="$BASE/G1_M3_11C_PPO_MANIFEST_REPAIR_V1/overlays/wheels_20261107_110316_3940764"
ROOT="$BASE/G1_PAPER_REPRO"
cd "$ROOT"

[[ -f m443_checkpoint_screen.py && -x "$PY" ]] || {
  echo 'Missing m443_checkpoint_screen.py or pinned Python executable' >&2; exit 1;
}
"$PY" -m py_compile m443_checkpoint_screen.py
"$PY" m443_checkpoint_screen.py --mode inventory
mkdir -p m445_logs
shopt -s nullglob

for SEED in "${SEEDS[@]}"; do
  for STEP in 320864256 374341632 401080320; do
    found=(m443_checkpoint_screen/m443_step${STEP}_seed${SEED}_*.json)
    if ((${#found[@]} > 0)); then
      echo "Refusing duplicate step=${STEP} seed=${SEED}: ${found[*]}" >&2
      echo 'Verify existing report; do not combine duplicated runs in comparison.' >&2
      exit 1
    fi
    LOG="m445_logs/m445_seed${SEED}_step${STEP}.log"
    [[ ! -e "$LOG" ]] || { echo "Refusing to overwrite log: $LOG" >&2; exit 1; }
    echo "M445_START scope=$SCOPE seed=$SEED step=$STEP"
    env -u LD_LIBRARY_PATH -u PYTHONHOME \
      PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda \
      JAX_DEFAULT_MATMUL_PRECISION=highest \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH="$E:$D:$C:$REPO:$ROOT" \
      "$PY" m443_checkpoint_screen.py \
        --mode run --step "$STEP" --seed "$SEED" \
        --cases stand,backward --magnitudes 0,1.0 --max-steps 1000 \
      2>&1 | tee "$LOG"
    echo "M445_DONE seed=$SEED step=$STEP"
  done
done

echo "M445_BATCH_COMPLETED scope=$SCOPE; all results in m443_checkpoint_screen/"
echo 'For final comparison, include ONE JSON per (checkpoint step, reset seed).'
