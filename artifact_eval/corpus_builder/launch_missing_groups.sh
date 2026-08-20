#!/bin/bash
# Kill-robust builders for the MISSING corpus groups: realistic-Inductor + flash_attention.
# Re-run to RESUME after any kill/reboot (per-config JSON checkpoints are skipped). This
# launcher is idempotent: a unit that is already active is left running (resume is automatic),
# so re-running never disturbs an in-flight build.
# systemd --user units (survive Claude session teardown; MemorySwapMax=0 => oomd-safe;
# MemoryMax => won't crash the box). One GPU each. Each build script HARD-EXITs (os._exit(3))
# on a per-config timeout / CUDA-poison, so `|| exit 3` makes Restart=on-failure resume past it.
#
# Needs systemd --user. Without it, run the same two commands by hand -- see corpus_builder/README.md.
set -u
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # artifact_eval/corpus_builder
TRITON=$(cd "$BASE/../.." && pwd)                    # repo root; bitequiv is not installed
PY=${PY:-$TRITON/.venv/bin/python}
LOGS=${LOGS:-$BASE/logs}
mkdir -p "$LOGS"

launch() {  # $1=gpu $2=unit $3=log $4=script $5=extra_args
  if systemctl --user is-active "$2" >/dev/null 2>&1; then
    echo "[$2] already active -- leaving it running (resume is automatic)"; return
  fi
  systemctl --user reset-failed "$2" 2>/dev/null
  systemd-run --user --unit="$2" -p MemorySwapMax=0 -p MemoryMax=40G \
    -p Restart=on-failure -p RestartSec=5 \
    --setenv=TRITON_ALWAYS_COMPILE=1 --setenv=CUDA_VISIBLE_DEVICES="$1" \
    --setenv=PYTHONPATH="$TRITON:$BASE" --setenv=CHECKER_CORPUS="${CHECKER_CORPUS:-}" \
    bash -c "cd $BASE; $PY -u $4 $5 >> $3 2>&1 || exit 3"
  echo "[$2] launched on GPU $1 -> $3"
}

# Realistic-Inductor on GPU 1; widened MAX Flash-Attention on GPU 3 (fabuildmax).
launch 1 realisticbuild "$LOGS/build_realistic.log" build_realistic_corpus.py "--seeds 12"
launch 3 fabuildmax     "$LOGS/build_fa.log"        build_fa_corpus.py         "--seeds 50"

echo "watch:  tail -f $LOGS/build_fa.log ; tail -f $LOGS/build_realistic.log"
echo "status: systemctl --user status fabuildmax realisticbuild"
