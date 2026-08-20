#!/bin/bash
# Kill-robust corpus builders. Re-run to RESUME after any kill/reboot (per-config checkpoints skipped).
# Two systemd --user services (survive Claude session teardown, MemorySwapMax=0 => oomd-safe,
# MemoryMax => won't crash the box), one per GPU, each building half the config shards.
# Each kernel uses --dtype all (loops its valid dtypes); GEMM family first (priority), then reductions.
#
# Needs systemd --user. Without it, run the same loop by hand -- see corpus_builder/README.md.
set -u
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # artifact_eval/corpus_builder
REPO=$(cd "$BASE/../.." && pwd)                      # repo root; bitequiv is not installed
PY=${PY:-$REPO/.venv/bin/python}
LOGS=${LOGS:-$BASE/logs}
mkdir -p "$LOGS"

QUEUE="gemm gemm_bias_relu_fp_fusion gemm_kgroup gemm_reduce_sum gemm_softmax \
sum dot sum_2d_axis0 sum_2d_col sum_2d_col_big col_sum_loop sum_3d_outer col_exp_sum \
sum_2d_keepdim sum_2d_deep sum_3d sum_3d_mid sum_4d sum_multiaxis col_dot col_max col_bf16 \
cond_reduce softmax layernorm rmsnorm welford"

launch() {  # $1=shard $2=gpu $3=unit $4=log
  systemctl --user reset-failed "$3" 2>/dev/null
  # Restart=on-failure + `|| exit 3`: a per-config timeout (builder os._exit(3)) makes bash exit 3, so
  # systemd cleans the cgroup (reaps the wedged ptxas) and relaunches; the loop resumes past the
  # timeout-marked config. RestartSec paces it; the default start-limit still stops a true crash-loop.
  systemd-run --user --unit="$3" -p MemorySwapMax=0 -p MemoryMax=40G \
    -p Restart=on-failure -p RestartSec=5 \
    --setenv=TRITON_ALWAYS_COMPILE=1 --setenv=CUDA_VISIBLE_DEVICES="$2" \
    --setenv=PYTHONPATH="$REPO" --setenv=CHECKER_CORPUS="${CHECKER_CORPUS:-}" \
    bash -c "cd $BASE; for k in $QUEUE; do $PY -u build_local_eval.py --kernel \$k --dtype all --shard $1 --nshards 2 --seeds 20 >> $4 2>&1 || exit 3; done; echo ALL_DONE_$1 >> $4"
}

launch 0 0 gemmbuild0 "$LOGS/build0.log"
launch 1 1 gemmbuild1 "$LOGS/build1.log"
echo "launched gemmbuild0 (GPU0 shard0) + gemmbuild1 (GPU1 shard1); queue: $(echo $QUEUE | wc -w) kernels"
echo "watch:  tail -f $LOGS/build0.log ; systemctl --user status gemmbuild0"
