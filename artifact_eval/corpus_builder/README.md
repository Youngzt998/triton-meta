# Building the checker corpus

The `checker.corpus` step grades a static checker over a corpus of already-compiled kernels. The
corpus is an **input**, not part of this repository: it is 11 GB and it is regenerated, not
archived. These are the scripts that build it. They are copied here from the directory the
original corpus was built in, with two edits, both marked in the source: the repository root is
now derived from the script's own location instead of being hard-coded, and the output directory
reads `$CHECKER_CORPUS` — the same variable the step reads, so building and grading cannot end up
pointed at two different places.

## What a corpus is

```
corpus/<kernel>/<dtype>/<cid>.ptx      compiled PTX for one autotuner configuration
corpus/<kernel>/<dtype>/<cid>.ttgir    the TTGIR for the same configuration
corpus/<kernel>/<dtype>/<cid>.json     {kernel, dtype, config, size, seeds, ok, empirical_key, ptx}
```

`cid` is a hash of the configuration. `empirical_key` is the ground truth: the kernel was
actually launched on `seeds` random inputs and the key is the hash of the per-seed output hashes.
Two configurations with the same `empirical_key` really did return the same bytes. `ok` is false
for a configuration that would not compile or launch; those rows are counted and excluded.

`empirical_key` depends on the **number of seeds**, so one group has to be built with one
`--seeds`. Keys are only ever compared inside a `(kernel, dtype)` group, so different groups
using different seed counts is fine — the builders use 20 for the main queue, 12 for the
realistic-Inductor slice and 50 for flash attention. A group with two different seed counts in it
is a bug; the step reports the seed counts it saw on every row so you can see it.

## The three builders

| script | what it builds | how the configurations are chosen |
|---|---|---|
| `build_local_eval.py` | the GEMM family and the reductions | GEMM: a hand-written axis cross-product, with the bit-relevant axes (`input_precision`, `enable_fp_fusion`) held fixed. Reductions: the evaluation framework's own `max_config_space()`. |
| `build_realistic_corpus.py` | the realistic torch-Inductor reductions (`A_*`, `C_*` … `J_*`, `LC_*`) | reuses the case list already in `bitequiv/evaluation/realistic_inductor_kernels.py`. |
| `build_fa_corpus.py` | flash attention | the full cross-product of causal, `HEAD_DIM`, `num_warps`, `BLOCK_M`, `BLOCK_N`, `num_stages`, pruned to the configurations the real autotuner would accept. |

All three write one atomic JSON per configuration and skip a configuration that is already
there, so they **resume**: re-run after a kill or a reboot and they carry on. All three cap a
single configuration at 180 seconds and hard-exit if one wedges the compiler, so a bad
configuration cannot stall the build.

## Running them

```
export PYTHONPATH=$(git rev-parse --show-toplevel)
export CHECKER_CORPUS=$HOME/bitwise-equiv/local_evaluation/corpus   # or wherever you want it
export CUDA_VISIBLE_DEVICES=0                                       # a GPU nobody else is using
cd artifact_eval/corpus_builder

for k in gemm gemm_bias_relu_fp_fusion gemm_kgroup gemm_reduce_sum gemm_softmax \
         sum dot sum_2d_axis0 sum_2d_col sum_2d_col_big col_sum_loop sum_3d_outer col_exp_sum \
         sum_2d_keepdim sum_2d_deep sum_3d sum_3d_mid sum_4d sum_multiaxis col_dot col_max \
         col_bf16 cond_reduce softmax layernorm rmsnorm welford; do
    python -u build_local_eval.py --kernel $k --dtype all --seeds 20
done
python -u build_realistic_corpus.py --seeds 12
python -u build_fa_corpus.py --seeds 50
```

`--shard i --nshards n` splits the configurations by hash across parallel builders, one per GPU.
`launch_builders.sh` and `launch_missing_groups.sh` do exactly the loop above under `systemd
--user`, two shards on two GPUs, with `MemorySwapMax=0` and `Restart=on-failure` so a wedged
configuration is reaped and the build resumes past it. They need `systemd --user`; without it use
the plain loop.

## Cost

**Disk: 11 GB** for the full corpus — 5.2 GB of it flash attention, 4.3 GB GEMM, 1.5 GB
everything else. 54,120 PTX files, 62,458 configuration records in 93 `(kernel, dtype)` groups.

**Time: order of a day, not an hour**, and it needs a GPU throughout, because every configuration
is compiled *and* launched on every seed. From the build logs of the run that produced the corpus
on this machine (4x NVIDIA GB300; log file timestamps, so these are wall clock including
restarts, and two of the three were resumed runs rather than from scratch):

```
GEMM + reductions      about 7 hours across two GPUs (two shards)
realistic-Inductor     about 5 hours on one GPU
flash attention        about 15 hours on one GPU
```

Flash attention is the expensive half in both time and disk. If you only want to check the
reduction and GEMM rows, run `build_local_eval.py` alone: that is about 6 GB and the shorter
build, and the step grades whatever groups exist.
