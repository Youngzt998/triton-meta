# Evidence that the fuzzer is not blind, and that the features really fire

`smoke_planted.txt` is the output of `smoke_planted.py`. It shows three things.

**A. A fault planted by hand in the candidate TTGIR is detected.**
The accumulator initialiser `dense<0.000000e+00>` was changed to
`dense<1.000000e+00>` in the candidate text only; all 65536 output elements
then differ, `max_abs_diff = 1.0`. The negative control (the same program
compiled and launched twice) is reported equal.

**B. A pass that changes FP math by design is detected.**
Dropping `accelerate-matmul` on an fp32 `tl.dot` (MMA tf32 vs the generic FMA
path) makes 65534 of 65536 elements differ, `max_abs_diff = 0.057`.

**C. The passes under test really run.**

Warp specialization, `evidence_ws_{ref,cand}.ttgir` — reference is the sm_90
pipeline with `hopper-warpspec` removed, candidate is the full pipeline, same
kernel (`mm_tma`, fp16, 512x512x512, block 128x128x64, num_stages 3,
num_warps 4, `tl.range(..., warp_specialize=True)`):

| marker | ref | cand |
|---|---|---|
| `ttg.warp_specialize` | 0 | 1 |
| `partition0(` | 0 | 1 |
| `ttng.init_barrier` | 3 | 33 |
| `ttng.wait_barrier` | 1 | 8 |
| `ttg.memdesc_index` | 13 | 55 |

The candidate really has the warp-specialize region with two partitions:

```mlir
ttg.warp_specialize(%c0, %c_ptr, %M, %N, %K, ...) attributes {requestedRegisters = array<i32: 232, 232>}
default { ... ttg.warp_yield }
partition0(..., %arg15: !ttg.memdesc<3x64x64xf16, #shared1, #smem, mutable>, ...) num_warps(4) { ... ttg.warp_return }
partition1(...) num_warps(4) { ... ttg.warp_return }
```

Software pipelining, `evidence_pipe_{ref,cand}.ttgir` — same idea with
`pipeline` removed (`mm_ptr`, fp16, block 64x128x64, num_stages 4):

| marker | ref | cand |
|---|---|---|
| `ttg.async_copy_global_to_local` | 0 | 8 |
| `ttg.async_commit_group` | 0 | 8 |
| `ttg.async_wait` | 0 | 2 |
| `ttg.memdesc_index` | 0 | 10 |

TMA lowering: `ttng.tensormap_create` 0 -> 2 when `tma-lowering` is put back.

One thing worth writing down, because it silently makes pipelining tests
vacuous: the kernel must be given the same **argument specialization** the JIT
computes (`tt.divisibility = 16` on aligned pointers and on integers that are a
multiple of 16). Without it every load gets `sizePerThread = [1, 1]`,
`canBeConvertedToAsyncLoad` says no, and the pipeliner does nothing at all --
the IR before and after the pass is identical and the test proves nothing.
