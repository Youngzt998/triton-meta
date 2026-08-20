# CODEGEN-0001: a pipelined shared buffer smaller than 128 bytes per stage makes the TMA copy address misaligned, and the kernel dies at launch

Verdict: **CONFIRMED — genuine compiler bug** (reviewed 2026-08-19)
Arch:    hopper-only (needs the TMA bulk copy, `sm_90a`; measured on H100)
Line:    tilelang r2 (TileLang 0.1.12, pypi wheel, nvcc 12.8.93, driver CUDA 13.0)
Class:   **invalid generated code with the stock pipeline**, not a bit
         difference. No pass is toggled; the default compilation is the broken
         one. `REVIEW-CRITERIA.md` §2 keeps this shape explicitly.
Culprit: `tl.cuda.ProducerConsumerWarpSpecialized` — it gives a pipelined
         shared buffer one slot per stage without rounding the slot up to the
         alignment the TMA copy needs.

Summary: an ordinary TileLang kernel that stages a small slice into shared
memory inside a `T.Pipelined` loop is compiled by the **stock** pipeline, with
default `pass_configs` and no toggles at all, into a kernel that dies at launch
with

```
CUDA error: misaligned address (cudaErrorMisalignedAddress)
```

Multi-buffering puts pipeline stage `k` at `base + k * slot_bytes`. TileLang
aligns `base`, but never `slot_bytes`. The TMA bulk copy needs its shared
destination to be 128-byte aligned, so **every slot size that is not a multiple
of 128 bytes traps** on the odd stages.

**The round-1 report with the same id is a different bug.** `tilelang/CODEGEN-0001`
(round 1) is a vectorized `floordiv` that emits CUDA nvcc will not compile. This
one is a runtime fault in code that compiles. They share only the `CODEGEN` id
space, which is per-round.

---

## 1. Root cause in the source

All line numbers are from the TileLang 0.1.12 source checkout at
`/home/youngzt/fuzz-tilelang-r2/tilelang-src` (same version as the installed
wheel).

**Step 1 — the pass that multi-buffers.** `T.Pipelined(num_stages=S)` does not
resize the buffer in the frontend, and neither `PipelinePlanning` nor
`InjectSoftwarePipeline` does it either. On an `sm_90a` target the work is done
by the warp-specialization pass:
`src/cuda/transform/producer_consumer_ws.cc:2756` calls
`ApplyMultiVersionBufferRewriter`, and that lands in
`src/cuda/transform/multi_version_buffer_rewriter.cc:494-509`:

```cpp
  static Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    if (buffer.scope() == "shared.barrier") { ... } else {
      new_buffer->shape.insert(new_buffer->shape.begin(),
                               PrimExpr(num_versions));      // <-- the slot dim
      ...
    }
    return Buffer(new_buffer);
  }
```

**This is the incorrect step.** A new leading dimension is prepended, so the
stride between two stages is exactly the old buffer's size. Nothing rounds that
size up to any alignment, and the function does not know — or ask — what
alignment the consumers of the buffer need. Traced on the failing kernel, the
shape goes `Xs (32,)` → `Xs (2, 32)` right after
`ProducerConsumerWarpSpecialized` and stays that way to the end.

**Step 2 — the alignment TileLang itself knows about, applied to the base only.**
When the copy is lowered to TMA, `src/cuda/op/copy.cc:352-366` reports the
requirement:

```cpp
static void RequireTMASmemAlignment(const LowerArgs &lower_args,
                                    const Buffer &shared_tensor,
                                    int cu_tensor_map_swizzle) {
  ...
  lower_args.require_smem_alignment(shared_tensor->data, mode.SmemAlignment());
}
```

and `src/layout/swizzle_mode.h:78-85` states the number and the reason in
TileLang's own words:

```cpp
  // Required shared-memory base alignment (bytes): ...
  // none->128 (bulk-copy base requirement), 32B->256, 64B->512, 128B->1024.
  int SmemAlignment() const { return 128 << CanonicalOrdinal(); }
```

The callback is keyed on the buffer's **data Var** — one number for the whole
allocation. `src/op/operator.h:122-128` says so explicitly: "record a minimum
shared-memory **base** alignment for a buffer's data Var".

**Step 3 — the allocator honours it, for the base.**
`src/transform/merge_shared_memory_allocations.cc:585-604` sums the whole
(already multi-versioned) shape into one flat size and aligns only the starting
cursor:

```cpp
      int alignment = align_bytes_;
      auto align_it = shmem_alignment_map_.find(var);
      if (align_it != shmem_alignment_map_.end())
        alignment = std::max(alignment, align_it->second);
      cursor = AlignPrimExpr(cursor, alignment);        // <-- base only
      buffer_byte_offsets_[var] = cursor;
```

There is even a compile-time post-condition for it, lines 488-501, which fails
loudly if a base is misplaced — and it passes here, because it checks
`offset % required == 0` and never `slot_bytes % required == 0`.

**The gap in one sentence:** the alignment contract is stated, enforced and
checked for the buffer *base*, while multi-versioning makes the real destination
`base + stage * slot_bytes`, and `slot_bytes` is never constrained.

The address that comes out, from `stock.cu`:

```c
extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
void* As = ((void*)((char*)buf_dyn_shmem + 0));      // slot 4096 B — fine
void* Xs = ((void*)((char*)buf_dyn_shmem + 8192));   // base 1024-aligned — fine
...
    tl::tma_load(X_desc, mbarrier[ko], (&(((half_t*)Xs)[(ko * 32)])), (ko * 32));
```

`Xs` slot stride is 32 halves = **64 bytes**, so stage 1 sits at
`buf_dyn_shmem + 8256` and `8256 % 128 == 64`. `tl::tma_load` is
`cp.async.bulk.tensor.1d.shared::cluster.global...`
(`src/tl_templates/cuda/copy_sm90.h:100`), which traps on that address.

## 2. Reproduction

Two ways. Both were run start to finish for this review.

**(a) Self-contained, no fuzzing harness** — `repro_standalone.py`, next to this
file. It sweeps the slot size, one child process per point (a CUDA fault poisons
the whole process, so points measured in one process are meaningless after the
first fault):

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
python /home/youngzt/tv/triton/tv/bug-report/tilelang/r2/CODEGEN-0001/repro_standalone.py
```

Verified output (2026-08-19, H100, GPU 0):

```
nelem=32 slot_bytes=64 slot%128=64
     tl::tma_load(X_desc, mbarrier[(ko & 1)], (&(Xs[((ko & 1) * 32)])), (ko * 32));
     torch.AcceleratorError: CUDA error: misaligned address
nelem=64 slot_bytes=128 slot%128=0
     RESULT: ran fine
nelem=96 slot_bytes=192 slot%128=64
     torch.AcceleratorError: CUDA error: misaligned address
nelem=128 slot_bytes=256 slot%128=0
     RESULT: ran fine
nelem=160 slot_bytes=320 slot%128=64
     torch.AcceleratorError: CUDA error: misaligned address
nelem=192 slot_bytes=384 slot%128=0
     RESULT: ran fine
```

The kernel is 8 lines:

```python
@T.prim_func
def main(X: T.Tensor((total,), "float16"), Y: T.Tensor((total,), "float16")):
    with T.Kernel(1, threads=32) as bx:
        Xs = T.alloc_shared((nelem,), "float16")
        for ko in T.Pipelined(4, num_stages=2):
            T.copy(X[ko * nelem], Xs)
            T.copy(Xs, Y[ko * nelem])
```

**(b) The original finding, through the fuzzing harness:**

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
cd /home/youngzt/fuzz-tilelang-r2
python -m tlfz.repro --spec '{"fam": "gemv", "p": {"M": 64, "K": 64, "bM": 64, "bK": 32, "th": 32, "st": 2, "dt": "float16"}, "pc": {}}' '{}' --mode normal --seed 5
```

Verified: the reference compile succeeds and the first run raises
`torch.AcceleratorError: CUDA error: misaligned address`. The repro script runs
the *reference* arm first, so the fault happens before any comparison. The
kernel is a plain GEMV — see `kernel_source.py`; nothing in it is unusual.

## 3. What the measurements prove, and where it reproduces

**The failing precondition is the shared destination address, not the slot
size.** The fuzzer's original table only varied the slot between 64 and >=128
bytes, which cannot tell those two apart, and it labelled the 128-byte rule as
*inferred*. Three independent checks now pin it:

1. **Slot-size sweep** (table above, 6 points): 64, 192 and 320 bytes fault;
   128, 256 and 384 bytes run. 192 and 320 are **larger** than 128 bytes, so
   "the slot is too small" is not the rule; `slot_bytes % 128 == 0` is.
2. **Shared side vs global side, separated.** In the sweep both move together.
   Two extra cases move them apart (a shared buffer of one size, a copy of
   another):

   | case | global step | shared slot | result |
   |---|---|---|---|
   | G | 256 B (128-aligned) | 192 B (64 mod 128) | **misaligned address** |
   | H | 192 B (64 mod 128) | 256 B (128-aligned) | runs |

   Only the shared side matters.
3. **compute-sanitizer names the instruction.** On the 64-byte case:
   `========= Misaligned shared or local address` …
   `Device Frame: main_kernel+0x870 in tvm_kernels.cu:43`, and line 43 of the
   generated CUDA is exactly the `tl::tma_load(..., &Xs[(ko & 1) * 32], ...)`
   call. It is a **shared** address, at the bulk copy.

Together with TileLang's own comment ("none->128 (bulk-copy base requirement)")
the 128-byte rule is no longer an inference.

**Where it reproduces.** Hopper (`sm_90a`) and anything else that takes the TMA
path. The judgement rests on measurement, not on reading: with
`{"tl.disable_warp_specialized": true}` the same source lowers the copy to
`tl::cp_async_gs` instead of `tl::tma_load`, and the kernel then **runs and
gives the right answer** (verified). `cp.async` has no 128-byte destination
rule, so the fault needs the TMA path, which is `sm_90+` only. Not tested on
Blackwell; the same code path exists there, so it is likely but unproven.

**Correction to the original ablation table.** The first version of this report
concluded "there is no good arm". That is right about the *default* pipeline but
wrong as a general statement: dropping `InjectSoftwarePipeline` or
`PipelinePlanning` indeed changes nothing (verified — the generated CUDA is
byte-identical, because neither of them is what multi-buffers here), but turning
warp specialization off does avoid it. So there is a **workaround**
(`tl.disable_warp_specialized=True`, at the cost of the TMA path), and the
multi-versioning belongs to `ProducerConsumerWarpSpecialized`.

**Blast radius.** Any TileLang kernel whose `T.Pipelined` loop stages a shared
buffer whose per-stage size is not a multiple of 128 bytes, on Hopper, with the
default pipeline. In this campaign it hit 2 of 8 random `gemv` draws. R5's
generator now avoids it with the rule `bK * itemsize >= 128`
(`tlfz/gen.py:600-612`); the sweep above shows that rule is **not sufficient** —
192- and 320-byte slots still fault — but it is enough to keep the known bug from
killing a worker on every cycle.

**Determinism.** Deterministic: 8 independent runs of the `bK=32, float16, st=2`
point in 8 fresh processes, 8 faults; every passing row above was also measured
in its own fresh process.

## 4. In plain words

TileLang was asked to overlap loads with compute, so it made two copies of the
staging buffer and lets the loop write into them in turn. It put the buffer
itself at a nicely rounded address, but it laid the two copies back to back. The
hardware's fast bulk-copy engine only accepts destinations at multiples of 128
bytes. When one copy is 64 bytes long, the second copy starts halfway into a
128-byte block, and the hardware refuses. The compiler already knows the
128-byte rule — it uses that number when it places the buffer — it just never
applies it to the spacing between the copies. The fix is to round the per-stage
size up to the same alignment the buffer base was given.

## 5. Can an SMT solver model this?

**Verdict: the semantics is easy to capture — but not as an equivalence check.
As a one-program safety check it is SMT-decidable in principle, and cheap.**

1. **Root cause in semantic terms.** After multi-versioning, a shared buffer
   access carries the index `stage * slot_elems + i`, and the TMA intrinsic
   takes `address_of(Xs[stage * slot_elems])` as its destination. That intrinsic
   has a *precondition* — the destination address must be `0 mod 128` — which no
   pass ever states. The bug is a violated precondition on an address, not a
   wrong value: every element that does get copied lands in the right place.
2. **Is it a pure input→output function difference?** No, and this is the
   important part. There is no wrong answer to compare against — the kernel
   never finishes. The two arms of an equivalence check would, if the hardware
   let them run, compute the *same* function. So **no equivalence check, however
   perfect its FP model or its memory model, will ever see this.** Nothing about
   the input values is involved: the fault depends only on compile-time
   constants (the buffer size, the stage count, the allocator's base offset).
3. **Fundamental verdict.** `SMT-decidable in principle` — with the caveat in
   (2). This is **not** `below the IR level`: the offending arithmetic is fully
   visible in TIR before codegen (the buffer's byte offset is an integer
   constant, the stage index is a loop variable with a known range, the
   intrinsic is `tl.tma_load`). It is not `needs whole-grid modeling` (one
   instance), not `needs a concurrency / memory-ordering model` (the TMA is
   asynchronous and uses an mbarrier, but the trap is not a race — a single
   elected lane issuing one copy is enough), and not `needs bit-exact FP` (no
   arithmetic on values at all).
4. **Minimum semantic vocabulary.** Shared memory as a second address space with
   a concrete byte offset per allocation; a buffer access lowered to a byte
   address, so `base + stage * slot_bytes` is an expression the checker can see;
   the pipeline stage as a symbolic integer with a known range `[0, S)`; and —
   the piece nothing models today — **instruction preconditions as assertions
   attached to an intrinsic**. With those, the verification condition is
   `∀ k ∈ [0, S): (base + k * slot_bytes) mod 128 == 0`, which is linear integer
   arithmetic with one modulus: a solver answers it instantly, and the
   counterexample it prints is `k = 1`.
5. **Would the abstraction hide it?** The FP axiom profile is irrelevant — no
   floating-point value takes part. What *would* hide it is any model that
   treats memory as a map from indices to values instead of from byte addresses
   to bytes: index-level modelling loses the byte offset and with it the whole
   question. `tv`'s core already stores memory as a byte-addressed array
   (`semantics/Memory.*`), which is the right shape; what is missing is the
   notion that an operation can *refuse* an address.
6. **Inherent cost.** Tiny. One linear-arithmetic query per (buffer, intrinsic)
   pair, with no tile unrolling and no FP. This is the cheap end of the
   spectrum, far cheaper than the value-equivalence queries the validator
   normally runs.

**Take-away for the validator project:** a class of real, launch-killing
compiler bugs is invisible to *equivalence* checking by construction, and needs a
second kind of query — per-instruction preconditions on one program. It is much
cheaper than equivalence checking, so it is a good thing to have alongside it.

## 6. Confidence

**Very high** that this is a genuine compiler bug.

- The failing rule is measured on six slot sizes with the mod-128 pattern clean
  in both directions, and the shared side is separated from the global side by
  two more cases.
- compute-sanitizer independently calls it a misaligned **shared** address and
  points at the `tl::tma_load` line.
- TileLang's own source states the 128-byte bulk-copy requirement and applies it
  to the base only; the path from `RewriteAllocBuffer` to the emitted address is
  short and readable.
- The user program is ordinary and correct; no toggle, no odd flag, no
  undefined behaviour on the kernel's side. The compiler picked TMA on its own.

What is *not* settled: I did not read the PTX ISA text in this environment, so
"128 bytes" rests on TileLang's own comment plus the measurement, not on a quote
from NVIDIA's manual. I did not test Blackwell. And I did not check whether some
other pass would round the slot up in a kernel that also has a swizzled operand
(a swizzled buffer asks for 256/512/1024-byte base alignment, and its slot is
usually a whole number of those anyway, which is probably why this went
unnoticed). Reading the PTX ISA section on `cp.async.bulk.tensor`, or trying an
`sm_100` target, would close both gaps.

## Artifacts

* `repro_standalone.py` — self-contained repro + the slot-size sweep (new, run
  for this review)
* `kernel_source.py` — the original TileLang GEMV that found it
* `kernel.tvmscript.py` — its PrimFunc
* `stock.cu` — the generated CUDA that faults
* `no_pipeline.cu`, `no_pipeplan.cu`, `no_async.cu` — the byte-identical
  ablations (dropping `InjectSoftwarePipeline`, dropping `PipelinePlanning`,
  `tl.enable_async_copy=False`)
