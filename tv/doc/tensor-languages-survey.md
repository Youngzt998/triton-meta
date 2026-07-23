# Tensor languages — IR/compiler survey (for the tensor-smt refactor)

**Purpose.** We want to decouple the SMT tensor semantics from Triton into
reusable `tensor-smt` / `tensor-gpu-smt` libraries (see the refactor goal). This
survey compares how other tile/tensor kernel languages structure their IR, to
inform *where the library boundary should sit* — i.e. which semantics are
"tensor-generic" (→ `tensor-smt`), which are "GPU/SIMT-specific"
(→ `tensor-gpu-smt`), and which are hardware-specific and out of scope for now.

> **Confidence / provenance.** This box was drafted **offline** from prior
> knowledge (~2026-01); outbound web access is blocked on this machine, so it
> is **not** freshly verified. Cells marked **⚠️** are uncertain (new and/or
> closed-source projects) and MUST be checked online before we rely on them.
> ✅ = high confidence (open, well-known). Triton is the baseline row.

---

## Table 1 — IR / compiler basics

| Language | Open source? (lang / compiler) | Frontend | Core compiler IR (pre-LLVM) | MLIR-based? | Target HW |
|---|---|---|---|---|---|
| **Triton** (base) | ✅ yes / yes (Apache-2.0) | Python `@jit` | **TTIR → TTGIR** (own MLIR dialects) → LLVM | ✅ **yes** | NVIDIA, AMD (Intel WIP) |
| **TileLang** | ✅ yes / yes | Python DSL | **TVM TensorIR (TIR)** | ❌ no — TVM stack | NVIDIA, AMD, CPU (via TVM) |
| **Google Pallas** | ✅ yes / yes (in JAX) | Python (JAX) | **Mosaic** (TPU, MLIR dialect); **Triton/Mosaic-GPU** (GPU) | ✅ yes (Mosaic dialect; reuses Triton on GPU) | TPU, NVIDIA GPU |
| **cuTile** (NVIDIA) | ⚠️ lang public / compiler ⚠️ likely closed | Python (CUDA) | ⚠️ NVIDIA tile IR (unpublished); likely → NVVM/LLVM | ⚠️ unknown (NVIDIA uses MLIR elsewhere) | NVIDIA GPU |
| **AWS NKI** | ⚠️ lang/API public / compiler ❌ closed (Neuron) | Python | ⚠️ Neuron compiler IR (closed); StableHLO/MLIR at the front ⚠️ | ⚠️ partly (front only) | AWS Trainium/Inferentia |
| **Mojo / MAX** (Modular) | ⚠️ stdlib open / compiler ❌ closed | Mojo language | own **MLIR**-based stack (incl. GPU dialects) | ✅ yes | NVIDIA, AMD, CPU |
| **Hidet** (CentML) | ✅ yes / yes | Python DSL | own graph+task IR ("Hidet IR") | ❌ no | NVIDIA GPU |
| **IREE** | ✅ yes / yes | StableHLO/TOSA (not a kernel DSL) | **Linalg → ... (MLIR)** | ✅ yes | many (GPU/CPU/accel) |
| **Helion** (PyTorch) | ⚠️ yes (new) | Python DSL | lowers to **Triton** ⚠️ | via Triton | NVIDIA (via Triton) |

Notes on scope: IREE and StableHLO are *model/graph* compilers, not tile kernel
DSLs, but included because they define "tensor-generic" MLIR ops (Linalg) that
are a useful reference for the `tensor-smt` boundary. ThunderKittens / CUTLASS
CuTe are C++ embedded libraries (no standalone IR) — omitted. Gluon is Triton's
experimental frontend over the *same* IR — same as Triton for our purposes.

---

## Table 2 — vs Triton (what the IR adds / lacks) and relevance to the split

| Language | Adds vs Triton | Lacks / differs vs Triton | Reuses Triton IR? | Fit for `tensor-smt` (generic) vs `tensor-gpu-smt` (GPU) |
|---|---|---|---|---|
| **Triton** | — | — | — | Defines both layers today; the thing we're splitting |
| **TileLang** | explicit loop nests (TIR), explicit memory scopes (`T.alloc_shared/fragment`), explicit copy/pipeline primitives, `T.gemm`, layout inference as schedule | no `!tt.ptr` pointer+mask load/store (buffer/region based); not MLIR; more imperative-loop than tile-expression | ❌ no (TVM) | Strong overlap on tile ops/reduce/gemm → `tensor-smt`; memory-scope + pipeline is GPU-ish → `tensor-gpu-smt` |
| **Pallas (GPU)** | `Ref`-based load/store, `BlockSpec` (grid+block indexing) | thin frontend; GPU path *is* Triton | ✅ yes (GPU → Triton) | Nearly free: same IR as Triton |
| **Pallas (TPU)** | Mosaic: TPU vector/DMA semantics, different memory (VMEM/SMEM) | no SIMT/warp/PTX; TPU exec model | ❌ (Mosaic) | Tile/reduce/dot → `tensor-smt`; TPU memory/exec is HW-specific (new backend later) |
| **cuTile** ⚠️ | tile Python DSL, compiler-managed layout/scheduling (Triton-like philosophy) | ⚠️ IR unpublished; NVIDIA-only | ⚠️ unknown | Likely maps well to `tensor-smt` + NVIDIA `tensor-gpu-smt` — verify |
| **AWS NKI** | explicit on-chip memory (SBUF/PSUM), tensor-engine ops, explicit DMA, NeuronCore tiling | not SIMT/GPU; no `program_id`/warp; systolic/tensor-engine model | ❌ no | Elementwise/reduce/dot → `tensor-smt`; memory+engine model is a *third* HW target (not GPU) |
| **Mojo/MAX** ⚠️ | general language + MLIR GPU dialects, layout types | compiler closed; less tile-declarative | ❌ no (own MLIR) | MLIR-based → conceptually close; details closed |
| **Hidet** | task-based scheduling, explicit tensor programs | own IR (not MLIR), not pointer+mask | ❌ no | Op semantics → `tensor-smt`; scheduling is HW-ish |
| **IREE / Linalg** | named+generic tensor ops (`linalg.matmul`, `linalg.generic`), affine iteration | not a kernel DSL; higher level | ✅ MLIR (different dialects) | Best *reference* for a language-neutral `tensor-smt` op set |

---

## Cross-cutting observations (for the library boundary)

1. **MLIR is common but not universal.** Triton, Pallas(TPU/Mosaic), Mojo, IREE
   are MLIR; TileLang (TVM TIR) and Hidet are not. → the `tensor-smt` interface
   must be **IR-agnostic** (take our own neutral op/value model), not tied to
   `mlir::Operation`. Each language writes a thin *adapter* that walks its IR and
   calls our builder API (exactly Goal 2 for Triton).

2. **Two clear layers show up everywhere:**
   - *Tensor-generic*: elementwise arith, `exp/log/…`, broadcast/reshape,
     masked load/store over an address space, reductions, dot/matmul,
     equivalence of final memory. → `tensor-smt`.
   - *GPU/SIMT-specific*: `program_id`/grid, warp/lane, layouts
     (blocked/shared/mma), `convert_layout`, shared memory, async copy, warp
     specialization. → `tensor-gpu-smt`.
   Triton's own **TTIR ≈ tensor-generic**, **TTGIR ≈ GPU-specific** split is a
   strong hint the two-library cut is natural.

3. **Non-GPU targets exist (TPU, Trainium).** NKI and Pallas-TPU are *not* SIMT.
   If we ever want them, `tensor-smt` must stay free of GPU assumptions, and a
   sibling to `tensor-gpu-smt` (e.g. per-hardware memory/exec models) is where
   TPU/Trainium would go. Keep `tensor-smt` from leaking `program_id`/warp.

4. **Pointer+mask vs buffer/region vs Ref+BlockSpec.** Memory addressing differs:
   Triton = pointer tiles + mask; TileLang/Hidet = buffer regions; Pallas = `Ref`
   + `BlockSpec`; NKI = named on-chip buffers + DMA. → the `tensor-smt` memory
   interface should model the *abstract* "masked access to an address space"
   (which our current byte-addressable `Memory` already does) and let adapters
   translate their addressing into it.

5. **Cheapest second client = Pallas-GPU** (it *is* Triton IR) and **cuTile**
   (same tile philosophy, ⚠️ verify IR). **Most different = NKI** (non-GPU HW).
   A good validation of the split: after the refactor, adding Pallas-GPU should
   need almost no new `tensor-smt` code.

---

## TODO — verify online (blocked here)

- **cuTile**: is the compiler open? what IR does it use — MLIR? does it reuse any
  Triton/CUTLASS infra? (whole row is ⚠️)
- **AWS NKI**: confirm the Neuron compiler IR; is any of it MLIR/StableHLO-based?
- **Mojo/MAX**: which MLIR GPU dialects; how tile-level is the user-facing IR?
- **Helion**: confirm it lowers to Triton and has no separate IR.
- **TileLang**: confirm current backend is still TVM TensorIR (not a new MLIR path).
- **Pallas**: confirm current GPU lowering (Triton vs Mosaic-GPU) split.
