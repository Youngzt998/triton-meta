# Tensor languages — IR/compiler survey (for the tile-smt refactor)

**Purpose.** We want to decouple the SMT tensor semantics from Triton into
reusable `tile-smt` / `tile-gpu-smt` libraries (see the refactor goal). This
survey compares how tile/tensor kernel languages structure their IR, to decide
*where the library boundary sits* — which semantics are "tensor-generic"
(→ `tile-smt`), "GPU/SIMT-specific" (→ `tile-gpu-smt`), or hardware-specific
(TPU/Trainium — future).

> **Provenance.** Surveyed on 2026-07-23 by reading the **actual cloned source**
> under `/home/youngzt/tv/` (offline; no web). Cells are evidenced from code
> unless marked **⚠️**. `cuTile` was **not cloned** and stays ⚠️ (unverified).
> Modular's compiler is a **closed prebuilt binary** (dialect `.td` not in repo).

---

## Table 1 — IR / compiler basics (evidenced)

| Language | Open? (lang / compiler) | Frontend | Core IR & framework | MLIR? | Target HW | Emits Triton IR? |
|---|---|---|---|---|---|---|
| **Triton** (base) | yes / yes | Python `@jit` | **TTIR → TTGIR** (own MLIR dialects) | **yes** | NVIDIA, AMD | — (is Triton) |
| **TileLang** | yes / **yes** (full C++ compiler local) | Python DSL | **TVM TensorIR**, a fork `tvm::tirx` + `tl.tileop.*` op nodes | **no** (TVM) | NVIDIA, AMD, CPU | no |
| **Pallas** (JAX) | yes / yes | Python (JAX) | frontend = `Ref`+`BlockSpec`; 3 backends ↓ | mixed | TPU + NVIDIA GPU | **GPU: yes** |
| ↳ Pallas `triton/` | " | " | **literally emits Triton TTIR** | yes | NVIDIA GPU | **yes** |
| ↳ Pallas `mosaic/` (TPU) | " | " | **MLIR `tpu` dialect** (`jaxlib/mosaic/dialect/tpu/*.td`) | yes | TPU | no |
| ↳ Pallas `mosaic_gpu/` | " | " | **MLIR `mosaic_gpu` dialect** (TMA/mbarrier/wgmma) | yes | NVIDIA GPU | no |
| **Hidet** | yes / yes | Python DSL | own 3-level Python IR (graph / task-compute / TIR-like) | **no** | NVIDIA GPU | no |
| **Helion** | yes / yes | Python DSL | **no own IR** — torch.fx → **emits Triton** (also pallas/metal/cute backends) | via Triton | NVIDIA (via Triton) | **yes** |
| **IREE / Linalg** | yes / yes | StableHLO/TOSA (not a kernel DSL) | **MLIR** `linalg`/`tensor`/`memref`/`vector` | **yes** | many | no |
| **Mojo / MAX** | stdlib yes / **compiler closed** | Mojo lang | MLIR dialects `mo`/`kgen`/`pop` but **`.td` closed** | yes (closed) | NVIDIA, AMD, CPU | no |
| **cuTile** (NVIDIA) | ⚠️ / ⚠️ (not cloned) | Python (CUDA) | ⚠️ unknown | ⚠️ | NVIDIA GPU | ⚠️ |

---

## Table 2 — access/exec model, vs Triton, and where it belongs

| Language | Access / memory model | Exec model class | Adds vs Triton | Lacks / differs vs Triton | Split fit |
|---|---|---|---|---|---|
| **Triton** | pointer tiles `!tt.ptr` + per-element **mask** | SIMT-tile (warp/lane) | — | — | TTIR≈`tile-smt`, TTGIR≈`tile-gpu-smt` |
| **TileLang** | `Buffer`+`BufferRegion` (+ `Range`); scopes via string (`shared.dyn`,`local.fragment`) | imperative buffer + explicit loop nest | explicit loops, scopes, TMA/wgmma/mbarrier/cluster in one IR; `Fragment` layouts | **no pointers, no data-mask** (mask→loop predicate); no make_range/splat/broadcast ops; reduce is dim-enum not combine-region; single-level IR | generic: buffer+range+reduce+gemm; gpu: scopes/fragments/mbarrier/TMA |
| **Pallas-Triton** | `Ref`+`BlockSpec index_map`; `pl.load/store`(mask optional) → pointer+mask | SIMT-tile GPU | `BlockSpec` windowing frontend | (frontend only; IR = Triton) | **≈ free (= Triton)** |
| **Pallas-Mosaic (TPU)** | **memref** + affine index; `sublane_mask`; **DMA + semaphores** | **systolic MXU + VMEM/DMA** | DMA/semaphore async, tiled sublane×lane layouts, `tpu.matmul` FIFO staging, multi-core | **no pointers, no per-element mask, no warp/lane, no convert_layout** | generic: elementwise/reduce/`dot`(=dot_general)/blocked-access; **hw-specific: DMA/sem/VMEM/PSUM/layout** |
| **Hidet** | `TensorType`/`PointerType`+layout; scopes Global/Shared/Register; scalar `TensorElement`+`IfStmt` | imperative loop + intrinsics | declarative task/compute level above body; graph IR | no data-mask op (predication manual); GPU = Python-emitted intrinsics not typed ops | generic: compute/reduce lambdas map to core; gpu: cuda intrinsics/scopes/cute |
| **Helion** | `hl.tile`/`hl.load/store/dot/reduce` → generated `tl.*`+mask | SIMT-tile GPU (via Triton) | auto masking/indexing/autotune at language level | none underneath (IR = Triton) | **≈ free (= Triton)** for its Triton backend |
| **IREE / Linalg** | `tensor`(value)/`memref`(+memory-space)/`vector`; `vector.transfer_read/write`+i1 mask+padding; `linalg.generic`(indexing_maps+iterator_types+body) | affine iteration (neutral) | language-neutral op algebra (best reference) | not a kernel DSL; no pointers/program_id/GPU layouts | **the neutral-core reference for `tile-smt`** |
| **Mojo / MAX** | CuTe `LayoutTensor`(+address_space,masked); explicit SIMT `thread/warp/cluster` | explicit SIMT + CuTe layout | layout algebra + full language at user level | **no open low-level IR** to validate against | reference only (compiler closed) |
| **cuTile** ⚠️ | ⚠️ | ⚠️ (tile, Triton-like) | ⚠️ | ⚠️ | ⚠️ verify |

---

## Per-language evidence (key paths)

- **TileLang** — no MLIR (`grep mlir:: src/` empty). Base is TVM fork `tvm::tirx`
  (`/home/youngzt/tv/tvm/include/tvm/tirx/{stmt,buffer,expr}.h`). Tile ops are
  `TileOperator` C++ nodes (`tilelang/src/op/{gemm,copy,reduce,region,parallel}.h`)
  under `tl.tileop.*`, each with `Lower()`→TIR + `InferLayout()`. Masks synthesized
  as loop predicates in `CopyNode::MakePredicate`. Scopes are string storage
  scopes (`include/tvm/ir/type.h`).
- **Pallas** — frontend `jax/_src/pallas/{core.py,primitives.py}` (`BlockSpec`,
  `load_p`/`swap_p`, `program_id_p`). GPU: `pallas/triton/lowering.py` builds
  `tt_dialect.{make_range,splat,broadcast,addptr,load,store,dot,reduce_return}`
  — real TTIR. TPU: `pallas/mosaic/lowering.py` + `jaxlib/mosaic/dialect/tpu/*.td`
  (`enqueue_dma`/`wait_dma`, `!tpu.semaphore`, `tpu.matmul`+FIFO, VMEM/SMEM/HBM,
  `VectorLayout`); in-VMEM load → `vector.load`/`memref.load`, reduce →
  `vector.multi_reduction`. GPU-Mosaic: `jaxlib/mosaic/dialect/gpu/mosaic_gpu.td`.
- **IREE/Linalg** — `linalg.generic` = `indexing_maps`(affine) + `iterator_types`
  (parallel/reduction) + scalar body region
  (`.../mlir/.../Dialect/Linalg/IR/LinalgStructuredOps.td`). Masked access =
  `vector.transfer_read/write`+mask+padding (`.../Dialect/Vector/IR/VectorOps.td`).
  memory-space on `memref` type. `scf` control flow (same as Triton).
- **Hidet** — own Python IR: graph (`graph/`), task/compute (`ir/compute/primitives.py`
  `GridCompute`/`ReduceCompute`), TIR-like (`ir/expr.py`,`ir/stmt.py`). Scopes via
  `DeclareScope`. GPU = intrinsic lib `ir/primitives/cuda/` + CuTe sub-IR `ir/cute/`.
- **Helion** — `_compiler/device_ir.py` (torch.fx), Triton backend emits `tl.load`
  (`extra_mask`)/`tl.dot`/`tl.reduce`/`tl.program_id` (`_compiler/triton/*`,
  `output_header.py`). Backends: `triton/pallas/metal/cute`.
- **NKI (Trainium)** — compiler closed; only `nki.language`(`nl.*`)+`nki.isa`(`nisa.*`)
  in `/home/youngzt/tv/nki-samples/src`. Memories chosen by `buffer=`: HBM /
  **SBUF** (partition×free, `pmax=128`) / **PSUM** (fp32 matmul accumulator).
  Access = tile + `nl.ds` slice + explicit `nisa.dma_copy`; `mask=` is an affine
  coordinate predicate (not a pointer mask). Engines: `nisa.nc_matmul` (PE/systolic),
  `nisa.tensor_reduce`/`tensor_tensor`/`activation` (vector/scalar), DMA. Has SPMD
  `nl.program_id`+grid; **no pointers, no per-element pointer mask, no warp/lane**.
- **Mojo/MAX** — compiler is a closed wheel (`bazel/modular_wheel_repository.bzl`);
  no `.td` in repo. Open: `max/kernels/src/layout/` (`LayoutTensor`) +
  `mojo/stdlib/std/gpu/` (SIMT `thread_idx`/`warp`/`cluster`). Graph dialect `mo`
  (`!mo.tensor`) exists but defined in closed `.td`.

---

## Cross-cutting observations (for the library boundary)

1. **MLIR is common but not universal.** MLIR: Triton, Pallas(Mosaic TPU+GPU),
   IREE, Mojo. Not MLIR: TileLang (TVM `tirx`), Hidet (own Python IR). → the
   `tile-smt` interface must be **IR-agnostic** (our own neutral value/op
   model), not tied to `mlir::Operation`. Each language ships a thin *adapter*
   that walks its IR and calls our builder API (Goal 2, first for Triton).

2. **Everyone converges on the same generic core.** Independent of hardware,
   every language expresses: elementwise map (scalar arith/math body), reduction
   over an axis (with a combiner), **matmul = contraction over a shared K dim**,
   broadcast/reshape/transpose, **blocked/windowed access to a logical N-D
   tensor**, and correctness = **final global-memory equality**. The single best
   *neutral encoding* is IREE's `linalg.generic` triple: **affine indexing-maps
   + parallel/reduction iterator-types + a scalar body region** — which is also
   exactly what an SMT encoder wants (affine map → `select` on an array; parallel
   axis → ∀ / reduction axis → fold; body → op-by-op translation).

3. **The divergence is entirely in the access / layout / sync layer:**
   - *access mechanism*: pointer+per-element-mask (Triton/GPU) · memref+affine
     index (Mosaic TPU) · buffer-region (TileLang) · Ref+BlockSpec (Pallas) ·
     named-tile+DMA (NKI). All are lowerings of "affine window of a logical
     tensor + optional predicate".
   - *register layouts* & `convert_layout` — GPU-only.
   - *async/sync*: mbarrier/TMA/warp-spec (GPU) vs semaphores/DMA (TPU/Trainium).
   - *memory placement*: shared/register (GPU) vs VMEM/SBUF/PSUM (TPU/Trainium)
     — non-semantic hints the equivalence check should ignore.
   - *execution axis*: warp/lane/thread (GPU) — absent on TPU/Trainium.

4. **Triton's own TTIR≈generic / TTGIR≈GPU split is the natural cut**, and it is
   corroborated by every other language: the generic ops recur everywhere; the
   GPU-only stuff (layouts, warp, shared mem, TMA/mbarrier, warp-spec) is what
   TTGIR adds.

5. **Cheapest extra clients:** Pallas-GPU and Helion (both literally emit Triton
   IR) → nearly free once the split exists. **Most different:** NKI (Trainium)
   and Pallas-Mosaic (TPU) — non-SIMT hardware.

---

## Answer — can ONE `tile-smt` cover both TPU/systolic IR and GPU IR?

**Yes, if `tile-smt` is kept hardware-neutral.** The evidence is strong: on
both TPU (Mosaic) and Trainium (NKI) the *tensor math* is identical to GPU —
elementwise maps, axis reductions, matmul as a K-contraction, blocked access to a
logical tensor, and the correctness spec is the same (the values written back to
**global memory / HBM**). Pallas even proves it: the *same* JAX kernel lowers to
Triton on GPU and to the `tpu` dialect on TPU from one frontend.

What breaks a naive design is baking **GPU-only assumptions** into the core.
To fit TPU/systolic too, `tile-smt` MUST NOT hardcode:
- a **flat linear-address pointer + arbitrary per-element mask** as the load/store
  primitive (TPU/Trainium have no pointers; masking is sublane/affine-coordinate
  granular);
- a **warp / lane / thread** execution axis;
- a **shared-vs-register** dichotomy or `convert_layout` register-layout ops.

Instead model access abstractly: **a tensor is a logical N-D value in a memory
tier (global vs on-chip); it is accessed by an affine tile/slice region with an
optional affine in-bounds predicate; moved between tiers by an opaque
semantics-preserving `copy`; and combined by tier-neutral `map` / `reduce(axis)`
/ `contract(K)`**. Then pointer+mask (Triton), memref+index (Mosaic), and
tile+DMA (NKI) are all just *lowerings* of the same region-access-with-predicate,
and layout/placement/engine choices become **uninterpreted attributes the
equivalence check ignores**. Correctness is pinned to global-memory equality.

Practical shape → a **three-tier** design (do the first two now for Triton):
- **`tile-smt`** (hardware-neutral): logical tensor values, `map`/`reduce`/
  `contract`, affine windowed access + predicate, memory as an address space,
  final-memory equivalence. No pointers/warps/layouts baked in.
- **`tile-gpu-smt`** (GPU/SIMT): pointer+mask lowering, register layouts /
  `convert_layout`, shared memory, warp/lane, async (TMA/mbarrier), warp-spec.
- **(future) `tile-accel-smt`** (per non-GPU hardware): TPU/Mosaic & Trainium/
  NKI DMA+semaphores, scratchpad placement (VMEM/SBUF/PSUM), systolic MXU
  staging, partition-dim layout.

Caveat: this covers **functional equivalence on global memory** and treats
layout/placement as ignorable. Cross-hardware numerics still differ (e.g. PSUM
is fp32-accumulate; GPU MMA rounding differs) — those live in the FP-mode
question, not the access model. For the current Abstract-FP mode (uninterpreted
FP ops), the shared core already covers TPU and GPU alike.

---

## TODO — still to verify (blocked/absent here)

- **cuTile** (NVIDIA): not cloned. Is the compiler open? MLIR? does it reuse
  Triton/CUTLASS? — whole row ⚠️.
- **Mojo/MAX**: dialect `.td` are closed; confirm whether any low-level GPU IR is
  ever exposed for validation (currently: no).
