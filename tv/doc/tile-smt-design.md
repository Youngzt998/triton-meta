# tile-smt — design draft

Draft for decoupling the SMT tensor semantics out of Triton into a reusable,
IR-agnostic library. Context/goal: `tv/CLAUDE.md` §2 and
`tv/doc/tensor-languages-survey.md`. This is a **draft** — revise as we discuss.

## Locked decisions (2026-07-23)
1. **Builder API** (not a neutral IR). Each language ships an *adapter* that
   walks its own IR and calls `tile_smt::Context` builder methods; the library
   builds Z3 directly. No intermediate tile-smt IR data structure.
2. **Incremental access model.** The core exposes an abstract *masked windowed
   access* interface whose default implementation is the current **linear
   byte-heap + pointer** model (Triton uses it directly). Designed so the access
   backend can later be swapped for memref/affine-window (TPU) — but pointers
   stay in the core for now.
3. **`program_id` / SPMD grid lives in the core** (it is a generic "parallel
   instance id"; TPU/NKI have it too), encoded as a fresh symbolic `BV32` shared
   by both programs.

## Principles
- The library **includes no MLIR/Triton headers** — only Z3 + its own light
  types. Language coupling lives entirely in the adapter.
- Split line = "read language IR" (adapter) vs "semantics → Z3 encoding" (lib).
  Today's `handleArith*/handleTt*` are exactly `adapter + lib` fused; we separate
  them.
- Migrate incrementally; every step keeps `triton-tv` working and the eval suite
  (`run_eval.py all`) + unit tests green.

## Library boundary
| Layer | Contents |
|---|---|
| **`tile-smt`** (hw-neutral core) | value model (`Scalar`/`Tensor`/`Ptr`), own `DType`+`Shape`, `AbstractFp` (+future Real/FPA), elementwise arith/math, structural ops (iota/splat/broadcast/reshape/expand_dims), reduce/scan (combine), dot/contract, `program_id`, abstract memory + masked windowed load/store (default = linear byte-heap+pointer), `checkEquivalence` (witness), control-flow state merge (if→ite, for→unroll) |
| **`tile-gpu-smt`** (GPU layer; mostly future) | layouts / `convert_layout`, shared memory, warp/lane, async (TMA/mbarrier), warp specialization |
| **triton adapter** (`tv/`, imports both) | walk `tt.func`; `Env` (`mlir::Value`→`Value`); `mlir::Value`→`MemId` map; per-op: read operands/attrs → call builder → bind result |

## Type & value model (no MLIR types)
```cpp
namespace tile_smt {
enum class DType { I1, I8, I16, I32, I64, F16, BF16, F32, F64, Ptr };

// Opaque handle for "which memory/address space" a pointer belongs to.
// The adapter maps its own provenance (e.g. mlir::Value of a kernel arg) → MemId.
enum class MemId : uint32_t {};

struct Scalar { z3::expr e; DType ty; };
struct Tensor { z3::expr e; std::vector<int64_t> shape; DType elem;
                std::optional<MemId> ptrBase; };   // set iff elem==Ptr
struct Ptr    { z3::expr e; DType pointee; MemId base; };   // scalar pointer, BV64
using Value = std::variant<Scalar, Tensor, Ptr>;
}
```
Rationale for `MemId`: the core must not hold `mlir::Value`. Provenance (needed to
pick the right per-arg `Memory`) becomes an opaque id owned by the adapter.

## Builder API (sketch)
```cpp
class Context {                    // owns z3::context, AbstractFpRegistry, FPMode
public:
  // symbolic inputs
  Scalar programId(int axis);                    // fresh shared BV32
  Value  freshInput(DType, Shape, std::string name);   // scalar/tensor arg
  Ptr    freshPtr(DType pointee, MemId, std::string name);
  Scalar constInt(int64_t, DType);  Scalar constFloat(APFloat-bits, DType);
  Tensor splatConst(...);  Tensor iota(int64_t start, Shape, DType);

  // elementwise (scalar or tensor; shape-checked)
  Value add(Value,Value); Value sub(Value,Value); Value mul(Value,Value);
  Value div(Value,Value); Value max(Value,Value); Value exp(Value); Value neg(Value);
  Value cmpInt(IPred, Value, Value); Value cmpFloat(FPred, Value, Value);
  Value andOp(Value,Value); Value extsi(Value,DType); /* ... */

  // structural
  Tensor splat(Scalar, Shape);  Tensor broadcast(Tensor, Shape);
  Tensor reshape(Tensor, Shape);  Tensor expandDims(Tensor, int axis);
  Ptr    addPtr(Ptr, Scalar off); Tensor addPtr(Tensor ptrs, Tensor off);

  // reduce / dot — combine passed as a callback so the core stays op-agnostic
  Scalar reduce(Tensor, int axis, std::function<Value(Value,Value)> combine);
  Tensor dot(Tensor a, Tensor b /*, dims */);
};
```
`reduce` takes a callback: the adapter turns a Triton `tt.reduce` combine region
into `[&](a,b){ return ctx.max(a,b); }`. The core just folds it; no notion of a
"region".

## Memory & access (core; linear-pointer default)
```cpp
class Memory {                     // Array(BV64, BV8), one per MemId
  z3::expr array;
};
struct MemState { std::map<MemId, Memory> mems; };   // per-arg memories

Tensor  load (const MemState&, MemId, Tensor addrs, Tensor mask, Tensor other);
MemState store(const MemState&, MemId, Tensor addrs, Tensor vals, Tensor mask); // lambda update

// equivalence: caller passes the two programs' output memories paired by position
// (the adapter pairs src arg i ↔ tgt arg i); core adds a witness-address query.
CheckResult checkEquivalence(const MemState& a, const MemState& b,
                             span<std::pair<MemId,MemId>> pairing, z3::solver&);
```
Access interface is abstract enough that a future memref/affine-window backend can
replace the pointer arithmetic without touching op semantics. `store` keeps the
current single-`z3::lambda` update; equivalence keeps the symbolic-witness check.

## AbstractFp (core)
Moves as-is into `tile-smt`, with `mlir::FloatType` replaced by `DType`.
Uninterpreted per-op functions + axioms; FP semantics are hardware-neutral.

## Control-flow merge (core; when scf.* lands)
The generic parts of `scf.if`/`scf.for` are hardware-neutral and belong in core:
- `if`: `merge(cond, thenState, elseState)` = `ite` over each result `Value` and
  each `Memory.array`.
- `for`: static unroll = thread `State` through N body copies.
The adapter supplies the bodies (by walking regions); the core supplies the merge
/ unroll primitives.

## Adapter (triton) shape
- `Env`: `map<mlir::Value, Value, ValuePtrLess>` — stays in the adapter.
- `mlir::Value(ptr arg) → MemId` map — adapter-owned.
- `walk(tt.func)`: for each op, read operands via `Env`, read attrs from the
  `mlir::Operation`, call the matching `Context`/memory builder, bind results.
- `State` becomes an adapter driver holding `Env` + `MemState` + a `Context&`.

## Migration steps (each keeps eval + unit tests green)
1. Create `tv/tile-smt/` (namespace `tile_smt`) + `DType`. Move `AbstractFp`
   in, swap `mlir::FloatType`→`DType`. Add a tiny adapter shim so existing code
   compiles.
2. Move `Memory` + value wrappers into the lib; `mlir::Type`→`DType`; introduce
   `MemId`; adapter builds `mlir::Value→MemId`. Keep `store` lambda + witness.
3. Extract elementwise/structural/reduce/dot semantics from `semantics/mlir/*`
   into `Context`; handlers become thin adapters (read operands → call builder).
4. `Env`/walking stay in adapter; `State` holds `Context`+`MemState`.
5. Stand up `tv/tile-gpu-smt/` skeleton (near-empty today).

## Deferred / open
- Long-term: make the access interface pluggable (memref/affine window) so
  TPU/Mosaic and Trainium/NKI can reuse `tile-smt` (survey §"Answer").
- `tile-gpu-smt` real contents arrive with TTGIR support (layouts, shared mem,
  warp, async, warp-spec).
- Real/IntegerRange/FPA FP modes.
- Exact `Combine`/`dot` dim representation; multi-dim reduce.
- Directory names `tile-smt` / `tile-gpu-smt` are provisional.
