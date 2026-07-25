# M0 Migration Plan — Extract `tile-smt` from the Triton-coupled validator

**Status: DRAFT — design only; M0 not started.** Context: `tv/doc/roadmap.md`
(M0). Target architecture + locked decisions: `tv/doc/tile-smt-design.md`,
`tv/doc/tile-smt-goals.md`. All `file:line` refs are to the current tree at
`/home/youngzt/tv/triton`.

## 0. Goal & invariants

Move today's Triton-coupled SMT implementation in `tv/semantics/` onto the
`tile-smt` architecture, **without changing behavior**. Two sides:

- **SMT side** — a new MLIR-free core library `tile-smt` (own `DType`/`Shape`/
  `MemId`, `Memory` + `AbstractFp` + value wrappers + op *semantics* as a builder
  `Context` + `checkEquivalence`), building and unit-testing with **only Z3**.
- **Triton side** — the current op handlers + `Env`/`State` become a thin
  **adapter** that walks TTIR and calls the `tile-smt` builder.

Invariants that gate every sub-step:
1. `python tv/eval/run_eval.py all` stays green (pairs + inequal + compile-options).
2. `ninja … tv-validator-tests` + ctest `-R TestTritonTV` stay green.
3. After migration, `grep -rE 'mlir::|triton::' tv/semantics/` returns nothing, and
   the `tile-smt` CMake target links only Z3 (no MLIR/Triton libs).
4. The witness/lambda SMT encoding is preserved byte-for-byte (Memory.cpp:146-198
   store lambda; State.cpp:159-183 witness) — verdicts/timing don't regress.
5. The `tv/triton-tv` binary keeps its build path (`<build>/tv/triton-tv`) so
   `tv/eval/common.py:86-87` still finds it.

**Builder structure (design refinement).** The "adapter" is organized as a
**builder layer** under `tv/builder/`: `builder/mlir/` (shared by all
MLIR-based languages — `dtypeOf`, `Env`, `MemId` mapping, the block-walk driver,
and the standard `arith`/`math`/`scf` handlers) and `builder/triton/` (Triton's
`tt.*` handlers + the `.ttir` entry). So `tv/semantics/` does not merely "shrink
to an adapter" — its contents split into the **core** (`tv/semantics/`) and these
**builders**. See `tile-smt-design.md` §"Builder layer". Build targets gain
`tile-smt-builder-mlir` (core+MLIR) and `tile-smt-builder-triton`
(core+builder-mlir+Triton). In the tables below, read "adapter" as the builder
layer, and split adapter-side rows: `Env`/`dtypeOf`/walk-driver/`arith`,`math`,
`scf` → `builder/mlir`; `tt.*` handlers + entry → `builder/triton`.

---

## 1. Directory layout + build

### 1.1 Directory layout
```
tv/
  semantics/                # CORE tile-smt semantics — MLIR-free, links ONLY z3 (namespace tile_smt)
    Types.h                 # DType, Shape, MemId, FPMode + helpers (getElemSort/getByteWidth/fpExpSigBits)
    Value.h                 # Scalar/Tensor/Ptr wrappers + MemId (was Z3Scalar/Z3Tile/Z3Ptr)
    FpModel.h               # FpModel interface (AbstractFp = mode a)
    AbstractFp.{h,cpp}      # DType-based (was semantics/AbstractFp.*)
    Memory.{h,cpp}          # DType/MemId-based; store lambda + witness kept verbatim
    Context.{h,cpp}         # NEW: builder API
    Equivalence.{h,cpp}     # checkEquivalence(MemState, MemState, pairing, solver)
    CMakeLists.txt          # core target `tile-smt` (z3 ONLY) [+ future tile-gpu-smt]
    test/                   # Z3-only unit tests, NO MLIR (AbstractFp/Memory/Context/Equivalence + SimpleTest.h)
  builder/                  # per-language builders (model a language onto the core)
    mlir/                   # shared for ALL MLIR langs: dtypeOf, Env(mlir::Value->Value), MemId
                            #   mapping, block-walk driver (State), arith.*/math.*/scf.* handlers
      CMakeLists.txt        # target tile-smt-builder-mlir (core + MLIR)
    triton/                 # Triton-specific: tt.* handlers
      CMakeLists.txt        # target tile-smt-builder-triton (core + builder-mlir + Triton dialect)
  bin/
    triton-tv.cpp           # validator main; target `triton-tv` links tile-smt-builder-triton
  test/ eval/ doc/ paper/   # existing (test/validator's MLIR-needing tests may move under builder/)
```
Note: the existing `tv/semantics/` is refactored *in place* into the MLIR-free
core — `Env`/`State` + the `mlir/` handlers move OUT to `builder/`, and
`triton-tv.cpp` moves to `bin/`. Core headers+impl sit directly in `semantics/`
(no include/lib split); the include root stays `tv/`, so code keeps
`#include "semantics/Context.h"` (matches today's `#include "semantics/..."`).
`tile-gpu-smt/` (a second core lib, also built from `semantics/`) is deferred to
M2; M0 doesn't need it, so no dead code.

### 1.2 The Z3-only `tile-smt` CMake target (built from `tv/semantics/`)
New `tv/semantics/CMakeLists.txt` replicates the Z3-find block at
`tv/CMakeLists.txt:1-23` (hoisting it to share is an optional later cleanup; for
M0 duplicate to keep the change local):
```cmake
add_library(tile-smt STATIC
  semantics/Types.cpp semantics/FpModel.cpp semantics/AbstractFp.cpp
  semantics/Memory.cpp semantics/Context.cpp semantics/Equivalence.cpp)
target_include_directories(tile-smt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)  # tv/ root → #include "semantics/..."
# Z3 ONLY — deliberately no ${triton_libs}, no MLIRIR, no MLIRSupport.
if(TARGET z3::libz3)
  target_link_libraries(tile-smt PUBLIC z3::libz3)
else()
  target_include_directories(tile-smt PUBLIC ${Z3_INCLUDE_DIRS})
  target_link_libraries(tile-smt PUBLIC ${Z3_LIBRARIES})
endif()
option(TILE_SMT_BUILD_TESTS "Build tile-smt Z3-only unit tests" ON)
if(TILE_SMT_BUILD_TESTS)
  add_subdirectory(test)
endif()
```
Contrast with today's `TritonTVSemantics` (`tv/CMakeLists.txt:31-52`) which links
`${triton_libs} MLIRIR MLIRSupport`: `tile-smt` links none of those — the
mechanically checkable success criterion. Keep default flags (exceptions on;
z3++ uses `z3::exception`). `tv/CMakeLists.txt` does
`add_subdirectory(semantics)` + `add_subdirectory(builder/mlir)` +
`add_subdirectory(builder/triton)` (before the `triton-tv` tool), all under
`tv/` — keeps tv wiring inside `tv/`.

### 1.3 tile-smt unit tests without MLIR
New `tv/semantics/test/CMakeLists.txt` mirrors `add_tv_test`
(`tv/test/validator/CMakeLists.txt:19-45`) but `LIBS` = `tile-smt` only (no MLIR,
no `${triton_libs}`). Copy the dependency-free `SimpleTest.h` (`SimpleTest.h:1-19`)
into `semantics/test/`. Migrated tests drop MLIR: today `AbstractFpTest.cpp:18`
builds `mlir::Float32Type::get(&mlirCtx)`, `MemoryModelTest.cpp:57-60` builds
`mlir::IntegerType`/`Float32Type` → become `DType::F32`/`DType::I32`. This is the
proof the core compiles/runs without MLIR.

### 1.4 The builder targets link core + MLIR
Replace today's single `TritonTVSemantics` (`tv/CMakeLists.txt:31-52`) with two
builder targets: **`tile-smt-builder-mlir`** (links `tile-smt` + `MLIRIR`
`MLIRSupport` + `${triton_libs}` as needed for arith/math/scf + shared tools) and
**`tile-smt-builder-triton`** (links `tile-smt-builder-mlir` + Triton dialect libs
for `tt.*`). `triton-tv` links `tile-smt-builder-triton` (transitively core+MLIR),
keeping its target name/output path. The `tv/test/validator` exes that need MLIR
link the relevant builder target(s); the Z3-only `tile-smt` tests link only
`tile-smt`. (Only the core is MLIR-free; builders deliberately depend on MLIR.)

---

## 2. Neutral types

### 2.1 Concrete definitions (`semantics/Types.h`, `Value.h`)
```cpp
namespace tile_smt {
enum class DType { I1, I8, I16, I32, I64, F16, BF16, F32, F64, Ptr };
using Shape = std::vector<int64_t>;          // was llvm::SmallVector<int64_t>
enum class MemId : uint32_t {};              // opaque provenance handle
enum class FPMode { Abstract, Real, IntegerRange, FPA };   // was Semantics::FPMode (Memory.h:25-30)

struct Scalar { z3::expr e; DType ty; };
struct Tensor { z3::expr e; Shape shape; DType elem; std::optional<MemId> ptrBase; };  // ptrBase set iff elem==Ptr
struct Ptr    { z3::expr e; DType pointee; MemId base; };
using Value = std::variant<Scalar, Tensor, Ptr>;

z3::sort getElemSort(z3::context&, DType, FPMode);   // was Memory.cpp:47-73
unsigned getByteWidth(DType);                        // was Memory.cpp:75-83
std::pair<unsigned,unsigned> fpExpSigBits(DType);    // was Memory.cpp:14-20
}
```
Load-bearing deltas vs today's structs (`Memory.h:33-64`):
- `Z3Scalar.mlirType` (`mlir::Type`) → `Scalar.ty` (`DType`).
- per-value `fpMode` field **dropped** (`Memory.h:36,49`) — it's always the run's
  single mode, which lives on `Context`/`Memory`. (Optional: keep it — it's a
  neutral enum — if minimizing churn; dropping is cleaner.)
- `Z3Tile.elemType` → `Tensor.elem` (`DType`, `=Ptr` for pointer tiles) +
  `Tensor.ptrBase` (`optional<MemId>`, was `mlir::Value`, `Memory.h:50`).
- `Z3Ptr.pointeeType`→`Ptr.pointee` (`DType`); `Z3Ptr.baseArg` (`mlir::Value`,
  `Memory.h:60`) → `Ptr.base` (`MemId`).

`MemId` is a plain enum → usable as a `std::map` key directly (contrast
`mlir::Value` needing `ValuePtrLess`, `Env.h:19-23`; that comparator stays in the
adapter for the `Env` map).

### 2.2 `mlir::Type -> DType` (in the **adapter**)
`DType dtypeOf(mlir::Type)` lives in the adapter (needs MLIR); all the inline
classifications below collapse to it.

| MLIR type test | → | current site(s) |
|---|---|---|
| `IntegerType` w=1 | `I1` | Memory.cpp:49-52, ArithOps.cpp:88-92 |
| `IntegerType` w=8/16/32/64 | `I8/I16/I32/I64` | Memory.cpp:49-53,75-79 |
| `Float16Type` | `F16` | Memory.cpp:16 |
| `BFloat16Type` | `BF16` | Memory.cpp:17, AbstractFp.cpp:14-16 |
| `Float32Type` | `F32` | Memory.cpp:18 |
| `Float64Type` | `F64` | Memory.cpp:19 |
| `triton::PointerType` | `Ptr` (+ pointee via `dtypeOf`) | State.cpp:41, TritonOps.cpp:118 |
| `RankedTensorType` | element `DType`+`Shape` | ArithOps.cpp:66-71, TritonOps.cpp:47-49 |

Inverse (`DType->width`, `->exp/sig bits`) moves to the **core** (`getByteWidth`/
`fpExpSigBits`, bodies at Memory.cpp:14-20,75-83).

### 2.3 `mlir::Value(ptr arg) -> MemId` (in the **adapter**)
Today provenance is the ptr-arg `mlir::Value`, threaded via `Z3Ptr.baseArg`/
`Z3Tile.ptrBase` and used as the `State::ptrMems` key (`State.h:37`). In M0:
- Adapter owns `map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem` + a counter.
- `initFromFunc` (`State.cpp:41-49`): per ptr arg, mint `MemId`, create the core
  `Memory` in `MemState.mems[id]`, bind `env[arg]=Ptr{addr,pointee,id}`.
- `Tensor.ptrBase` becomes a `MemId`, set at ptr `tt.splat` (`TritonOps.cpp:82`),
  propagated by `tt.addptr` (`TritonOps.cpp:110,131-132`).
- `tt.load`/`store` resolve `Memory` via the `MemId` (was `mlir::Value`,
  `TritonOps.cpp:171-173,197-202`) into `MemState.mems`.

---

## 3. What moves into `tile-smt` vs stays in the adapter

| Component (today) | Destination | Notes |
|---|---|---|
| `FPMode` (`Memory.h:25-30`) | **core** `Types.h` | already neutral |
| `AbstractFp`(+Registry) (`AbstractFp.{h,cpp}`) | **core** | ctor/registry take `DType`; `suffixFor`/key drop the width+bf16 hack (`AbstractFp.cpp:11-17,259-275`). Pure Z3 otherwise. = FpModel mode (a). |
| `getElemSort/getByteWidth/getFPExpSigBits/toBV` (`Memory.cpp:14-83`) | **core** | take `DType`; `toBV` stays private to Memory |
| `Memory` (`Memory.{h,cpp}`) | **core** | keep store lambda (`Memory.cpp:146-198`) + `readBytes` (`:94-103`) VERBATIM; `Z3Tile`→`Tensor`; still holds `z3::context&` (risk 6.1) |
| `Z3Scalar/Z3Tile/Z3Ptr`/`Z3Value` (`Memory.h:33-64`) | **core** `Value.h` | see §2.1 |
| `MemState` (`map<MemId,Memory>`) | **core** | replaces `State::ptrMems` (`State.h:37`) |
| op **semantics** in `ArithOps.cpp`/`TritonOps.cpp` | **core** `Context` methods | `applyBinaryOp` core (`ArithOps.cpp:33-63`), make_range/splat/addptr/reduce exprs (`TritonOps.cpp`) |
| op **dispatch + IR reading** (`llvm::cast<..Op>`, getOperand, attrs, env) | **adapter** | handlers: lookup operands → `dtypeOf` → `Context.<op>` → bind |
| `handleArithConstant` attr decode (`ArithOps.cpp:77-157`) | **adapter** decode; **core** `constInt/constFloat/splatConst/denseIntTile/denseFloatTile` | attribute types are MLIR → decode in adapter; z3 tile build in core |
| `Env`+`ValuePtrLess`+`makeSymbolicValue`+`initFuncArgs` (`Env.{h,cpp}`) | **adapter** | `mlir::Value`-keyed; `makeSymbolicValue` → `dtypeOf` then core `freshInput/freshPtr` |
| `State` (`State.{h,cpp}`) | **adapter** driver | holds `Context&`+`MemState`+`Env`+`map<Value,MemId>`; dispatch (`State.cpp:79-119`) + interpretIf/For/While stubs (`:125-153`) stay |
| `checkEquivalence` (`State.cpp:159-183`) | **core** `Equivalence.cpp` | takes `(MemState,MemState,span<pair<MemId,MemId>>,solver)`; witness body (`:171-181`) VERBATIM |
| `triton-tv.cpp` | **adapter** entry | parse, build two drivers, pair args→MemId, call core check; dead `floatingSATTest/realSATTest` (`:43-161`) left untouched (out of scope) |

`program_id` → **core** `Context::programId(int axis)` (fresh shared `BV32`,
names `program_id_x/y/z` so both programs share the symbol, `TritonOps.cpp:24-30`);
adapter just forwards `pidOp.getAxis()`.

---

## 4. Builder API surface for add + softmax (`semantics/Context.h`)
```cpp
namespace tile_smt {
class Context {
public:
  explicit Context(z3::context& z, FPMode mode);
  z3::context& z3();  FpModel& fp();

  Scalar programId(int axis);                                      // TritonOps.cpp:21-37
  Value  freshInput(DType, const Shape&, const std::string& name);
  Ptr    freshPtr(DType pointee, MemId, const std::string& name);  // State.cpp:43-44

  Scalar constInt(int64_t v, DType);                               // ArithOps.cpp:88-92
  Scalar constFloatBits(uint64_t ieeeBits, DType);                 // ArithOps.cpp:145-149
  Tensor splatConst(const Scalar&, const Shape&);                  // ArithOps.cpp:21-27
  Tensor denseIntTile(std::span<const int64_t>, DType, const Shape&);      // ArithOps.cpp:106-117
  Tensor denseFloatTile(std::span<const uint64_t>, DType, const Shape&);   // ArithOps.cpp:133-141
  Tensor iota(int64_t start, const Shape&, DType);                 // make_range, TritonOps.cpp:52-54

  Value add(const Value&, const Value&);   // addi/addf   ArithOps.cpp:163-171,317-330
  Value sub(const Value&, const Value&);   Value mul(const Value&, const Value&);
  Value andOp(const Value&, const Value&); // bool && vs bitwise &   ArithOps.cpp:193-209
  Value extsi(const Value&, DType dst);                            // ArithOps.cpp:215-252
  enum class IPred { eq,ne,slt,sle,sgt,sge,ult,ule,ugt,uge };
  Value cmpInt(IPred, const Value&, const Value&);                 // ArithOps.cpp:258-287
  Value div(const Value&, const Value&);                          // ArithOps.cpp:332-335
  Value maxnum(const Value&, const Value&);                       // ArithOps.cpp:337-340
  Value exp(const Value&);                                        // ArithOps.cpp:346-367

  Tensor splat(const Scalar&, const Shape&);                      // TritonOps.cpp:79-80
  Tensor splatPtr(const Ptr&, const Shape&);                      // carries MemId, TritonOps.cpp:81-82
  Ptr    addPtr(const Ptr&, const Scalar& off);                   // scalar, TritonOps.cpp:102-111
  Tensor addPtr(const Tensor& ptrs, const Tensor& off);          // tile,   TritonOps.cpp:114-136

  Scalar reduce(const Tensor&, int axis,
                const std::function<Value(const Value&,const Value&)>& combine); // TritonOps.cpp:218-265
};
}
```
- `add/sub/mul` are polymorphic int-vs-float on `DType` (branch to BV ops or
  `FpModel`), realizing the design's single `add(Value,Value)`.
- `reduce` takes a combine callback; the adapter turns the `tt.reduce` region into
  `[&](a,b){return ctx.maxnum(a,b);}`; core folds over `[0,n)` — no region notion.
- both `addPtr` overloads (scalar row-base + tile elementwise).
- `dot`/multi-dim reduce are **M1**, not M0.

Memory ops stay on `Memory`/`MemState` (already core), called by the load/store
handlers: `Memory::load(Tensor,Tensor,Tensor) const` (`Memory.cpp:105`),
`Memory::store(Tensor,Tensor,Tensor)` (`Memory.cpp:146`) — bodies unchanged,
`Z3Tile`→`Tensor`; adapter resolves `Memory&` from `MemState.mems[tensor.ptrBase]`.

### 4.4 Equivalence + MemId pairing
`triton-tv.cpp:224-283` pairs `srcArgs[i]`↔`tgtArgs[i]`: (a) input mem/ptr equal
(`:234-237`), (b) scalar/tensor inputs equal (`:240-243`), (c) witness disjunction
over paired memories (`:272-283`). In M0: adapter builds
`vector<pair<MemId,MemId>>` from the positional pairing; (a)/(b) stay adapter
(walk both Envs/MemStates); (c) → core `checkEquivalence(...)`, body copied from
`State.cpp:171-181`.

---

## 5. Ordered migration sub-steps (each small, compiling, eval+tests green)
Strategy: stand up the core behind an adapter shim, peel handlers one group at a
time, delete the shim last. Because the core directory *is* `tv/semantics/`
(refactored in place), the migration is: (i) grow `semantics/` into the MLIR-free
core; (ii) move `Env`/`State`/`mlir/` handlers OUT to `builder/{mlir,triton}/`;
(iii) move `triton-tv.cpp` to `bin/`. "shim" below = a temporary compatibility
header / `using`-alias so intermediate steps still compile and stay green.

- **Step 0 — scaffold the core lib.** Add `tv/semantics/Types.h`
  (DType/Shape/MemId/FPMode) + `Types.cpp`, `tv/semantics/test/` (one
  `TypesTest.cpp` + `SimpleTest.h`), and `tv/semantics/CMakeLists.txt` defining the
  Z3-only `tile-smt` target; wire `add_subdirectory(semantics)` in
  `tv/CMakeLists.txt`. The existing coupled `semantics/*.cpp` keep building under
  the old `TritonTVSemantics` target for now (coexist). Verify: `ninja tile-smt
  tile-smt-tests` green; `run_eval.py all` untouched → green.
- **Step 1 — AbstractFp → core (DType-based) + adapter shim.** Port
  `AbstractFp.{h,cpp}`; delete `semantics/AbstractFp.{h,cpp}`, replace with a shim
  header (`#include "semantics/AbstractFp.h"` + `using`) and an adapter
  `getFp(Registry, mlir::FloatType)` wrapper via `dtypeOf`. Rewrite AbstractFpTest
  into the Z3-only core test. Verify: all ctest + `run_eval.py all` green.
- **Step 2 — Memory + value wrappers + MemId/MemState → core.** Port
  `Memory.{h,cpp}`, `Z3*`→`Scalar/Tensor/Ptr`, keep store lambda/readBytes
  verbatim, introduce `MemId`/`MemState`. Adapter gets `dtypeOf` + `ptrBase/baseArg`
  flip to `MemId` (sites `TritonOps.cpp:82,110,131-132,171-173,197-202`). Biggest
  step — may split 2a (types+Memory in core, ptrMems still keyed via a MemId shim)
  / 2b (switch ptrMems to MemId). Verify: core `MemoryTest` (Z3-only) green;
  `run_eval.py all` green (encoding untouched → identical verdicts).
- **Step 3 — introduce `Context`; peel elementwise/structural/reduce.** Bodies
  lifted from `ArithOps.cpp`/`TritonOps.cpp`; rewrite each `handle*` to lookup →
  `ctx.<op>` → bind, one op-group at a time, re-eval after each. Verify per group:
  `run_eval.py all` green; add `ContextTest` (Z3-only).
- **Step 4 — `checkEquivalence` → core; `State` = driver.** Add
  `Equivalence.{h,cpp}`; rewire `triton-tv.cpp:272-283` to build the MemId pairing
  and call it. Verify: all green; `permute_passes.py` (add+softmax) all-EQUIVALENT.
- **Step 5 — delete shims + verify boundary.** Repoint includes to `semantics/...`;
  run the §7 DoD checklist.

---

## 6. Risks / pitfalls
- **6.1 `Memory` holds `z3::context&`** (`Memory.h:84`) → copy-assignment deleted,
  copy-construction works (tests rely on `Memory m1 = init;`, `MemoryModelTest.cpp:254`).
  Keep copy-constructible; `MemState.mems` insertion via `emplace`/piecewise
  (as `State.cpp:47-49`). Do NOT switch to `z3::context*` (would churn every
  `ctx.`→`ctx->` and risk the encoding).
- **6.2 `State`/`Context` not assignable** (ref members). Keep the
  `std::optional::emplace` threading (`State.cpp:62-73`; reduce fold
  `TritonOps.cpp:246-253`). Never `s = interpret(...)`.
- **6.3 `tt.reduce` region interpretation stays in the adapter** (`TritonOps.cpp:231-255`);
  core `reduce` only gets a callback. Don't move region-walking into core.
- **6.4 Preserve witness/lambda encoding exactly** (`Memory.cpp:178-197` +
  `State.cpp:164-181`); `array != array` extensionality returns `unknown`
  (`Memory.cpp:164-177`). Both halves move together, unchanged; lane order
  (last-writer-wins) + little-endian packing (`Memory.cpp:94-102`) must not change.
  Guarded by `MemoryModel.LargeTileWitnessEquivalence` (`MemoryModelTest.cpp:341-370`).
- **6.5 tile-smt tests never include MLIR.** After Step 5,
  `grep -rE '#include "mlir|mlir::|triton::' tv/semantics/` empty. `EnvTest`/`StateTest`
  legitimately need MLIR → stay under `tv/test/validator/`.
- **6.6 DenseElementsAttr iteration + i1 bool carrier are adapter concerns**
  (`ArithOps.cpp:106-117,133-141`); z3 store-chain build → core `denseIntTile/
  denseFloatTile(span<bits>)`. `DType::I1`→`bool_sort` mapping in core `getElemSort`
  to match `Memory.cpp:49-52`.
- **6.7 f16 vs bf16** — today a width+`isa<BFloat16Type>` hack + magic key
  (`AbstractFp.cpp:263-266`); with `DType` key the registry directly. Verify
  `AbstractFp.RegistryDistinguishesF16AndBF16` (`AbstractFpTest.cpp:245-257`).
- **6.8 eval binary path must not move** (`tv/eval/common.py:70-74,86-87`); keep
  `triton-tv` target name/output location.

---

## 7. Definition of Done (checkable)
1. `run_eval.py all` green (pairs + inequal + compile-options; NEQ still caught).
2. `permute_passes.py` on add + softmax: all permutations EQUIVALENT (matches the
   259-permutation baseline in `tv/CLAUDE.md` §5).
3. `ninja … triton-tv tv-validator-tests tile-smt-tests` clean; ctest
   `-R TestTritonTV` and the `tile-smt-tests` group both green.
4. `grep -rE 'mlir::|triton::|#include "mlir|#include "triton' tv/semantics/` empty.
5. `tile-smt` `target_link_libraries` is Z3-only (check `ninja -t commands tile-smt`).
6. tile-smt test exes link only `tile-smt`(+Z3), run with no MLIR present.
7. No `mlir::` in `semantics/Context.h` (all params neutral).
8. `triton-tv <a> <b>` exit codes unchanged (0/1/2), solver timing not regressed.
