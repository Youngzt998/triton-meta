# tv Code Navigation Guide

A map for inspecting the post-M0 code: the MLIR-free **core** (`tv/semantics/`,
namespace `tile_smt`) and the **builders** (`tv/builder/`, namespace
`Semantics`). ~2.8k lines of implementation (+~1.1k tests). Line numbers are
anchors — jump by symbol name if a line has drifted.

## 0. Three layers + dependency direction

```
bin/triton-tv.cpp        (315)   main: parse .ttir, build 2 drivers, pair args, call core check
   │  depends on
   ▼
builder/  (namespace Semantics)   walks mlir::Operation, reads MLIR types/attrs, calls core
   mlir/    DTypeOf · Env · State (walk + dispatch + cf stubs) · ArithOps
   triton/  TritonOps  (tt.*)
   │  depends on
   ▼
semantics/  (namespace tile_smt)  pure Z3 op semantics — NO MLIR
   Types · Value · AbstractFp · Memory · Context · Equivalence
   │
   ▼
   z3
```

Rule: **core includes no MLIR/Triton**. Verify: `grep -rE 'mlir::|triton::|#include "mlir' tv/semantics/` is empty; `libtile-smt.a` links z3 only.

## 1. Recommended reading order (shallow → deep, core before builder)

| # | Read | Why here | Focus |
|---|---|---|---|
| 1 | `semantics/Types.h` (45) → `Value.h` (62) | the value vocabulary everything else uses | `DType` (Types.h:18); `Scalar/Tensor/Ptr` (Value.h:30/41/52) + `Value` variant (:58). Note **Tensor.e = Array(BV32, elem)**; a pointer tile is a `Tensor` with `elem==Ptr` + `ptrBase` (MemId), not a tile of `Ptr` |
| 2 | `Memory.h` (77) → `Memory.cpp` (159) | the trickiest + perf-critical encoding | per-pointer `Array(BV64,BV8)` heap; `MemState` (Memory.h:71); **`store` = one `z3::lambda`** (cpp:~107), `load` (~64), `readBytes` (:53), `toBV` (:33). Last-writer-wins, little-endian. Real FP mode throws |
| 3 | `AbstractFp.h` (149) → `AbstractFp.cpp` (282) | the FP soundness boundary | opaque BV id + uninterpreted funcs; **`addAxioms` (cpp:206) has only 5 axioms** (5 reserved consts distinct; add/mul/max commutative; neg involutive). Deliberately **no** assoc / NaN-prop / zero-identity. `lt`/`le` throw (:183). `sum`/`dot` funcs declared but unused |
| 4 | `Context.h` (141) → `Context.cpp` (367) | **the authoritative modeled-op list** | all op signatures (Context.h:52-129); `applyBinaryOp` scalar-vs-tensor λ fold (cpp:67); int-vs-float routing (:191); `reduce` fold (:347); two `addPtr` overloads (:321/:331). `div`/`maxnum`/`exp` throw on non-float |
| 5 | `Equivalence.h` (42) → `Equivalence.cpp` (32) | how the verdict is decided | `checkEquivalence` (cpp:5): fresh witness addr, OR of `select(m1,w)!=select(m2,w)` over paired MemIds. Avoids `array!=array` extensionality (returns unknown) — paired with the single-lambda store |
| 6 | `builder/mlir/DTypeOf` (27/44) → `Env` (82/72) → **`State` (114/185)** → `ArithOps.cpp` (252) → `builder/triton/TritonOps.cpp` (242) | the MLIR side | `dtypeOf` (DTypeOf.cpp:10) = only Type→DType; `Env` = SSA→Value map + `makeSymbolicValue`; **`State.cpp:88-148` = the dispatch table**; `interpretIf/For/While` (:156/172/182) all **stub**; handlers are thin (lookup → `ctx.<op>` → bind) |
| 7 | `bin/triton-tv.cpp` (315) | how it's wired end to end | `initFromFunc` (:226); input-equality asserts (:235-261); `addAxioms` before check (:278); `checkEquivalence` (:297). Dead `floatingSATTest`/`realSATTest` are old scratch — skip |
| 8 | `semantics/test/ContextTest.cpp` (320) + `MemoryTest`/`EquivalenceTest` | living spec of core semantics | z3-only, one assertion per op; best doc of what each op *means* (e.g. add commutes, sub does not) |

## 2. End-to-end data flow (follow one run of `triton-tv a.ttir b.ttir`)

1. Parse both files to `mlir::ModuleOp`, get the `tt.func` (bin, MLIRParser + RegisterTritonDialects).
2. Build one `Context` + `State` per program, **sharing one `z3::context`** (bin:~211-230).
3. `State::initFromFunc`: per pointer arg mint a `MemId` + a symbolic `Memory`; per scalar/tensor arg build a symbolic `Value` (State + Env).
4. Assert corresponding args equal (bin:235-261): ptr args → initial heap + address equal; scalar/tensor args → value equal.
5. `State::interpretBlock` walks ops in order → dispatch (State.cpp:88) → handler → `ctx.<op>` / `Memory` → `env.bind`. State threads `env` + `memState`.
6. `checkEquivalence` (core): assert heaps differ at the witness address → `solver.check()`. UNSAT → EQUIVALENT (0); SAT → NOT (1, counterexample); unknown → (2).
7. FP axioms are emitted just before the check (bin:278).

## 3. Key design invariants — where to look, what to verify

| Invariant | Where | Verify |
|---|---|---|
| **MLIR-free core** | `tv/semantics/` | `grep -rE 'mlir::\|triton::\|#include "mlir' tv/semantics/` empty; `ar t libtile-smt.a` = core .o only |
| **Single-lambda store** (perf) | `Memory.cpp` `store` | one lambda per masked-tile store (not an n×byteWidth chain); lanes ascending → last-writer-wins |
| **Witness equivalence** (decidability) | `Equivalence.cpp` + store | uses `select(lambda, witness)`, never `array != array` |
| **FP soundness boundary** (audit this) | `AbstractFp.cpp` `addAxioms` | which axioms exist (5) vs deliberately absent (assoc / NaN / zero-identity) — could a missing law make a pass falsely (in)equivalent? |
| **MemId provenance** | `Value` `ptrBase`/`Ptr.base` + `State` | minted per ptr arg; propagated by splat/addptr; load/store resolve `Memory` via MemId |
| **Context stays neutral** | `Context.h` | no `mlir::`, no region concept; `reduce` takes a callback (region walked in builder); dtypeOf/attr decode live in builder |

## 4. Extension point (used repeatedly in M1)

Adding a modeled op = **three edits**:
1. a `Context` method (the Z3 body) in `semantics/Context.{h,cpp}`;
2. a thin handler in `builder/mlir/ArithOps.cpp` or `builder/triton/TritonOps.cpp` (lookup operands → `dtypeOf` → `ctx.<op>` → bind);
3. one dispatch line in `builder/mlir/State.cpp` (:88-148).

Ops not in the dispatch table hit `llvm_unreachable` (State.cpp:148) — a hard crash, not a "not modeled" flag (a candidate change for M1).

## 5. Questions worth asking while inspecting

- **Soundness**: are the FP axioms enough / too strong? mask semantics correct? last-writer-wins right? div/rem by zero? sign of extsi vs extui?
- **Completeness**: which ops are stub/unreachable (State.cpp dispatch + `interpret*`)? crash vs flag on unmodeled op?
- **Boundary**: is the core truly MLIR-free? did any MLIR concept leak into `Context`?
- **Encoding fidelity**: are `store`/`witness` byte-for-byte the pre-M0 encoding (verdicts/timing not regressed)?

See also: `tv/doc/m0-plan.md` (structure/CMake), `tv/paper/noticable.md` (store-blowup fix rationale), `tv/CLAUDE.md` (current state).
