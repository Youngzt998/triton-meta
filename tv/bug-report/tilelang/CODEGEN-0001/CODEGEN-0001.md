# CODEGEN-0001: vectorized signed-integer floordiv/floormod emits invalid CUDA

Verdict: **CONFIRMED — genuine compiler bug** (reviewed 2026-08-17)
Arch:    generic (nothing Hopper-specific in the failing code)
Line:    tilelang (TileLang 0.1.12, pypi wheel, H100 sm_90a, nvcc 12.8.93)
Class:   **hard compile failure with the stock pipeline**, not a bit difference.
         It does not have the `input IR -> output IR, different bits` shape the
         rest of this directory uses; `REVIEW-CRITERIA.md` §2 keeps this shape
         explicitly.

Summary: any signed-integer `//` (floordiv) or `%` (floormod) inside a
`T.Parallel` loop that the vectorizer widens makes TileLang emit a **scalar** C
ternary whose condition is a **4-wide vector**. nvcc rejects it:

```
error: expression must have bool type (or be convertible to bool)
```

---

## 1. Root cause in the source

Two independent scalar-shaped assumptions, in two different layers. Both have to
be understood; fixing either one hides this particular case, only the second one
fixes the class.

### 1a. `LowerIntrin` produces a `Select` for a *vector* signed floordiv

`3rdparty/tvm/src/tirx/transform/lower_intrin.cc`, `VisitExpr_(FloorDivNode*)`.
Three consecutive tests in that function are written for scalars and all three
miss when the type is `int32x4`:

```cpp
    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {   // line 111
      return op->a >> make_const(dtype, shift);          //  <-- not taken: op->b is a Broadcast, not an IntImm
    }
    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      ...
      if (const IntImmNode* b_as_intimm = op->b.as<IntImmNode>()) {              // line 121
        ...                                              //  <-- not taken: same reason
      }
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);
      if ((dtype == DataType::Int(32) || dtype == DataType::Int(64)) &&          // line 135
          support_bitwise_op_) {
        return rdiv + (rmod >> make_const(dtype, dtype.bits() - 1));   //  <-- not taken
      } else {
        return tirx::Select(rmod >= 0, rdiv, rdiv - make_const(dtype, 1));       // line 140  <-- TAKEN
      }
```

* Line 111 and 121: after `VectorizeLoop` the divisor is `Broadcast(2, 4)`, not
  an `IntImm`, so neither the shift shortcut nor the range shortcut fires.
* Line 135: `DataType` equality compares code, bits **and lanes**.
  `DataType::Int(32)` has `lanes == 1`, so `int32x4 == DataType::Int(32)` is
  **false** and the branchless `rdiv + (rmod >> 31)` form is skipped.

So the vector case falls to line 140 and a `Select` with a 4-lane condition is
created. `FloorMod` has the identical shape at lines 196 and 202, which is why
`%` fails too.

Order matters: `VectorizeLoop` runs at `tilelang/cuda/pipeline.py:185`, and
`LowerIntrin` runs much later at `tilelang/engine/lower.py:243` (dump files
`038_tl.VectorizeLoop.py` vs `081_tl.LowerIntrin.py`). So the `Select` is born
*after* vectorization. `src/transform/vectorize_loop.cc:425` does know how to
widen a `Select` — it just never sees this one.

### 1b. The CUDA printer has no vector `Select`

`3rdparty/tvm/src/target/source/codegen_c.cc:1048`:

```cpp
void CodeGenC::VisitExpr_(const SelectNode* op, std::ostream& os) {
  std::string cond = PrintExpr(op->condition);
  os << "(" << cond << " ? ";
  ...
```

No lane handling at all. `CodeGenTileLangCUDA` does **not** override
`SelectNode` (grep over `src/`: only the CuTeDSL, Python and Metal codegens do),
so the CUDA backend inherits this scalar printer. Every other vector operation
in the same expression *is* printed lane by lane — `PrintVecBinaryOp`
(`src/cuda/codegen/codegen_cuda.cc:1022`) emits `__2.x = ...; __2.y = ...;` and
so on. The `Select` is the one hole.

### The result

`generated_bad.cu`, lines 32-65. The condition `__1` is built lane by lane into
a `ushort4`, both arms are built lane by lane into `int4`, and then:

```c
  ushort4 __1;
    __1.x = (v_.x<=__2.x);  __1.y = ...;  __1.z = ...;  __1.w = ...;   // rmod >= 0, per lane
  int4 __3;  ...                                                       // rdiv
  int4 __4;  ...                                                       // rdiv - 1
  *(int4*)(Cl + (i_1 * 4)) = (__1 ? __3 : __4);                        // line 65 -- scalar ?: on a ushort4
```

`__1` is a `ushort4`, which has no conversion to `bool`, so nvcc stops.

### Fix shape

* **Narrow fix**, hides this case: at `lower_intrin.cc:135` and `:196` compare
  the element type (`dtype.element_of() == DataType::Int(32)`) so vectors also
  get the branchless `rdiv + (rmod >> 31)` form.
* **Real fix**, closes the class: give `CodeGenTileLangCUDA` a `SelectNode`
  override that, when `op->dtype.lanes() > 1`, emits one `out.x = cond.x ? a.x :
  b.x;` per lane, exactly as `PrintVecBinaryOp` already does. The Metal codegen
  (`src/metal/codegen/codegen_metal.cc:383`) already has its own `SelectNode`
  override; CUDA does not. Without this, `int8`/`int16` floordiv still hits it
  (line 135 excludes those widths even for scalars) and so does any other pass
  that creates a vector `Select`.

## 2. Reproduction

```bash
source /home/youngzt/fuzz-tilelang/env.sh
python /home/youngzt/tv/triton/tv/bug-report/tilelang/CODEGEN-0001/repro.py
```

Verified 2026-08-17, output:

```
generated CUDA, the offending line:
  65: *(int4*)(Cl + (i_1 * 4)) = (__1 ? __3 : __4);
the condition it uses is a 4-wide vector, built element by element:
  __4.x = (__5.x-v__7.x);
  ...
nvcc:
  Compilation error:
  /tmp/.../tvm_kernels.cu(65): error: expression must have bool type (or be convertible to bool)

same kernel with the vectorizer off:
  compiled OK
```

No input draw and no seed are involved: this fails at compile time, before any
GPU work. `TILELANG_DISABLE_CACHE=1` (set by `env.sh`) still matters, otherwise
a cached artifact can hide the failure.

### Minimal kernel

```python
@T.prim_func
def main(A: T.Tensor((64, 64), "int32"), C: T.Tensor((64, 64), "int32")):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        As = T.alloc_shared((64, 64), "int32")
        Cl = T.alloc_fragment((64, 64), "int32")
        T.copy(A[0, 0], As)
        for i, j in T.Parallel(64, 64):
            Cl[i, j] = As[i, j] // 2      # signed floordiv
        T.copy(Cl, C[0, 0])
```

### Which cases fail

Same 64x64 kernel, only the expression and dtype changed:

| expression | dtype | result | why |
|---|---|---|---|
| `As[i, j] // 2` | int32 | **FAIL** | line 135 guard misses `int32x4` |
| `As[i, j] // 2` | int16 | **FAIL** | width not in the guard at all |
| `As[i, j] // 2` | int8 | **FAIL** | same |
| `As[i, j] // 3` | int32 | **FAIL** | same as int32 above |
| `As[i, j] % 2` | int32 | **FAIL** | `FloorMod`, lines 196/202, same shape |
| `T.floordiv(As[i, j], 2)` | int32 | **FAIL** | same node |
| `As[i, j] // 2` | uint32 | ok | unsigned: `CanProveGreaterEqual(a, 0)` holds, plain `truncdiv`, no `Select` |
| `As[i, j] / 2` | float32 | ok | float path, no sign correction |

### Workaround

`pass_configs={"tirx.disable_vectorize": True}` compiles the same kernel fine —
a scalar `Select` prints as a valid scalar ternary.

## 3. Where it reproduces

**Any target that uses a `CodeGenC`-derived printer, on any GPU.** The judgement
rests on:

* Neither layer is arch-aware. `lower_intrin.cc` is target-independent TVM;
  `CodeGenC::VisitExpr_(SelectNode*)` is the generic C printer.
* The failing code contains no Hopper construct: no wgmma, no TMA, no mbarrier,
  no cluster, no warp specialization. It is a shared-memory copy, an
  elementwise loop and a copy out.
* It is a **compile-time** failure, so the GPU model, the driver and even having
  a GPU are irrelevant. Only nvcc's front end sees it.

Verified only on CUDA / sm_90a / nvcc 12.8.93 here. The Metal backend has its
own `SelectNode` override so it is probably not affected; ROCm and WebGPU were
not checked.

## 4. Plain-language explanation

Dividing a signed integer and rounding **down** is not what the hardware's
divide instruction does — the instruction rounds toward zero. So the compiler
writes `if the remainder is negative, subtract one`. That "if" becomes a
`cond ? a : b` in the generated C.

When the loop is vectorized, four elements are handled at once, so the "is the
remainder negative" answer is four answers, one per element. The compiler builds
those four answers correctly, and builds both alternatives correctly, but then
writes the choice as a single `?:` — as if there were only one answer. C has no
such thing as a `?:` driven by four booleans, so the CUDA compiler refuses the
file and the kernel never builds.

## 5. Can an SMT solver model this?

**Not applicable in the equivalence sense, and that is the useful answer.**

* There is nothing to compare. One side does not exist: compilation stops, so
  there is no second program and no output bits. An equivalence checker needs
  two programs.
* The bug is also not a *semantic* mistake anywhere in the IR. The IR is fine:
  `Select(cond_4lane, a_4lane, b_4lane)` means an elementwise select, and that
  is exactly the right meaning for a vectorized floordiv. If the printer emitted
  four scalar selects, an SMT check would prove the vector and scalar forms
  equivalent — correctly, and while missing the point.
* The defect lives **below the IR**, in the text the printer emits. That places
  it out of reach of any IR-level checker, in the same class as a wrong PTX
  encoding: a tool would have to model the C printer, not the IR.

What *would* catch it is a different and much cheaper kind of tool: a
well-formedness check on the printer, i.e. "every IR node kind the vectorizer
can produce with `lanes > 1` must have a vector printing path". That is a
coverage/lint question, not a solver question. A compile-only sweep — build
every corpus kernel with the stock pipeline and just check that nvcc accepts the
output — finds this whole class for free and needs no GPU.

## 6. Confidence

**High.** Every step was read in the source and matched against the emitted
text: the `Select` at `lower_intrin.cc:140`, the three scalar-shaped tests above
it, the missing `SelectNode` override in the CUDA codegen, and line 65 of
`generated_bad.cu` where a `ushort4` is used as a ternary condition. The case
table lines up with the source: unsigned is fine because it never reaches the
`Select`, float is fine because it takes the float path, all signed widths fail.
Turning the vectorizer off fixes it, which pins the vector width as the trigger.

The remaining doubt is about scope, not about the bug:

* Is `int32` floordiv inside `T.Parallel` a supported pattern, or is TileLang
  entitled to require the user to avoid it? It is plain Python `//` on a plain
  `int32` tile, so this reads as supported, but no upstream statement was found.
* Whether the same failure appears on ROCm and WebGPU was not checked; the
  reasoning says yes, the measurement covers CUDA only.

Both are settled by the same cheap experiment: run the minimal kernel under each
backend and check whether the backend's codegen has a vector `Select` path.

## Artifacts

* `repro.py` — the standalone reproduction above
* `generated_bad.cu` — the full generated CUDA; line 65 is the bad one
