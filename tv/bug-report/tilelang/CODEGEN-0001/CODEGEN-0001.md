# CODEGEN-0001: vectorized signed-integer floordiv/floormod emits invalid CUDA

Verdict: CONFIRMED
Arch:    generic (nothing Hopper-specific in the failing code)
Class:   **codegen bug, not a bit-equivalence finding.**  This is a hard compile
         failure with the *stock* pipeline, so it does not have the
         `input IR -> output IR, different bits` shape the rest of this
         directory uses.  It is filed here because it was found while
         triaging the corpus and it is cheap, minimal and reproducible.
Found:   2026-08-17, TileLang 0.1.12 (pypi wheel), H100 sm_90a, nvcc 12.8.93

## Summary

Any signed-integer `//` (floordiv) or `%` (floormod) inside a `T.Parallel` loop
that the vectorizer widens makes TileLang emit a **scalar** C ternary whose
condition is a **4-wide vector**. nvcc rejects it:

```
error: expression must have bool type (or be convertible to bool)
```

Signed floordiv lowers to a select that corrects the rounding direction. After
`VectorizeLoop` widens the loop, the select's operands are emitted element by
element (`__3.x`, `__3.y`, …) but the select itself is printed as one scalar
`cond ? a : b`, with the vector as the condition.

## Reproduction manual

```bash
source /home/youngzt/fuzz-tilelang/env.sh
python /home/youngzt/tv/triton/tv/bug-report/tilelang/CODEGEN-0001/repro.py
```

Expected output:

```
generated CUDA, the offending line:
  65: *(int4*)(Cl + (i_1 * 4)) = (__1 ? __3 : __4);
the condition it uses is a 4-wide vector, built element by element:
  __4.x = (__5.x-v__7.x)
  ...
nvcc:
  Compilation error:
  /tmp/.../tvm_kernels.cu(65): error: expression must have bool type (or be convertible to bool)

same kernel with the vectorizer off:
  compiled OK
```

## Minimal kernel

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

## Which cases fail

Same 64x64 kernel, only the expression and dtype changed:

| expression | dtype | result |
|---|---|---|
| `As[i, j] // 2` | int32 | **FAIL** |
| `As[i, j] // 2` | int16 | **FAIL** |
| `As[i, j] // 2` | int8 | **FAIL** |
| `As[i, j] // 3` | int32 | **FAIL** |
| `As[i, j] % 2` | int32 | **FAIL** |
| `T.floordiv(As[i, j], 2)` | int32 | **FAIL** |
| `As[i, j] // 2` | uint32 | ok — unsigned needs no sign correction, so no select |
| `As[i, j] / 2` | float32 | ok |

## Workaround

`pass_configs={"tirx.disable_vectorize": True}` compiles the same kernel fine.
That is the evidence that the vectorizer is what turns the select into a vector
select, and that the CUDA printer is what mishandles it.

## Where to look

* `VectorizeLoop` (`tilelang/transform`, `src/transform/`) widens the
  `tir.Select` produced by the signed floordiv lowering.
* `CodeGenTileLangCUDA` (`src/target/`, derived from TVM's CUDA codegen) prints
  the widened `Select` as one scalar `?:`. The fix is the same shape as the
  element-wise printing it already does for the arms: emit
  `out.x = cond.x ? a.x : b.x;` per lane, or materialise the condition into a
  scalar when the lanes are provably uniform.

## Artifacts

* `repro.py` — the standalone reproduction above
* `generated_bad.cu` — the full generated CUDA, line 65 is the bad one

## SMT reachability analysis

Not applicable: this is a compile failure, not a semantic difference between
two compilations. There is no "output IR" to compare against.
