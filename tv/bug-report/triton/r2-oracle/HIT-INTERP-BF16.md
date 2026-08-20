> **Read this first — this is not a pass-toggle finding.** It was found as a
> by-product of building the R3 oracle (`/home/youngzt/fuzz/r2-oracle`), not by
> the sweep. There is no "reference pipeline versus candidate pipeline" here and
> no culprit pass. **Do not try to fit it to the Group A / Group B criteria in
> `REVIEW-CRITERIA.md` §1** — those criteria are written for "same kernel, two
> pass sets, different bits", and none of them apply. Judge it as §2's last
> bullet ("compile failure, crash, or invalid generated code — different shape
> from a bit difference") judges its class: a defect of a different shape. The
> six-part deep analysis in §4 is filled in below in full.

# HIT-INTERP-BF16: interpreter mode does bf16 and fp8 arithmetic on the raw bit patterns

Verdict: **CONFIRMED** (reproduced from scratch, both arms, three times)
Arch:    **generic** — the wrong arm is pure Python and numpy and runs with no
         GPU visible at all (measured with `CUDA_VISIBLE_DEVICES=""`)
Oracle:  none of the campaign's oracles. Found while building O1.
Line:    r2-oracle (found by, not produced by)
Culprit: `python/triton/runtime/interpreter.py` — no compiler pass involved
SMT verdict: **SMT-decidable in principle, but only if the question is
         re-framed from "pass versus pass" to "builder versus builder", and
         only under the exact bit-to-bit FP axiom profile.** See §5.

## Summary

`TRITON_INTERPRET=1` makes Triton run a kernel with a Python/numpy interpreter
instead of compiling it. numpy has no `bfloat16` type and no `float8` type, so
the interpreter stores those values as `uint16` and `uint8` — the raw IEEE
encoding kept as an unsigned integer. The elementwise arithmetic path then
applies the numpy operator straight to that storage. So `a + b` on two
`bfloat16` tensors adds their 16-bit encodings as integers, with wraparound,
and returns the result as if it were a `bfloat16`. No error, no warning: the
kernel runs to completion and stores a wrong number.

Measured, inputs `-4..3` and `-3..4`:

| | interpreter | compiled (H100) | correct |
|---|---|---|---|
| `a + b`, bf16 | `-1.76e-38, -5.88e-39, inf, -1, 1, inf, -5.88e-39, -1.76e-38` | `-7, -5, -3, -1, 1, 3, 5, 7` | `-7, -5, -3, -1, 1, 3, 5, 7` |
| `a * b`, bf16 | `1.08e-19, 0, 0, 0, 0, 0, 0, 1.08e-19` | `12, 6, 2, -0, 0, 2, 6, 12` | `12, 6, 2, -0, 0, 2, 6, 12` |
| `tl.dot`, bf16 in / fp32 out | `9104769024, 15013498880, …` | `-9, -1, …` | `-9, -1, …` |

**On whether it is known — the honest answer, and it cuts both ways.**

* **bf16 is a documented limitation.** `docs/programming-guide/chapter-3/debugging.rst:68`
  has a "Limitations" section that says: *"It does not support operations on
  `bfloat16` numeric types. To perform operations on `bfloat16` tensors, use
  `tl.cast(tensor)` to convert the tensor to `float32`."* Triton's own test
  suite enforces the same thing with `pytest.skip("bfloat16 is not supported in
  the interpreter")` at `python/test/unit/language/test_core.py` lines 135,
  3814, 4107 and 7353, plus a comment at line 2187 ("Interpreter: Only
  bfloat16 <-> float32 is supported"). So for bf16 this report is **not**
  "undocumented silent wrongness". It is: **a documented limitation that is
  enforced nowhere, so instead of raising, the tool returns garbage.**
* **fp8 is not documented anywhere.** The Limitations list names only bf16 and
  indirect memory access. No doc mentions fp8 and the interpreter. No test in
  `python/test/` skips fp8 for the interpreter. And for fp8 the compiled path
  does not merely disagree — it **refuses to compile** (`RuntimeError:
  PassManager::run failed`, in `ConvertTritonGPUToLLVM`). So for fp8 the
  interpreter accepts and silently miscomputes a program that the compiler
  rejects outright.

Either way the fix is the same and small: on this path, raise instead of
returning a number. That is why it is worth filing.

---

## 1. Root cause in the source

Public checkout `/home/youngzt/fuzz/triton-public`, commit `3277063a6`,
`triton 3.8.0`. One file: `python/triton/runtime/interpreter.py`.

**Step one — the storage choice (defensible on its own).**
`_get_np_dtype` (starts line 146) has no numpy type to map these formats onto,
so it maps them to unsigned integers of the same width:

```python
# python/triton/runtime/interpreter.py:162-169
        # bfloat16 types are stored as uint16
        tl.bfloat16: np.dtype(np.uint16),
        # float8 types are stored as uint8
        tl.float8e5: np.dtype(np.uint8),
        tl.float8e5b16: np.dtype(np.uint8),
        tl.float8e4nv: np.dtype(np.uint8),
        tl.float8e4b8: np.dtype(np.uint8),
        tl.float8e4b15: np.dtype(np.uint8),
```

This by itself is fine. numpy really has no bf16, and holding the encoding is a
reasonable way to store the value. `TensorHandle` (line 28) keeps both facts
side by side: `data` is the `uint16` array, `dtype` is `tl.bfloat16`.

**Step two — the wrong step. This is the line.**

```python
# python/triton/runtime/interpreter.py:557-573
    def binary_op(self, lhs, rhs, op):
        tl_dtype = lhs.dtype.scalar

        if lhs.data.dtype == np.bool_ and rhs.data.dtype == np.bool_:
            ...
        else:
            output = op(lhs.data, rhs.data)      # <-- line 568, the defect
```

**Line 568 applies the numpy operator to `lhs.data`, which for bf16 and fp8
holds the encoding and not the number.** `binary_op` reads `lhs.dtype.scalar`
on line 558 and so it *knows* the value is a `bfloat16`, but it uses that only
to label the result on line 573. It never asks whether `data` is a real numpy
float for that `dtype`. There is a special case directly above it for `bool`
(lines 560-566, added for issue #10919, which is the same class of mistake:
numpy's storage type not meaning what the Triton type means) — but no case for
bf16 or fp8.

So `create_fadd = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)`
(line 575) becomes, for bf16, `np.add` on two `uint16` arrays: **unsigned
16-bit integer addition of the two IEEE encodings, wrapping at 2^16.**

The arithmetic checks out exactly. bf16 `-4` is `0xC080`, `-3` is `0xC040`.
`0xC080 + 0xC040 = 0x180C0`, which wraps to `0x80C0`. `0x80C0` read back as
bf16 is sign 1, exponent 1, mantissa 0x40, that is `-1.7632e-38` — which is the
first element the interpreter printed. Same for fp8 e4m3: `-4` is `0xC8`, `-3`
is `0xC4`, `0xC8 + 0xC4 = 0x18C`, wraps to `0x8C`, which is `-0.0234375` — the
first element of that row. The mechanism is not a guess; the output bits are
exactly unsigned addition of the input bits.

Two more entry points have the same defect:

* `unary_op`, line 677-678: `return TensorHandle(op(arg.data), arg.dtype.scalar)`
  — `create_fneg` is `np.negative` on `uint16`, so negating a bf16 wraps as an
  unsigned integer. Measured: `-(-4)` gives `1` instead of `4`.
* `create_dot`, line 715-722. It has a fix for fp8 (lines 718-721 convert both
  operands through fp16 with `_convert_float`) but **no branch for bf16**, so
  line 722 runs `np.matmul` over the `uint16` encodings. Measured: an entry that
  should be `-9` comes out as `9104769024`.

Reductions inherit it: `tl.sum` on bf16 keeps bf16 (`_pick_sum_dtype`,
`python/triton/language/standard.py:270-280`, returns the input dtype for
floats), and the fold calls `binary_op` for each step. Measured: sum of
`[-4,-3,-2,-1,0,1,2,3]` gives `-2` instead of `-4`.

**Why some bf16 ops are still right — and why that makes it worse.**
The interpreter has a set of one-op fixes, so bf16 partly works:

* `cast_impl`, lines 529-537: bf16 ↔ fp32 goes through `_convert_float`, which
  is a correct hand-written encoder. So casts are right.
* `create_fabs`, lines 680-688: masks the sign bit on the raw bits. Right by
  construction.
* `create_dot` for fp8, lines 718-721. Right.
* And in the frontend, `_promote_bfloat16_to_float32`
  (`python/triton/language/core.py:2965-2971`) casts bf16 up to fp32 before
  `tl.maximum`, `tl.minimum`, `tl.clamp`, `tl.max`, `tl.min`, `tl.cumsum`,
  `tl.cumprod`. Its comment says *"hardware doesn't support FMAX, FMIN, CMP for
  bfloat16"* — it is a **hardware** workaround, so it fixes the interpreter
  only by accident, and only for those seven functions. Note it says "CMP" but
  plain `<` does not go through it, and `<` on bf16 is measurably wrong in the
  interpreter (it compares the encodings as unsigned integers, so every
  negative number sorts above every positive one).

So a user who tries `tl.maximum` on bf16 in the interpreter gets the right
answer, and a user who tries `+` gets garbage, with nothing to tell them apart.

**Where the enforcement should be and is not.** `interpreter.py` contains no
guard, no warning and no raise for bf16 or fp8 anywhere. Its only two
`NotImplementedError`s are at lines 811 and 814, for `extern_elementwise` and
`inline_asm`. Those are the model to follow: a documented limitation should
raise, the way those two do.

---

## 2. Reproduction

Verified by running it, start to finish, on this box. Two scripts, both in
`HIT-INTERP-BF16/` next to this file. Each one re-launches itself twice as a
child process — once with `TRITON_INTERPRET=1` and once without — because
`TRITON_INTERPRET` is read when `@triton.jit` runs, so one process cannot do
both arms.

```bash
source /home/youngzt/fuzz/env.sh
CUDA_VISIBLE_DEVICES=0 python \
  /home/youngzt/tv/triton/tv/bug-report/triton/r2-oracle/HIT-INTERP-BF16/repro.py
CUDA_VISIBLE_DEVICES=0 python \
  /home/youngzt/tv/triton/tv/bug-report/triton/r2-oracle/HIT-INTERP-BF16/repro_dot.py
```

There is no seed and no saved tensor. The inputs are fixed whole numbers,
written in the script: `a = [-4,-3,-2,-1,0,1,2,3]`, `b = [-3,-2,-1,0,1,2,3,4]`.
They are whole numbers small enough that no format involved can round, so the
"correct" column is exact and both arms must match it bit for bit.

### Minimal kernel source

```python
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(out_ptr + i, tl.load(a_ptr + i) + tl.load(b_ptr + i))
```

Three lines. One grid program, eight lanes, no loop, no mask, no reduction.

### Observed

```
case                 interpreter (TRITON_INTERPRET=1)         compiled (GPU)                           correct                        interp==correct
add/float32          -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7     True
add/float16          -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7     True
add/bfloat16         -1.76324e-38, -5.87747e-39, inf, -1, 1,  -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7     False
                       bits interp=c0804080807f80bf803f807f4080c080  correct=e0c0a0c040c080bf803f4040a040e040
                       bits gpu   =e0c0a0c040c080bf803f4040a040e040  gpu==correct: True
add/float8e5         -9.15527e-05, -3.05176e-05, inf, -1, 1,  RuntimeError: PassManager::run failed    -7, -5, -3, -1, 1, 3, 5, 7     False
                       bits interp=86827cbc3c7c8286  correct=c7c5c2bc3c424547
add/float8e4nv       -0.0234375, -0.0078125, 256, -1, 1, 256, RuntimeError: PassManager::run failed    -7, -5, -3, -1, 1, 3, 5, 7     False
                       bits interp=8c8478b83878848c  correct=cecac4b838444a4e
mul/bfloat16         1.0842e-19, 0, 0, 0, 0, 0, 0, 1.0842e-19 12, 6, 2, -0, 0, 2, 6, 12                12, 6, 2, -0, 0, 2, 6, 12      False
                       bits interp=00200000000000000000000000000020  correct=4041c0400040008000000040c0404041
                       bits gpu   =4041c0400040008000000040c0404041  gpu==correct: True
max/bfloat16         -3, -2, -1, 0, 1, 2, 3, 4                -3, -2, -1, 0, 1, 2, 3, 4                -3, -2, -1, 0, 1, 2, 3, 4      True
max/float8e5         -4, -3, -2, -1, 1, 2, 3, 4               RuntimeError: PassManager::run failed    -3, -2, -1, 0, 1, 2, 3, 4      False
                       bits interp=c4c2c0bc3c404244  correct=c2c0bc003c404244
neg/bfloat16         1, 1.5, 2, 4, 0, -4, -2, -1.5            4, 3, 2, 1, -0, -1, -2, -3               4, 3, 2, 1, -0, -1, -2, -3     False
                       bits interp=803fc03f00408040000080c000c0c0bf  correct=804040400040803f008080bf00c040c0
                       bits gpu   =804040400040803f008080bf00c040c0  gpu==correct: True
sum/bfloat16         -2, -2, -2, -2, -2, -2, -2, -2           -4, -4, -4, -4, -4, -4, -4, -4           -4, -4, -4, -4, -4, -4, -4, -4 False
                       bits interp=00c000c000c000c000c000c000c000c0  correct=80c080c080c080c080c080c080c080c0
                       bits gpu   =80c080c080c080c080c080c080c080c0  gpu==correct: True
cmp/bfloat16         0, 0, 0, 0, 1, 1, 1, 1                   1, 1, 1, 1, 1, 1, 1, 1                   1, 1, 1, 1, 1, 1, 1, 1         False
cmp/float8e5         0, 0, 0, 0, 1, 1, 1, 1                   RuntimeError: PassManager::run failed    1, 1, 1, 1, 1, 1, 1, 1         False
upcast/bfloat16      -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7               -7, -5, -3, -1, 1, 3, 5, 7     True

9 of 13 cases: the interpreter's answer is wrong
```

and from `repro_dot.py`:

```
tl.dot / float32
   interpreter top-left 2x2 : [[-9.0, -1.0], [3.0, -9.0]]
   compiled    top-left 2x2 : [[-9.0, -1.0], [3.0, -9.0]]
   interpreter bits == gpu bits : True

tl.dot / bfloat16
   interpreter top-left 2x2 : [[9104769024.0, 15013498880.0], [10170155008.0, 16336838656.0]]
   compiled    top-left 2x2 : [[-9.0, -1.0], [3.0, -9.0]]
   correct     top-left 2x2 : [[-9.0, -1.0], [3.0, -9.0]]
   interpreter bits == gpu bits : False
   gpu bits == correct bits     : True
```

Note the last row of the first table: `upcast/bfloat16` loads bf16, casts to
fp32, adds, casts back — and it is correct on both arms. That is the workaround
the Limitations doc tells users to use, and it does work. It also shows the
inputs and the storage are fine; only the arithmetic path is broken.

### Which side is wrong

There is no ambiguity to argue about. `-4 + -3 = -7`. The compiled arm says
`-7`, the "correct" column computed independently with torch in fp32 says `-7`,
and the interpreter says `-1.76e-38`. The interpreter is wrong.

### Determinism evidence

The interpreter arm was run three times with `CUDA_VISIBLE_DEVICES=""` (no GPU
visible at all). All three runs produced a byte-identical result line
(`md5 2d967670be96085edce08b0dfed07df8`, three times). The compiled arm agrees
with the independently computed correct answer on every case where it compiles.
`repro.py` as a whole was run twice end to end with the same output.

### Also present in the Meta fork

Not specific to the public checkout. `/home/youngzt/tv/triton/python/triton/runtime/interpreter.py`
has the same mapping at its lines 161-168 and the same `binary_op` body at its
lines 464-465 (there without even the `bool` special case). Read only; nothing
was built or run there.

---

## 3. Where it reproduces

**Arch-independent, and here is what that rests on.**

The usual arch question — "does the failing IR contain wgmma, TMA, mbarrier,
cluster or warp specialization?" — does not apply, because the failing side
produces no IR at all. `TRITON_INTERPRET=1` bypasses compilation completely:
`InterpreterBuilder` is handed the same frontend calls that the MLIR builder
would get, and it returns numpy arrays instead of `mlir::Value`s. No TTIR, no
TTGIR, no LLVM, no PTX, no ptxas, no driver, no kernel launch.

That is an argument from reading the code, so I also measured it: the
interpreter arm produces the same wrong bits with `CUDA_VISIBLE_DEVICES=""`,
that is with no GPU visible to the process at all. It cannot depend on the
GPU because it never reaches one.

It follows that it is also backend-independent: nothing on the path reads the
target, the architecture, or the compute capability, so AMD and any future
backend see the identical behaviour. The only arch-dependent part of this
report is the *reference* arm, which ran on an H100 (sm90) and was right there.

The one real environment dependency is numpy, because the wrong answer is
numpy's unsigned-integer wraparound. That is defined behaviour in numpy and not
version-sensitive in a way that would change the verdict. Measured on
numpy 2.5.2, torch 2.13.0+cu130, python 3.12, triton 3.8.0 (`3277063a6`).

**Which dtypes.**

| dtype | interpreter storage | affected? |
|---|---|---|
| `float32`, `float64`, `float16` | native numpy float | no — measured correct |
| all integer types, `int1` | native numpy int / bool | no |
| `bfloat16` | `uint16` | **yes** |
| `float8e5` (e5m2) | `uint8` | **yes** |
| `float8e4nv` (e4m3fn) | `uint8` | **yes** |
| `float8e5b16`, `float8e4b8`, `float8e4b15` | `uint8` | **yes** by the same line; not separately measured, no torch dtype to feed them from |

**Which operations**, for bf16 (measured unless marked):

* wrong: `+`, `-`, `*`, unary `-`, all six float comparisons, `tl.sum` and any
  `tl.reduce`/`tl.scan` whose combine stays in bf16, `tl.dot`. (Measured: `+`,
  `*`, unary `-`, `<`, `tl.sum`, `tl.dot`. The rest are the same two lines, 568
  and 678.)
* right: casts to and from fp32, `tl.abs`, `tl.where`, load and store, and
  `tl.maximum` / `tl.minimum` / `tl.clamp` / `tl.max` / `tl.min` / `tl.cumsum` /
  `tl.cumprod` (frontend promotes them to fp32). Division is also right,
  because the frontend upcasts any float divide to fp32
  (`python/triton/language/semantic.py:85-86`).

For fp8 there is no frontend promotion at all, so even `tl.maximum` is wrong
(measured: `max(-4, -3)` returns `-4`). `tl.dot` and `tl.dot_scaled` are right
for fp8, because they were fixed by hand.

---

## 4. Plain-language explanation

Triton has a debug mode that runs your kernel as a Python program instead of on
the GPU. Python's numpy library has no bfloat16 number type, so this mode keeps
each bfloat16 value as a plain 16-bit integer — the value's bit pattern, stored
as a number.

That is fine for carrying the value around. The mistake is that when the kernel
asks to add two bfloat16 values, the debug mode adds those two integers. It adds
the bit patterns, not the numbers, and the answer overflows and wraps around
like an odometer. Then it labels the wrapped integer as a bfloat16 and hands it
back.

So `-4 + -3` comes out as `-0.0000000000000000000000000000000000000176`
instead of `-7`. And nothing complains. The kernel finishes, the numbers look
like numbers, and a person debugging their kernel sees wrong results and has no
reason to suspect the debugger rather than their own code.

The same thing happens for the fp8 types. There it is worse in one way: if you
try to compile that kernel for the GPU, the compiler refuses — it has no
instruction for adding two fp8 numbers. The debug mode does not refuse. It
gives you an answer.

Triton's docs do say the debug mode "does not support operations on bfloat16".
The problem is that nothing in the code enforces it. A limitation you have
written down but do not check is, from the user's chair, the same as no
limitation at all — until the numbers come out wrong.

---

## 5. SMT reachability analysis

**Assume the SMT validator does not exist yet and has no current limits.** The
question is whether the semantics at the root of this defect is capturable in
principle.

### 5.1 Root cause in semantic terms

There is no IR diff to point at, and that is the whole difficulty. On the
compiled side the frontend call `create_fadd` becomes `arith.addf` on `bf16`.
On the interpreter side the *same* frontend call is answered by a different
builder, which evaluates `np.add` on the operands' 16-bit encodings. Written as
one sentence:

> `arith.addf : bf16` is implemented as `arith.addi : i16` applied to the
> operands' IEEE encodings.

That is not a rounding difference, not a reassociation, not a precision choice.
It is a different operation.

### 5.2 Is the difference a pure input→output function difference?

Yes, as purely as it gets. The reproduction is one grid program, eight lanes, no
loop, no mask, no reduction, no shared memory, no barrier, no atomic, and no
inter-instance interaction. The wrong output follows deterministically from the
two input values — measured identical over three runs and, in fact, derivable
by hand from the input bits. Nothing about scheduling, concurrency, memory
ordering, the grid, or the hardware plays any part.

So on the classification axis the campaign cares about, this is the easy end of
the scale. The hard part is somewhere else.

### 5.3 Fundamental verdict

**`SMT-decidable in principle` — but only after re-framing the question, and the
re-framing is the interesting result.**

A translation validator compares two IRs: it lifts each to a common semantics
and asks whether they agree on all inputs. That question cannot even be *typed*
here, because the interpreter has no IR. It is not a pass. It is a second
implementation of the source language, and it produces numpy arrays where the
compiler produces `mlir::Value`s. Pointed at this defect as it is normally
posed — "is the output IR equivalent to the input IR?" — an SMT validator finds
nothing, because there is only one IR in the room and it is correct.

But there is a re-framing that works, and it is not a stretch, because it
matches how the tool is already built. `tv/` separates an MLIR-free core from
per-language *builders*, and a builder's one job is to map a surface language's
operations onto the core's abstract tensor operations. `InterpreterBuilder` is
exactly such a builder: same `TritonSemantic` frontend above it, different back
end below it. So the well-typed question is **builder versus builder**:

> Does `InterpreterBuilder.create_fadd` denote the same function as
> `TritonOpBuilder.create_fadd`?

That question is decidable, and cheaply. Both sides are finite bit-vector
functions. The interpreter side is a straight-line, shape-static numpy program
with no data-dependent control flow — `np.add` on `uint16` is exactly `bvadd`
on a 16-bit bit vector. The compiler side is `arith.addf` on bf16, which is
`fp.add` on SMT-LIB's `(_ FloatingPoint 8 8)`. The check is one formula over two
free 16-bit variables:

```
∀ x, y : BitVec(16) .
    bv_to_bf16( bvadd(x, y) )  ==  fp.add( bv_to_bf16(x), bv_to_bf16(y) )
```

It comes back SAT immediately. `x = y = 0x3F80` (that is `1.0 + 1.0`) already
does it: `bvadd` gives `0x7F00`, about `1.7e38`; `fp.add` gives `0x4000`, which
is `2.0`.

The other classifications do not fit. It is not `needs whole-grid modeling` —
one lane shows it. Not `needs a concurrency / memory-ordering model` — there is
no concurrency anywhere on the path. Not `below the IR level` — it is above the
IR, in a component that stands in for the IR. Not `expressible but impractical`
— see 5.6, the query is about as small as a query gets. The only honest caveat
is the scoping one above: it is decidable, but not by the check the tool is
normally asked to run.

### 5.4 Minimum semantic vocabulary required

Short list, and one item on it is the whole game.

1. **The value's storage and the value's meaning must be separate things in the
   model.** This is the requirement. `TensorHandle` carries `data: np.uint16`
   and `dtype: tl.bfloat16` at the same time, and the defect is precisely that
   an operation meant for the *meaning* was applied to the *storage*. A model
   in which "this value is a bf16" is a single fact cannot express the defect at
   all — both sides look like a bf16 add. The model needs "this value is a bf16
   whose representation is these 16 bits", plus the two conversions
   `fp.to_ieee_bv` and its inverse, so it can say which of the two an operation
   touched.
2. **IEEE-754 at non-standard widths.** bf16 is 1+8+7, so `(_ FloatingPoint 8 8)`.
   e5m2 is `(_ FloatingPoint 5 3)`. Both are inside SMT-LIB's `FloatingPoint`
   sort as it is parameterised, so nothing new is needed. e4m3fn is *not* IEEE
   (no infinities, a single NaN encoding), so it needs a hand-written encoding —
   still a total function on 8 bits, so still finite and still cheap.
3. **Fixed-width integer wraparound**, i.e. plain `bvadd` — because the wrong
   answer *is* the wraparound.
4. Nothing else. No memory model beyond a flat array of values. No layouts, no
   shared memory as a second address space, no async copy, no barrier, no loop
   as a reduction over a symbolic trip count. This defect needs none of the
   vocabulary that the hard GPU bugs need.

### 5.5 Would the abstraction hide it?

**Yes — completely, under an abstract FP encoding. This defect needs the exact
bit-to-bit FP axiom profile.**

Under an abstract encoding, where each floating-point value is an opaque
identifier and `addf` is an uninterpreted function constrained only by a few
axioms (commutativity, `neg` being its own inverse, a handful of distinct
reserved constants), lifting the interpreter side leaves two choices and both
are useless:

* **Lift it as `addf`**, because the frontend method was called `create_fadd`.
  Then both sides are the same uninterpreted `addf` and the checker proves them
  **equivalent**. A false pass — the tool reports the buggy implementation is
  correct.
* **Lift it as `bvadd` on the representation.** Then one side is an
  uninterpreted function and the other is a concrete bit-vector operation, and
  the solver cannot relate them at all. It answers SAT — but for the wrong
  reason, since an uninterpreted function is allowed to be anything. It would
  answer SAT for a *correct* implementation too, so the answer carries no
  information.

Only a profile in which `addf` really is IEEE `fp.add` distinguishes them.

Would it be confused with a benign reassociation? No, and the reason is worth
stating, because it is what makes this defect *easy* for a checker that models
FP concretely: the difference is not an ordering. `1.0 + 1.0` giving `1.7e38`
is not any grouping of the same sum. So the reassoc-allowed profile also
catches it, as long as each individual operation is still modelled as a
concrete IEEE operation. The single profile that hides it is the fully
uninterpreted one — which is the default, and which is the point worth taking
away.

### 5.6 Inherent cost

Tiny. The counterexample query is one 16-bit `bvadd` against one bf16 `fp.add`
over two free 16-bit variables — a 2^32 space that bit-blasts to a few hundred
clauses and solves in milliseconds. This is close to the cheapest non-trivial
SAT query one could write. The general rule (UNSAT cheap, SAT expensive, so
shrink the tile) barely applies: the tile does not need shrinking, because the
defect is per-element and one element is enough. At full width it is just N
independent copies of the same 32-bit problem.

### 5.7 The real ceiling, stated plainly

The cost is not the solver. The cost is the lifting, and there is a trap in it
that I think is the most useful thing this data point has to offer.

To run the check at all, someone must write the lifting from the interpreter's
numpy program into the core. That lifting is a second frontend, written by a
person. If that person writes it by reading `InterpreterBuilder` and encoding
*what each method means* rather than *what each method does*, they will write
`fp.add` for `create_fadd` — because the method is called `create_fadd` — and
the defect is lifted away before the solver ever sees it. The bug is invisible
to any lifter that trusts the method name.

So the general lesson: **a translation validator only catches a defect in a
component it models mechanically.** A hand-written model of a buggy component
inherits the modeller's assumption about what that component was supposed to do,
and that assumption is exactly what the defect violates. For this bug the lifter
would have to be derived from the numpy operations actually executed — a trace,
or a symbolic run of the real Python — not from a reading of the source.

And one closing observation for the SMT project, since this report exists to be
a ground-truth data point: the campaign's premise is `input IR → output IR`, and
this finding does not have that shape. It is a differential test between two
implementations of one source language. It suggests the tool's highest-value
target may not be pass-versus-pass at all. Triton has at least two full
implementations of its own language — the compiler and the interpreter — and
they are *supposed* to agree on every program. That is a much larger and much
less examined equivalence obligation than any single pass, and nobody is
checking it with anything stronger than a test suite that skips the hard cases.

---

## 6. Confidence

**Very high that the behaviour is real and that the mechanism is understood.
Moderate on how a Triton maintainer will rank it.**

What makes the behaviour certain:

* Reproduced from scratch, minimally, in both arms, and re-run to confirm.
* The wrong output bits are *derivable by hand* from the input bits as unsigned
  addition with wraparound (`0xC080 + 0xC040 = 0x180C0 → 0x80C0`), and they
  match what was measured, for bf16 and for both fp8 types. This is not a
  plausible story fitted to a symptom; the numbers only come out that way if
  line 568 is doing integer addition on the encodings.
* The ground truth is not in dispute: `-4 + -3 = -7`, and both the compiled arm
  and an independent torch fp32 computation say `-7`.
* Deterministic: three identical runs, and it reproduces with no GPU visible.

What I am deliberately *not* claiming:

* I have not shown that every operation on every one of the six affected types
  is wrong. I measured `+`, `*`, unary `-`, `<`, `tl.sum` and `tl.dot` for
  bf16, and `+`, `tl.maximum` and `<` for two of the five fp8 types. The rest I
  infer from the shared lines 568 and 678, which is code reading, not
  measurement. Three fp8 types (`float8e5b16`, `float8e4b8`, `float8e4b15`) were
  not measured at all, because torch has no matching dtype to build inputs from.
* I have not checked whether this is already an open issue upstream. GitHub is
  blocked from this box, so I could only search the checkout. Inside the
  checkout it is known in the docs and in the tests, as described above.

Where the honest doubt sits — and it is about the *filing*, not the finding:

For **bf16**, a maintainer can reasonably answer "working as documented, see the
Limitations section". I think that answer is wrong, but it is not unreasonable,
and the report should be filed in a way that meets it head on: the ask is not
"make bf16 work", it is "**raise instead of returning a number**", the way
`extern_elementwise` and `inline_asm` already do at lines 811 and 814. The
supporting argument is that the limitation as written is not checkable by the
user — several bf16 operations *do* work (casts, `tl.maximum`, `tl.abs`,
`tl.where`), which teaches the user that bf16 is fine, right up until they use
`+`.

For **fp8** the case is stronger and I would lead with it: it is in no document
and in no test skip, and the compiled path *rejects* the same program. A tool
that returns an answer where the compiler returns an error is hard to defend.

What would settle the remaining doubt: filing it upstream and seeing whether the
maintainers treat the missing guard as a bug or as intended. If someone wants to
close the measurement gap first, the cheap next step is a sweep over every
`create_*` entry point crossed with all six affected dtypes, comparing the
interpreter against a torch fp32 reference — that turns the inferred rows of the
table in §3 into measured ones. It would not change the verdict, only its width.
