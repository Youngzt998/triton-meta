"""Take the kernel `torch.compile` actually emits, and run it directly.

Why not just time the compiled callable? Because the five arms need five different Inductor
configs, and `torch._dynamo.reset()` between two compiles in one process invalidates the first
compiled callable -- so each arm has to compile in its own process, and then the arms are timed
minutes apart on a GPU whose clock ramps from 120 MHz to 2070 MHz. Measured: the same config came
out 0.0294 ms and 0.0497 ms in two runs that differed only in how long the process had been busy.

So: each arm compiles in a subprocess and writes out the Triton source Inductor generated, and one
timing process reloads all of them and times them round-robin against each other. What gets timed
is exactly the text in `output_code.py`, with the `@triton_heuristics` wrapper peeled off and the
launch made explicit.

That also makes arm 3 honest. "Ours" is arm 2's own source with the epilogue lines replaced --
same mainloop, same tile shape, same grid, same launch. bitequiv's plan for these shapes is mode
`plain`, whose bit contract is "BK real k-elements per tl.dot, in increasing k, into one fp32
accumulator, scaled and rounded once at the end", and Inductor's mm template mainloop already
satisfies it exactly. So the mainloop needs no replacing, and every microsecond of the difference
between arm 2 and arm 3 is the epilogue's rounding, not a template we wrote.
"""
from __future__ import annotations

import os
import re
import sys

import triton

TEM = re.compile(r"^(\w+) = async_compile\.triton\('\1', '''(.*?)''', device_str=", re.S | re.M)
CALL = re.compile(r"^\s+(\w+)\.run\((.*?)\)\s*$", re.M)
EXTERN = re.compile(r"^\s+extern_kernels\.(\w+)\((.*?)\)\s*$", re.M)
SIZE = re.compile(r"^\s+assert_size_stride\((\w+), \((.*?)\), \((.*?)\)\)", re.M)
MMOUT = re.compile(r"^\s+extern_kernels\.mm\(.*?out=(\w+)\)", re.M)


def extract(output_code: str):
    """name -> the raw Triton source Inductor generated for that kernel."""
    return {m.group(1): m.group(2) for m in TEM.finditer(output_code)}


def launch_meta(src: str):
    """num_warps / num_stages / enable_fp_fusion / the constexpr defaults in the kernel body.

    `enable_fp_fusion` matters as much as the tile does. Inductor sets it from its own config --
    `"enable_fp_fusion": not config.emulate_precision_casts` in `codegen/triton.py` -- because
    with fp contraction ON, LLVM narrows `fpext(fptrunc(fpext(a) * c))` back to fp16 and then
    folds the following add into one `fma.rn.f16`, which SKIPS the rounding the cast asked for.
    Launching an `emulate_precision_casts` kernel without the flag is therefore not launching the
    kernel torch.compile would have run.
    """
    out = {}
    for k in ("num_warps", "num_stages"):
        m = re.search(rf"^{k}=(\d+),", src, re.M)
        if m:
            out[k] = int(m.group(1))
    m = re.search(r"'enable_fp_fusion': (True|False)", src)
    if m:
        out["enable_fp_fusion"] = m.group(1) == "True"
    for m in re.finditer(r"^\s+(\w+) : tl\.constexpr = (.+)$", src, re.M):
        out.setdefault("constexpr", {})[m.group(1)] = m.group(2).strip()
    m = re.search(r"^def (triton_\w+)\((.*?)\):", src, re.M)
    out["fn_name"], out["args"] = m.group(1), [a.strip() for a in m.group(2).split(",")]
    return out


def grid_of(output_code: str, name: str):
    """The literal grid Inductor passes. Its template kernels take it as three trailing ints."""
    for m in CALL.finditer(output_code):
        if m.group(1) != name:
            continue
        args = [a.strip() for a in m.group(2).split(",") if "=" not in a]
        tail = [a for a in args if re.fullmatch(r"\d+", a)]
        if len(tail) >= 3:
            return tuple(int(v) for v in tail[-3:])
        if len(tail) == 1:  # a pointwise kernel: the one int is xnumel
            return ("xnumel", int(tail[0]))
    return None


def arg_shapes(output_code: str):
    """caller-side name -> shape tuple, read off the `assert_size_stride` lines.

    Inductor emits one per graph input, which is how a caller-side name (`arg3_1`) is turned back
    into "this is the [M,1] one".
    """
    out = {}
    for m in SIZE.finditer(output_code):
        dims = [d.strip() for d in m.group(2).split(",") if d.strip()]
        try:
            out[m.group(1)] = tuple(int(d) for d in dims)
        except ValueError:  # a dynamic dimension: not a shape we can match on
            pass
    return out


def run_args(output_code: str, name: str):
    """The caller-side argument names of `<name>.run(...)`, in order, keyword arguments dropped."""
    for m in CALL.finditer(output_code):
        if m.group(1) == name:
            return [a.strip() for a in _top_level_split(m.group(2)) if "=" not in a]
    return []


def bind(output_code: str, src: str, name: str, operands=()):
    """Say what the launcher must pass in each of the generated kernel's parameter slots.

    This cannot be done by position. Inductor orders the kernel's parameters by the order the
    graph needs them, not by the order the traced function takes them, and both of the obvious
    guesses are wrong on a real case:

      resid      `(in_ptr0, arg_A, arg_B, out_ptr0)`  -- the extra operand comes FIRST
      swiglu_cw  `(arg_A, arg_B, in_ptr2, in_ptr3, out_ptr1)` with `in_ptr2` the [M,1] routing
                 weight and `in_ptr3` the [M,N] gate, i.e. the reverse of the traced order

    So the binding is read off the `.run(...)` line, which names the caller-side buffer in every
    slot, and each `in_ptr` is then classified by the SHAPE of that buffer. A wrong guess here
    does not raise -- Triton takes a bare pointer -- it silently reads the wrong tensor, so this
    returns a list and the caller builds the call from it rather than assuming an order.

    Returns a list, one entry per kernel parameter: "A", "B", "out", ("extra", i) with i indexing
    `operands`, or ("const", literal) for the trailing grid/`xnumel` integers.
    """
    raw = launch_meta(src)["args"]
    params = [a.split(":")[0].strip() for a in raw]
    callers = run_args(output_code, name)
    shapes = arg_shapes(output_code)
    mm_out = MMOUT.search(output_code)
    mm_out = mm_out.group(1) if mm_out else None
    kinds = _load_kinds(src)  # in_ptr name -> "mn" / "m1" / "n", from how the body indexes it
    order, seen = [], {}
    for i, p in enumerate(params):
        caller = callers[i] if i < len(callers) else None
        if "tl.constexpr" in raw[i]:
            continue  # passed as a keyword, not positionally
        if p == "arg_A":
            order.append("A")
        elif p == "arg_B":
            order.append("B")
        elif p.startswith("out_ptr"):
            order.append("out")
        elif p.startswith("in_out_ptr"):
            order.append("mm")  # the pointwise arm overwrites the mm output in place
        elif p.startswith("in_ptr"):
            # Inductor names graph inputs `arg<N>_1` and intermediates `buf<N>`, and the mm output
            # can reach the pointwise kernel under an alias of the buffer `extern_kernels.mm` wrote
            # -- so "is it a buffer" is the test, not "is it THE buffer".
            if caller is not None and (caller == mm_out or caller.startswith("buf")):
                order.append("mm")
                continue
            kind = kinds.get(p)
            if caller in shapes:  # the shape is the authority; the index expression is a check
                s = shapes[caller]
                by_shape = "n" if len(s) == 1 else ("m1" if s[-1] == 1 else "mn")
                if kind is not None and kind != by_shape:
                    raise RuntimeError(f"{name}: {p} is indexed as {kind} but its caller {caller} is {s}")
                kind = by_shape
            if kind is None:
                raise RuntimeError(f"{name}: cannot tell what {p} is")
            j = seen.get(kind, 0)
            seen[kind] = j + 1
            want = [k for k, o in enumerate(operands) if o == kind]
            if j >= len(want):
                raise RuntimeError(f"{name}: {p} is a {j + 1}th {kind} operand, but the epilogue "
                                   f"declares {operands}")
            order.append(("extra", want[j]))
        elif re.fullmatch(r"\d+", str(caller or "")):
            order.append(("const", int(caller)))
        else:
            raise RuntimeError(f"{name}: no rule for parameter {p!r} (caller {caller!r})")
    for kind in set(operands):
        if seen.get(kind, 0) != operands.count(kind):
            raise RuntimeError(f"{name}: the epilogue declares {operands.count(kind)} {kind} operand(s) "
                               f"but the kernel reads {seen.get(kind, 0)}")
    return order


def call_args(order, A, B, out, extras, mm=None):
    """Turn a `bind` order into the positional argument tuple."""
    role = {"A": A, "B": B, "out": out, "mm": mm}
    got = []
    for o in order:
        if isinstance(o, tuple):
            got.append(extras[o[1]] if o[0] == "extra" else o[1])
        else:
            got.append(role[o])
    return tuple(got)


def strip_decorator(src: str) -> str:
    """Drop the `@triton_heuristics.<...>(...)` wrapper and keep the bare `@triton.jit`."""
    i = src.index("@triton_heuristics.")
    j = src.index("@triton.jit", i)
    return src[:i] + src[j:]


def load(src: str, tag: str, cache_dir: str):
    """`triton.jit` reads the decorated function back with `inspect.getsourcelines`, so the source
    has to live in a real file -- an `exec` of a string fails with "should be defined in a Python
    file". Write it out and import it; the file left behind is also the thing to read when
    checking that only the epilogue changed."""
    import hashlib
    import importlib.util

    src = strip_decorator(src)
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"gen_{tag}_{h}.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(f"gen_{tag}_{h}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, launch_meta(src)["fn_name"]), path


# --------------------------------------------------------------------------------------------
# the epilogue swap
# --------------------------------------------------------------------------------------------
# Every `.to(tl.float16).to(tl.float32)` below is one eager kernel boundary: eager materialises a
# fp16 tensor there, so the value is rounded to fp16 and read back. The LAST op of each chain is
# left unrounded because the store's own cast performs the same rounding -- for the fp16 outputs
# because `v.to(f16).to(f32).to(f16) == v.to(f16)`, and for the fp8 output because its last op is
# a clamp to +-448, exact in fp16, so the dropped rounding was already a no-op.
#
# The sigmoid spelling is the whole story for `silu`, so it is worth being precise about.
#
# torch's CUDA sigmoid for half is `1.f / (1.f + ::expf(-x))` in fp32: the accurate `__nv_expf`
# and the correctly rounded `div.rn.f32`. Transcribing that literally is `silu_divrn` below, and
# it costs about 0.0020 ms per 16.8M elements over an approximate divide -- which is most of what
# bit-exactness would cost.
#
# It is not needed. The claim to defend is that the OUTPUT BYTES match, not that the instructions
# match, and the input is an fp16 value, so the entire input domain is 63488 values and can be
# checked exhaustively rather than sampled. `probe_sigmoid.py` does that. Over all 63488, for
# sigmoid and for the composed silu alike:
#
#   tl.sigmoid, which is what Inductor emits           2 differ   (approximate divide)
#   1/(1 + libdevice.exp(-x)), Triton's own `/`        2 differ   (approximate divide)
#   div.rn / rcp.rn on the accurate exp                0 differ   but +0.0020 ms
#   guarded Newton on div.approx and ex2.approx        0 differ   and +0.0000 ms   <- used here
#
# The last one: `div.approx.f32` is MUFU.RCP plus a multiply, about 22 good bits; one Newton step
# `q + (1 - d*q)*q` takes that past what fp32 carries, and the fp16 store keeps only 11 bits. The
# `tl.where` is not cosmetic -- for x below about -88 the exp overflows to inf, so q is exactly 0
# and the correction computes inf*0 = NaN. Unguarded, that alone is 9844 of the 63488 wrong, which
# reads like an accuracy failure and is not one.
_ROUND = ".to(tl.float16).to(tl.float32)"

# Every `EPI_SRC` chain below is fp16-only: `_ROUND` names the dtype. A bf16 case has different
# rounding points and there is no arm 3 for it, so `three_way.py` refuses rather than measuring
# the wrong thing.
#
# A round is only a round if the compiler keeps it. `_ROUND` alone does NOT survive: LLVM narrows
# `fpext(fptrunc(fpext(a) * c))` back to fp16 and then contracts the following add into a single
# `fma.rn.f16`, which computes the product to full width and never rounds it. Measured on
# `lora`: 8,211,796 of the 4,294,967,296 (accumulator, base) pairs came out wrong at
# scale=0.25, e.g. acc = base = 2^-24 gives 2^-23 where eager gives 2^-24, because the fused
# form keeps 2^-25 instead of rounding it to zero. The PTX is one `fma.rn.f16` for the whole
# chain.
#
# The fix is the flag PyTorch itself uses for exactly this: `enable_fp_fusion=False`, which
# Inductor sets whenever `emulate_precision_casts` is on. It is a launch option, not text, so
# `three_way.py` passes `NO_FP_FUSION` for arm 3 and checks that the mainloop's PTX is unchanged
# by it -- otherwise arm 3 would differ from arm 2 in more than the epilogue.
NO_FP_FUSION = False

_SIGMOID = [
    "tmp_d = 1.0 + libdevice.fast_expf(-{x})",
    "tmp_q = libdevice.fast_dividef(1.0, tmp_d)",
    "tmp_s = tl.where(tmp_q > 0.0, tl.fma(tl.fma(-tmp_d, tmp_q, 1.0), tmp_q, tmp_q), tmp_q)" + _ROUND,
]


# `F.silu` is NOT `x * sigmoid(x)`. torch's CUDA silu for a half dtype is
#
#     opmath_t x_acc = x;  return x_acc / (1 + ::exp(-x_acc));
#
# in fp32 -- the accurate `__nv_expf` and one correctly rounded `div.rn.f32`, not a reciprocal
# followed by a multiply. Inductor's own generated text agrees: it emits `tmp1 / (exp(-tmp1) + 1)`,
# but with Triton's `/`, which is not correctly rounded (`ours.py` measured 2 of 63488 fp16
# outputs differing for the sigmoid twin of this).
#
# Two spellings are carried. `_silu_rn` is the literal transcription: over all 4,294,967,296
# (accumulator, gate) pairs of `swiglu` it gives torch's bytes on every one.
#
# `_silu_fast` replaces the divide with one Newton step on `div.approx` (about 22 good bits, taken
# past what fp32 carries) and the accurate exp with `ex2.approx`. It is the cheap one and it is
# NOT exact: 162,804 of the same 4,294,967,296 pairs differ. The first is the plainest case there
# is, `silu(-0.0)`: the correction `fma(fma(-d, q, x), r, q)` computes `+0 + -0` and turns -0 into
# +0, where torch's `-0.0 / 2.0` keeps the sign. So it stays a priced alternative -- how much of
# the exact spelling's cost is the correctly rounded divide -- and never an exact one.
#
# The `tl.where` in `_silu_fast` is not cosmetic either. The Newton correction is `fma(-d, q, x)`,
# and for x below about -87 the exp overflows so d is inf: `-inf * 0 + x` is NaN. The guard falls
# back to the uncorrected quotient, which is the right answer there (x/inf is a signed zero).
def _silu_rn(x, tag):
    return [
        f"{tag}_d = 1.0 + libdevice.exp(-{x})",
        f"{tag}_s = libdevice.div_rn({x}, {tag}_d)" + _ROUND,
    ]


def _silu_fast(x, tag):
    return [
        f"{tag}_d = 1.0 + libdevice.fast_expf(-{x})",
        f"{tag}_r = libdevice.fast_dividef(1.0, {tag}_d)",
        f"{tag}_q = {x} * {tag}_r",
        f"{tag}_n = tl.fma(tl.fma(-{tag}_d, {tag}_q, {x}), {tag}_r, {tag}_q)",
        f"{tag}_s = tl.where({tag}_n == {tag}_n, {tag}_n, {tag}_q)" + _ROUND,
    ]


# What Inductor itself emits for silu: fp32 throughout, Triton's own divide, no intermediate
# rounding. Not bit-exact; carried to price the exact spellings against.
def _silu_approx(x, tag):
    return [f"{tag}_s = {x} / (1.0 + libdevice.exp(-{x}))"]


EPI_SRC = {
    "silu": ["tmp_x = acc" + _ROUND] + [s.format(x="tmp_x") for s in _SIGMOID] + ["tmp_out = tmp_x * tmp_s"],
    "silu_divrn": [  # the literal transcription of torch's C++, for pricing the one above
        "tmp_x = acc" + _ROUND,
        "tmp_s = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-tmp_x))" + _ROUND,
        "tmp_out = tmp_x * tmp_s",
    ],
    "silu_approx": [  # NOT bit-exact: the spelling Inductor uses, for pricing the exact one
        "tmp_x = acc" + _ROUND,
        "tmp_s = tl.sigmoid(tmp_x)" + _ROUND,
        "tmp_out = tmp_x * tmp_s",
    ],
    "none": ["tmp_out = acc"],
    # `torch.clamp` propagates NaN, so the clamp has to be the compare-and-select pair that
    # `triton_helpers` emits. `tl.minimum`/`tl.maximum` lower to `min.f32`/`max.f32`, which return
    # the NON-NaN operand, so they turn a NaN accumulator into -448 -- exact on every finite input
    # and wrong on all 2046 fp16 NaN patterns. `probe_epi_src.py` is what caught that.
    "fp8cast": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = (tmp_x * 0.375)" + _ROUND,
        "tmp_out = triton_helpers.minimum(triton_helpers.maximum(tmp_t, -448.0), 448.0)",
    ],
    # the same chain with the cheaper NaN-dropping clamp. Carried so the report can say how much of
    # any fp8cast win is the clamp spelling rather than anything to do with bit-exactness.
    "fp8cast_cheap": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = (tmp_x * 0.375)" + _ROUND,
        "tmp_out = tl.minimum(tl.maximum(tmp_t, -448.0), 448.0)",
    ],
    # Inductor's arithmetic exactly: fp32 throughout, no rounding back to fp16. NOT bit-exact.
    "fp8cast_approx": [
        "tmp_out = triton_helpers.minimum(triton_helpers.maximum(acc * 0.375, -448.0), 448.0)",
    ],
    # ---- the rest of the x-only epilogues, transcribed from `ours.py::_epilogue` --------------
    # Each is the same chain torch eager runs, with one `.to(fp16).to(fp32)` per eager kernel
    # boundary and none after the last op (the store's own cast does that rounding). The `_approx`
    # twin of each is Inductor's own arithmetic -- fp32 all the way through, no intermediate
    # rounding -- and is carried only to price the exact one against it.
    "softcap": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = (tmp_x * 0.0625)" + _ROUND,
        "tmp_h = libdevice.tanh(tmp_t)" + _ROUND,
        "tmp_out = tmp_h * 16.0",
    ],
    "softcap_approx": ["tmp_out = libdevice.tanh(acc * 0.0625) * 16.0"],
    "sigmoid": ["tmp_x = acc" + _ROUND] + [s.format(x="tmp_x") for s in _SIGMOID] + ["tmp_out = tmp_s"],
    "sigmoid_approx": ["tmp_out = tl.sigmoid(acc)"],
    # torch's relu is `threshold(x, 0, 0)`, i.e. `x <= 0 ? 0 : x`, so a NaN input falls through the
    # false branch and comes back as NaN. Spelling it `x > 0 ? x : 0` returns 0 instead and is
    # wrong on all 2046 fp16 NaN patterns; `tl.maximum` is wrong the same way.
    "relu_sq": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = tl.where(tmp_x <= 0.0, 0.0, tmp_x)" + _ROUND,
        "tmp_out = tmp_t * tmp_t",
    ],
    "relu_sq_approx": [
        "tmp_t = triton_helpers.maximum(acc, 0.0)",
        "tmp_out = tmp_t * tmp_t",
    ],
    "leaky": [
        "tmp_x = acc" + _ROUND,
        "tmp_out = tl.where(tmp_x > 0.0, tmp_x, tmp_x * 0.01)",
    ],
    "leaky_approx": ["tmp_out = tl.where(acc > 0.0, acc, acc * 0.01)"],
    "hardswish": [
        "tmp_x = acc" + _ROUND,
        "tmp_out = tmp_x * tl.minimum(tl.maximum(tmp_x + 3.0, 0.0), 6.0) / 6.0",
    ],
    "hardswish_approx": [
        "tmp_out = acc * triton_helpers.minimum(triton_helpers.maximum(acc + 3.0, 0.0), 6.0) / 6.0",
    ],
    # There is deliberately no `gelu` entry, and the reason is a result rather than an omission.
    # torch's gelu is exactly `(x * 0.5) * (1 + erf(x * M_SQRT1_2))` in fp32 -- checked over the
    # whole fp16 domain, 0 of 65536 differ from `F.gelu`. But Triton's `libdevice.erf` is not the
    # same function as the `erff` torch calls: they differ by one fp32 ulp on 838 of 65536 inputs,
    # and `1 + erf(z)` cancels to nothing as z -> -1, so in the left tail (x in [-5.54, -2.82],
    # where gelu is 1e-3 or smaller) that one ulp becomes 151 of 65536 differing fp16 outputs.
    # Eight spellings were tried -- reassociations, fp64, `normcdf`, `erfc` -- and none closed it,
    # because the gap is inside the polynomial, not in how it is composed. So gelu, the one
    # non-trivial epilogue cuBLASLt can already fuse, is also the one where no arm 3 exists.

    # ---- the epilogues that follow a real model's GEMM ----------------------------------------
    # `tmp_e0` and `tmp_e1` are the extra operands `patch_epilogue` keeps and rewires, in the
    # order `cases.EPI_OPERANDS` declares them. They come out of memory already at the output
    # dtype, so they need no rounding -- the unfused path reads the same bytes from the same
    # tensor. `{scale}` and `{lim}` are filled from `cases.EPI_PARAMS`.
    #
    # Each chain is one line per eager kernel, with a `.to(fp16).to(fp32)` at every boundary and
    # none after the last op, because the store's own cast does that rounding.
    "swiglu": (["tmp_x = acc" + _ROUND] + _silu_rn("tmp_e0", "tmp_g") + ["tmp_out = tmp_g_s * tmp_x"]),
    "swiglu_fast": (["tmp_x = acc" + _ROUND] + _silu_fast("tmp_e0", "tmp_g") + ["tmp_out = tmp_g_s * tmp_x"]),
    "swiglu_approx":
    _silu_approx("tmp_e0", "tmp_g") + ["tmp_out = tmp_g_s * acc"],

    # One multiply by a per-token scalar. Both operands are fp16 values, so their fp32 product is
    # EXACT (11 + 11 significand bits fit in 24, and no fp16 pair over- or underflows fp32) and
    # both paths round it once, on the store.
    "rweight": ["tmp_x = acc" + _ROUND, "tmp_out = tmp_x * tmp_e0"],
    "rweight_approx": ["tmp_out = acc * tmp_e0"],

    # One add. Not exact in fp32 -- two fp16 values 40 binades apart need 50 bits -- but eager
    # rounds in exactly the same two places: once by `add.f32`, once by the store's cast.
    "resid": ["tmp_x = acc" + _ROUND, "tmp_out = tmp_x + tmp_e0"],
    "resid_approx": ["tmp_out = acc + tmp_e0"],

    # peft applies the scale and the add as two kernels, so the scaled delta is rounded to fp16
    # before it is added. Dropping that round is not just a lost bit: it lets the backend contract
    # `base + acc * s` into one FMA, which is a different function.
    "lora": ["tmp_x = acc" + _ROUND, "tmp_t = (tmp_x * {scale})" + _ROUND, "tmp_out = tmp_e0 + tmp_t"],
    "lora_approx": ["tmp_t = acc * {scale}", "tmp_out = tmp_e0 + tmp_t"],

    # DeepSeek-V4's clamped SwiGLU with the routing weight folded in. `triton_helpers.minimum` and
    # `maximum` are the NaN-keeping compare-and-select pair, which is what `torch.clamp` is;
    # `tl.minimum`/`tl.maximum` are not, and are wrong on all 2046 fp16 NaN patterns.
    "swiglu_cw": ([
        "tmp_x = acc" + _ROUND,
        "tmp_xc = triton_helpers.minimum(triton_helpers.maximum(tmp_x, -{lim}), {lim})" + _ROUND,
        "tmp_gc = triton_helpers.minimum(tmp_e0, {lim})" + _ROUND,
    ] + _silu_rn("tmp_gc", "tmp_g") + ["tmp_t = (tmp_xc * tmp_g_s)" + _ROUND, "tmp_out = tmp_e1 * tmp_t"]),
    "swiglu_cw_fast": ([
        "tmp_x = acc" + _ROUND,
        "tmp_xc = triton_helpers.minimum(triton_helpers.maximum(tmp_x, -{lim}), {lim})" + _ROUND,
        "tmp_gc = triton_helpers.minimum(tmp_e0, {lim})" + _ROUND,
    ] + _silu_fast("tmp_gc", "tmp_g") + ["tmp_t = (tmp_xc * tmp_g_s)" + _ROUND, "tmp_out = tmp_e1 * tmp_t"]),
    "swiglu_cw_approx": ([
        "tmp_xc = triton_helpers.minimum(triton_helpers.maximum(acc, -{lim}), {lim})",
        "tmp_gc = triton_helpers.minimum(tmp_e0, {lim})",
    ] + _silu_approx("tmp_gc", "tmp_g") + ["tmp_out = tmp_e1 * (tmp_xc * tmp_g_s)"]),

    # Nemotron-3.5's `mlp_hidden_act`, the same chain as `relu_sq` under the model table's name.
    "relu2": ["tmp_x = acc" + _ROUND, "tmp_t = tl.where(tmp_x <= 0.0, 0.0, tmp_x)" + _ROUND, "tmp_out = tmp_t * tmp_t"],
    "relu2_approx": ["tmp_t = triton_helpers.maximum(acc, 0.0)", "tmp_out = tmp_t * tmp_t"],

    # Scale, then cast to fp8 -- with NO clamp, which is what `fusion_moe/fused_moe.py` measures
    # and is why this is not the same function as `fp8cast` above. eager rounds `x * s` to fp16
    # before the fp8 cast, so the round below is load-bearing: fp32 -> fp8 and fp32 -> fp16 -> fp8
    # do not agree wherever the fp16 step lands on an fp8 tie.
    "fp8q": ["tmp_x = acc" + _ROUND, "tmp_out = (tmp_x * {scale})" + _ROUND],
    "fp8q_approx": ["tmp_out = acc * {scale}"],
}

LOAD = re.compile(r"^(\s+)(tmp\w*) = (tl\.load\((\w+)\s*\+.*)$")


def _suffix_span(lines):
    """The generated suffix: everything after the last `mask = ...` up to the store."""
    store = max(i for i, ln in enumerate(lines) if "tl.store(" in ln)
    body_start = max(i for i, ln in enumerate(lines[:store]) if re.match(r"\s+mask = ", ln))
    return body_start, store


def _index_kind(expr: str):
    """What an epilogue load's index expression says the operand is.

    `xindex` is `idx_n + N*idx_m`, so a full [M,N] operand's index mentions `idx_n` or `xindex`
    and a per-token [M,1] operand's mentions `idx_m` alone.
    """
    if "idx_n" in expr or "xindex" in expr:
        return "mn"
    if "idx_m" in expr:
        return "m1"
    return "n"


def _load_kinds(src: str):
    """pointer argument name -> operand kind, for every load in the generated suffix.

    Empty for a pointwise kernel: that one has no template suffix and indexes everything off
    `xindex`. `bind` only uses this as a cross-check on the caller's shape, which is the
    authority, so an empty answer costs nothing there.
    """
    lines = src.split("\n")
    try:
        body_start, store = _suffix_span(lines)
    except ValueError:
        return {}
    out = {}
    for ln in lines[body_start + 1:store]:
        m = LOAD.match(ln)
        if m:
            out[m.group(4)] = _index_kind(m.group(3))
    return out


def patch_epilogue(src: str, epi: str, operands=None, params=None, drop_extra_loads=False) -> str:
    """Replace the arithmetic of Inductor's generated suffix, and nothing else.

    Kept: the whole mainloop, the tile shape, the grouped program ordering, the index and mask
    lines, THE LOADS OF THE EXTRA OPERANDS, and the store itself -- same tile, same grid, same
    launch. Replaced: only the `tmp<n> = ...` lines that compute the value.

    Keeping the loads is what lets an epilogue read anything but the accumulator. Each kept load
    is renamed to `tmp_e<i>`, with `<i>` indexing the epilogue's declared operand list, and it is
    the KIND of the load's index expression that decides which slot it fills -- never the order
    the loads appear in, which is Inductor's and not ours. A load that no declared operand claims
    raises rather than being dropped, because a dropped load is a kernel that quietly computes
    something else. `drop_extra_loads` is the one deliberate exception: the mainloop-only floor
    (`epi="none"`) wants the GEMM WITHOUT the epilogue's extra traffic, so there the loads go.
    """
    if operands is None:
        operands = _operands_of(epi)
    lines = src.split("\n")
    body_start, store = _suffix_span(lines)
    indent = re.match(r"(\s*)", lines[store]).group(1)

    seen, keep = {}, []
    for ln in lines[body_start + 1:store]:
        m = LOAD.match(ln)
        if m and drop_extra_loads:
            continue
        if m:
            kind = _index_kind(m.group(3))
            j = seen.get(kind, 0)
            seen[kind] = j + 1
            want = [k for k, o in enumerate(operands) if o == kind]
            if j >= len(want):
                raise RuntimeError(f"the suffix loads a {j + 1}th {kind} operand but the epilogue "
                                   f"declares {operands}")
            i = want[j]
            keep.append(f"{m.group(1)}tmp_e{i}_raw = {m.group(3)}")
            # normalise to fp32 whatever the load returned, so the epilogue text does not depend
            # on whether Inductor already appended its own `.to(tl.float32)`
            keep.append(f"{m.group(1)}tmp_e{i} = tmp_e{i}_raw.to(tl.float32)")
        elif not re.match(r"\s+tmp\w* = ", ln):
            keep.append(ln)
    for kind in (set(operands) if not drop_extra_loads else ()):
        if seen.get(kind, 0) != operands.count(kind):
            raise RuntimeError(f"the epilogue declares {operands.count(kind)} {kind} operand(s) but the "
                               f"suffix loads {seen.get(kind, 0)}")

    body = EPI_SRC[epi]
    if params:
        body = [ln.format(**params) if "{" in ln else ln for ln in body]
    new = keep + [indent + s for s in body]
    # the store keeps its address and its mask; only the value it stores is renamed
    call = lines[store].strip()
    inner = call[call.index("(") + 1:call.rindex(")")]
    parts = _top_level_split(inner)
    if len(parts) < 2:
        raise RuntimeError("could not find the stored value in: " + call)
    parts[1] = " tmp_out"
    st = indent + "tl.store(" + ",".join(parts) + ")"
    return "\n".join(lines[:body_start + 1] + new + [st] + lines[store + 1:])


def _operands_of(key: str):
    from cases import EPI_OPERANDS, base_key
    return EPI_OPERANDS.get(base_key(key), ())


def _top_level_split(s: str):
    """Split on commas that are not inside brackets. `tl.store(p + (f(x, [A, B])), v, m)` has
    three arguments, and a plain `split(',')` finds five."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


# --------------------------------------------------------------------------------------------
# the same epilogue as ONE separate kernel: arm 1's second half, and what the probes verify
# --------------------------------------------------------------------------------------------
# Built from the same `EPI_SRC` text arm 3 is patched with, so the two arms cannot drift in the
# arithmetic -- only in where the value came from. That also means the text `probe_pairs.py`
# sweeps over its complete domain is the text BOTH arms run.
#
# Two shapes, following `fusion_moe/fused_moe.py`: flat when every operand is [M,N], and one
# program per (row, column tile) when one is a per-token [M,1] column. The flat form would have
# to divide by N to find the row, and charging the baseline a division it does not need is how a
# baseline quietly loses.
_STANDALONE_FLAT = '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers  # noqa: F401
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math  # noqa: F401


@triton.jit
def epi(C, {ptrs}O, NEL, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mask = off < NEL
    acc = tl.load(C + off, mask=mask, other=0.0).to(tl.float32)
{loads}{body}
    tl.store(O + off, tmp_out, mask)
'''

_STANDALONE_ROW = '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers  # noqa: F401
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math  # noqa: F401


@triton.jit
def epi(C, {ptrs}O, M, N, sc, so, {strides}BLK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLK + tl.arange(0, BLK)
    mask = n < N
    acc = tl.load(C + m * sc + n, mask=mask, other=0.0).to(tl.float32)
{loads}{body}
    tl.store(O + m * so + n, tmp_out, mask)
'''


def standalone(key, operands=None, params=None, cache_dir=".", force_flat=False, body=None):
    """(callable, kind) for the epilogue on its own. `kind` is "flat" or "row".

    `force_flat` is for the probes, which feed one long vector of operand values rather than a
    matrix and a column: a broadcast changes which element meets which, never what the
    arithmetic does to a pair.

    `body` overrides `EPI_SRC[key]`. It exists for one reason: every chain in `EPI_SRC` rounds
    through fp16 by name, and a bf16 case has to round through bf16 in BOTH the fused kernel and
    this one. If the two round differently this arm stops being a reference -- the fused arm then
    comes out byte-different on every draw and every configuration, which reads like a broken
    kernel and is a broken baseline. `bitequiv.cublas_match.fused_plain.round_lines` is the one
    place that substitution is written; pass its result here.
    """
    if operands is None:
        operands = _operands_of(key)
    body = EPI_SRC[key] if body is None else body
    if params:
        body = [ln.format(**params) if "{" in ln else ln for ln in body]
    row = "m1" in operands and not force_flat
    ptrs = "".join(f"E{i}, " for i in range(len(operands)))
    strides = "".join(f"s{i}m, s{i}n, " for i in range(len(operands))) if row else ""
    loads = ""
    for i in range(len(operands)):
        addr = f"E{i} + m * s{i}m + n * s{i}n" if row else f"E{i} + off"
        loads += (f"    tmp_e{i}_raw = tl.load({addr}, mask=mask, other=0.0)\n"
                  f"    tmp_e{i} = tmp_e{i}_raw.to(tl.float32)\n")
    tmpl = _STANDALONE_ROW if row else _STANDALONE_FLAT
    src = tmpl.format(ptrs=ptrs, strides=strides, loads=loads, body="\n".join("    " + ln for ln in body))
    return _load_named(src, f"epi_{key}", cache_dir, "epi"), ("row" if row else "flat")


def _load_named(src, tag, cache_dir, fn_name):
    import hashlib
    import importlib.util
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_{hashlib.sha1(src.encode()).hexdigest()[:10]}.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def launch_standalone(fn, kind, c, extras, out, BLK=2048, num_warps=4):
    """`c` is the GEMM's output as it sits in memory, which is the point: the leading round in
    every chain is a no-op here, and that is what makes the unfused arm the reference."""
    M, N = c.shape
    if kind == "row":
        strides = []
        for e in extras:  # a broadcast axis gets stride 0, so one kernel serves [M,N], [M,1], [N]
            if e.dim() == 1:
                strides += [0, e.stride(0)]
            else:
                strides += [e.stride(0) if e.shape[0] != 1 else 0, e.stride(1) if e.shape[1] != 1 else 0]
        fn[(M, cdiv(N, BLK))](c, *extras, out, M, N, c.stride(0), out.stride(0), *strides, BLK=BLK, num_warps=num_warps,
                              enable_fp_fusion=NO_FP_FUSION)
        return out
    flat = [e.reshape(-1) for e in extras]
    fn[(cdiv(c.numel(), BLK), )](c.reshape(-1), *flat, out.reshape(-1), c.numel(), BLK=BLK, num_warps=num_warps,
                                 enable_fp_fusion=NO_FP_FUSION)
    return out


def retune(src: str, M: int, N: int, K: int, BM: int, BN: int, BK: int, GROUP_M: int, num_warps: int, num_stages: int):
    """Same template, different tile. Returns (source, grid).

    Inductor's autotuner chose its tile while looking at ITS epilogue. A correctly rounded divide
    needs several more live registers per element of the accumulator, and at 128x128 with 4 warps
    the accumulator alone is 128 registers a thread, so the exact epilogue spills and costs 0.0286
    ms where the approximate one costs 0.0082. Re-tuning is not a favour to arm 3; it is what
    Inductor's own search would have done had it been given the exact epilogue. Arm 2 is swept
    over the same grid so neither side gets the larger search.

    Only the constexpr block is rewritten. `EVEN_K` has to follow BLOCK_K or the template drops
    the k mask on a ragged tail; the grid has to follow the tile because the template derives
    `grid_m`/`grid_n` from M, N and the blocks and expects the launcher to match.
    """
    for k, v in (("BLOCK_M", BM), ("BLOCK_N", BN), ("BLOCK_K", BK), ("GROUP_M", GROUP_M)):
        src, n = re.subn(rf"^(\s+{k} : tl\.constexpr = )\d+$", rf"\g<1>{v}", src, flags=re.M)
        if n != 1:
            raise RuntimeError(f"{k} not found once in the template ({n} times)")
    src = re.sub(r"^(\s+EVEN_K : tl\.constexpr = )\w+$", rf"\g<1>{K % BK == 0}", src, flags=re.M)
    src = re.sub(r"^num_warps=\d+,$", f"num_warps={num_warps},", src, flags=re.M)
    src = re.sub(r"^num_stages=\d+,$", f"num_stages={num_stages},", src, flags=re.M)
    return src, (triton.cdiv(M, BM) * triton.cdiv(N, BN), 1, 1)


def tile_space(M, N, K, src):
    """Tiles worth trying for a shape whose K is tiny: the kernel is a store with some arithmetic
    attached, so what matters is registers per thread and how many stores are in flight.

    BLOCK_K is constrained by what Inductor already emitted. `EVEN_K` is resolved at CODEGEN time,
    not compile time: when it is true the generated text has no k mask at all, so raising BLOCK_K
    above K makes the load run off the end of A and B. (Caught the hard way -- half the fp8cast
    sweep came back byte-different, and the half was exactly the BLOCK_K=32 configs reading past a
    K=16 operand. It read as an interesting result until the addresses were checked.) So: if the
    emitted loads are masked, any BLOCK_K is safe; if they are not, only BLOCK_K that divides K.
    """
    masked = "mask=a_mask" in src or "mask=b_mask" in src
    bks = [b for b in (16, 32, 64, 128) if (b <= max(16, 2 * K) if masked else (K % b == 0))]
    if not bks:
        bks = [16]
    out = []
    for bm, bn in ((128, 128), (128, 64), (64, 128), (64, 64), (128, 256), (256, 128), (64, 256), (256, 64), (32, 128),
                   (128, 32), (32, 256), (256, 32)):
        if bm > M or bn > N:
            continue
        for bk in bks:
            for nw in (4, 8):
                for ns in (2, 3, 4):
                    out.append((bm, bn, bk, 8, nw, ns))
    return out


def launch(fn, grid, args, meta, enable_fp_fusion=None):
    """Returns the compiled kernel, so a caller can read its PTX without compiling it twice."""
    if enable_fp_fusion is None:
        enable_fp_fusion = meta.get("enable_fp_fusion", True)
    return fn[grid](*args, num_warps=meta.get("num_warps", 4), num_stages=meta.get("num_stages", 3),
                    enable_fp_fusion=enable_fp_fusion)


def cdiv(a, b):
    return triton.cdiv(a, b)
