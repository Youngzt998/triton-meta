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


def extract(output_code: str):
    """name -> the raw Triton source Inductor generated for that kernel."""
    return {m.group(1): m.group(2) for m in TEM.finditer(output_code)}


def launch_meta(src: str):
    """num_warps / num_stages / the constexpr defaults written into the kernel body."""
    out = {}
    for k in ("num_warps", "num_stages"):
        m = re.search(rf"^{k}=(\d+),", src, re.M)
        if m:
            out[k] = int(m.group(1))
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

_SIGMOID = [
    "tmp_d = 1.0 + libdevice.fast_expf(-{x})",
    "tmp_q = libdevice.fast_dividef(1.0, tmp_d)",
    "tmp_s = tl.where(tmp_q > 0.0, tl.fma(tl.fma(-tmp_d, tmp_q, 1.0), tmp_q, tmp_q), tmp_q)" + _ROUND,
]

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
    "fp8cast": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = (tmp_x * 0.375)" + _ROUND,
        "tmp_out = tl.minimum(tl.maximum(tmp_t, -448.0), 448.0)",
    ],
    # the same chain with Inductor's own clamp, which propagates NaN through an extra compare and
    # select per bound. Without this control the fp8cast speedup could be the clamp spelling
    # rather than anything to do with bit-exactness.
    "fp8cast_helpers": [
        "tmp_x = acc" + _ROUND,
        "tmp_t = (tmp_x * 0.375)" + _ROUND,
        "tmp_out = triton_helpers.minimum(triton_helpers.maximum(tmp_t, -448.0), 448.0)",
    ],
    # Inductor's arithmetic exactly: fp32 throughout, no rounding back to fp16. NOT bit-exact.
    "fp8cast_approx": [
        "tmp_out = triton_helpers.minimum(triton_helpers.maximum(acc * 0.375, -448.0), 448.0)",
    ],
}


def patch_epilogue(src: str, epi: str) -> str:
    """Replace the arithmetic of Inductor's generated suffix, and nothing else.

    Kept: the whole mainloop, the tile shape, the grouped program ordering, the index and mask
    lines, and the store itself. Replaced: only the `tmp<n> = ...` lines that compute the value.
    """
    lines = src.split("\n")
    store = max(i for i, ln in enumerate(lines) if "tl.store(" in ln)
    body_start = max(i for i, ln in enumerate(lines[:store]) if re.match(r"\s+mask = ", ln))
    indent = re.match(r"(\s*)", lines[store]).group(1)
    keep = [ln for ln in lines[body_start + 1:store] if not re.match(r"\s+tmp\w* = ", ln)]
    new = keep + [indent + s for s in EPI_SRC[epi]]
    # the store keeps its address and its mask; only the value it stores is renamed
    call = lines[store].strip()
    inner = call[call.index("(") + 1:call.rindex(")")]
    parts = _top_level_split(inner)
    if len(parts) < 2:
        raise RuntimeError("could not find the stored value in: " + call)
    parts[1] = " tmp_out"
    st = indent + "tl.store(" + ",".join(parts) + ")"
    return "\n".join(lines[:body_start + 1] + new + [st] + lines[store + 1:])


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


def launch(fn, grid, args, meta):
    fn[grid](*args, num_warps=meta.get("num_warps", 4), num_stages=meta.get("num_stages", 3))


def cdiv(a, b):
    return triton.cdiv(a, b)
