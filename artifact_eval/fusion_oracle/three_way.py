"""The three-way comparison: unfused eager, Inductor fused, and Inductor fused made bit-exact.

    1  exact2k     cuBLAS, then the epilogue as one separate Triton kernel with eager's rounding.
                   This is the bit reference and the shape of what a user gets today.
    2  inductor    the kernel `torch.compile` emits for `GEMM + epilogue` under
                   `mode="max-autotune-no-cudagraphs"`, taken verbatim out of `output_code.py`.
                   Fast, chosen by Inductor's own search, and NOT byte-identical to 1.
    3  ours        arm 2's own source with only the epilogue arithmetic replaced by the rounding
                   eager does. Same mainloop, same tile, same grid, same launch.

  2/3 is the headline: what byte-exactness costs against the kernel PyTorch itself produces.

Two more arms are carried because they answer the obvious objections:

    1i eager2k     Inductor's own ATEN two-kernel path (`extern_kernels.mm` + one pointwise),
                   compiled with `emulate_precision_casts=1`. It is what `torch.compile` gives a
                   user who does not get a fused template -- and it is NOT byte-identical to eager
                   either, which is worth knowing on its own.
    2e inductor_e  arm 2 with `emulate_precision_casts=1`, i.e. Inductor's own attempt at eager's
                   rounding points. Separates the cost of the extra rounding from the cost of the
                   correctly rounded transcendentals.

Every arm is timed in ONE process, round-robin, after the clock is warmed -- see measure.py for
why that matters. Compiling has to happen in five separate processes (dynamo reset invalidates
earlier compiled callables), so each of those writes its `output_code.py` to `cache/` and this
process reloads the text.

An epilogue may read more than the accumulator -- the base GEMM's output for a LoRA merge, the
gate projection for a SwiGLU, a per-token routing weight for an MoE down projection. Those loads
are kept and rewired by `inductor_kernel.patch_epilogue`, and which tensor goes in which argument
slot is read off Inductor's own `.run(...)` line rather than guessed, because Inductor's
parameter order is neither the traced order nor a stable one (for `resid` the extra operand comes
before `arg_A`; for `swiglu_cw` the two extras arrive reversed).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> .venv/bin/python three_way.py --cases 4096,4096,16,silu
    ... --cases 16384,2048,32,lora,scale=0.5      # an epilogue scalar, baked into the kernel
    ... --cases-file cases_moe.txt                # written by model_cases.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(AE)
for p in (ROOT, AE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "three_way.jsonl")
ATTEMPTS = os.path.join(HERE, "attempts.jsonl")
os.makedirs(CACHE, exist_ok=True)

# (name, max_autotune_gemm_backends, emulate_precision_casts)
COMPILES = (("aten_e", "ATEN", True), ("triton", "TRITON", False), ("triton_e", "TRITON", True))


def log(path, rec):
    """Append one record, retrying a transient write error.

    Seen once mid-sweep: `OSError: [Errno 5] Input/output error` out of `flush()` on a disk with
    19 TB free. It killed the run and cost the case in flight. The record is the only thing a long
    sweep produces, so a blip on the way to disk must not end it.
    """
    line = json.dumps(rec) + "\n"
    for attempt in range(6):
        try:
            with open(path, "a") as f:
                f.write(line)
                f.flush()
            return
        except OSError as e:
            print(f"  write to {os.path.basename(path)} failed ({e}); retry {attempt + 1}/6", flush=True)
            time.sleep(2 * (attempt + 1))
    print(f"  GAVE UP writing to {path}", flush=True)


# --------------------------------------------------------------------------------------------
# phase A: compile in a subprocess, save the generated source
# --------------------------------------------------------------------------------------------


def do_extract(M, N, K, epi, dtype_name, which, dst, params):
    import torch
    import torch._inductor.config as ic
    from torch._inductor.utils import run_and_get_code

    from cases import EPILOGUES, make_epi_args

    _n, backends, emul = next(c for c in COMPILES if c[0] == which)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    seed = M * 1000003 + N * 10007 + K
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(M, K, generator=g, device="cuda") / 8).to(dtype)
    w = (torch.randn(K, N, generator=g, device="cuda") / 8).to(dtype)
    extra = make_epi_args(epi, M, N, dtype, seed=seed)

    with ic.patch(max_autotune_gemm_backends=backends, emulate_precision_casts=emul):
        fn = torch.compile(lambda x, y, *rest: EPILOGUES[epi](x @ y, *rest, **params),
                           mode="max-autotune-no-cudagraphs")
        _o, codes = run_and_get_code(fn, a, w, *extra)
    torch.cuda.synchronize()
    with open(dst, "w") as f:
        f.write("\n\n#### NEXT ####\n\n".join(codes))
    print("EXTRACTED " + dst)


# --------------------------------------------------------------------------------------------
# phase B: rebuild every arm from source and time them together
# --------------------------------------------------------------------------------------------


def _inductor_arm(which, code, kern, arms, info, IK, torch, triton, operands, out_buf, M, N):
    """Rebuild one arm from the text Inductor emitted for it.

    Raises rather than guessing when the argument binding cannot be resolved. A wrong binding
    does not fail loudly on its own -- Triton takes a bare pointer and reads whatever is there --
    so the caller records the failure and drops the arm.
    """
    tem = {k: v for k, v in kern.items() if "_tem_" in k}
    poi = {k: v for k, v in kern.items() if "_tem_" not in k}
    if tem and not poi:  # one fused template: the whole op is that kernel
        name, src = next(iter(tem.items()))
        meta = IK.launch_meta(src)
        grid = IK.grid_of(code, name)
        fn, path = IK.load(src, f"{which}_{name}", CACHE)
        order = IK.bind(code, src, name, operands)
        info[which].update(fused=True, gen_path=path, cfg=meta.get("constexpr", {}), num_warps=meta.get("num_warps"),
                           num_stages=meta.get("num_stages"), grid=list(grid), src_name=name,
                           enable_fp_fusion=meta.get("enable_fp_fusion", True), bind=[str(o) for o in order])

        def run(a, w, extra, fn=fn, grid=grid, meta=meta, order=order):
            c = out_buf()
            IK.launch(fn, tuple(grid), IK.call_args(order, a, w, c, extra), meta)
            return c

        arms[which] = run
    elif poi and not tem:  # extern mm plus one pointwise: the two-kernel path
        name, src = next(iter(poi.items()))
        meta = IK.launch_meta(src)
        fn, path = IK.load(src, f"{which}_{name}", CACHE)
        nel = M * N
        best = {}
        # Inductor names its pointwise arguments by role. `in_out_ptr0` means it decided to
        # overwrite the mm output in place, which is one buffer less than a naive read-write
        # pair, so the mapping has to follow the names rather than assume a shape.
        names = [a.split(":")[0].strip() for a in meta["args"]]
        inplace = any(n.startswith("in_out_ptr") for n in names)
        order = IK.bind(code, src, name, operands)
        info[which].update(fused=False, src_name=name, gen_path=path, arg_names=names, inplace=inplace, tune=best,
                           enable_fp_fusion=meta.get("enable_fp_fusion", True), bind=[str(o) for o in order])

        def run(a, w, extra, fn=fn, meta=meta, order=order, nel=nel, best=best, inplace=inplace):
            xb, nw = best.get("XBLOCK", 1024), best.get("num_warps", 4)
            mm = torch.mm(a, w)
            c = mm if inplace else out_buf()
            call = IK.call_args(order, a, w, c, extra, mm=mm)
            fn[(triton.cdiv(nel, xb), )](*call, XBLOCK=xb, num_warps=nw,
                                         enable_fp_fusion=meta.get("enable_fp_fusion", True))
            return c

        arms[which] = run
    else:
        info[which] = {"skip": f"{len(tem)} template + {len(poi)} pointwise kernels, not handled"}


def build_arms(M, N, K, epi, dtype_name, srcs, tune_ours, params):
    """name -> (callable(a, w, extra) -> output tensor, info dict). Nothing here compiles torch."""
    import torch
    import triton

    import inductor_kernel as IK
    import ours as O
    from cases import EPI_OPERANDS, out_dtype_for

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    odt = out_dtype_for(epi, dtype)
    operands = EPI_OPERANDS.get(epi, ())
    epi_id = O.EPI_ID.get(epi)
    needs_r = epi_id in O.EPI_NEEDS_R if epi_id is not None else False
    arms, info = {}, {}

    def out_buf():
        return torch.empty(M, N, device="cuda", dtype=odt)

    # ---- the Inductor arms, straight out of output_code.py -----------------------------------
    for which in srcs:
        kern = IK.extract(srcs[which])
        info[which] = {
            "n_template": sum("_tem_" in k for k in kern), "n_pointwise": sum("_tem_" not in k for k in kern),
            "kernels": list(kern)
        }
        try:
            _inductor_arm(which, srcs[which], kern, arms, info, IK, torch, triton, operands, out_buf, M, N)
        except Exception as e:  # noqa: BLE001
            # One arm that cannot be rebuilt must not cost the case its other arms.
            info[which] = dict(info[which], error=f"{type(e).__name__}: {e}"[:300])

    # ---- arm 1: cuBLAS then the same epilogue as one separate kernel --------------------------
    # Built from the SAME `EPI_SRC` text arm 3 is patched with, so arm 1 and arm 3 cannot drift
    # apart in the arithmetic, and the text `probe_pairs.py` swept over its complete domain is the
    # one both of them run. `ours.py::_epilogue` stays for the epilogues it uniquely implements.
    ecfg = {"BLK": 2048, "num_warps": 4}
    if epi in IK.EPI_SRC:
        efn, ekind = IK.standalone(epi, operands, params, CACHE)

        def exact2k(a, w, extra, efn=efn, ekind=ekind):
            return IK.launch_standalone(efn, ekind, torch.mm(a, w), extra, out_buf(), **ecfg)

        info["exact2k"] = {"fused": False, "tune": ecfg, "kernel": ekind, "epilogue": IK.EPI_SRC[epi]}
    else:

        def exact2k(a, w, extra):
            return O.launch_epi(torch.mm(a, w), odt, epi_id, extra[0] if needs_r else None, **ecfg)

        info["exact2k"] = {"fused": False, "tune": ecfg, "kernel": "ours.py"}
    arms["exact2k"] = exact2k

    # ---- arm 3: arm 2's source with only the epilogue arithmetic replaced ---------------------
    # `enable_fp_fusion=False` goes with it. It is a launch option rather than text, but without
    # it the compiler contracts the epilogue's rounding away (see `inductor_kernel.NO_FP_FUSION`),
    # so the patched text would not mean what it says. It is also the flag Inductor itself pairs
    # with `emulate_precision_casts`, so it is what a determinism user would be given.
    #
    # It is NOT free of the mainloop, and the record says so rather than assuming otherwise:
    # `mainloop_fp_fusion_delta` compiles the epilogue-stripped template both ways and diffs the
    # PTX, and on the seven smoke cases it came back CHANGED. Until that diff is diagnosed, arm 3
    # differs from arm 2 in one launch option as well as in the epilogue text, and
    # `res["_fp_fusion"]` in every record is where a reader sees how much.
    if "triton" in srcs and info.get("triton", {}).get("fused"):
        code = srcs["triton"]
        name = info["triton"]["src_name"]
        src = IK.extract(code)[name]
        meta = IK.launch_meta(src)
        grid = tuple(info["triton"]["grid"])
        order = IK.bind(code, src, name, operands)
        for label, key in (("ours", epi), ("ours_divrn", epi + "_divrn"), ("ours_helpers", epi + "_helpers"),
                           ("ours_cheap", epi + "_cheap"), ("ours_fast", epi + "_fast"), ("ours_approx",
                                                                                          epi + "_approx")):
            if key not in IK.EPI_SRC:
                continue
            try:
                fn, path = IK.load(IK.patch_epilogue(src, key, operands, params), f"{label}_{name}", CACHE)
            except Exception as e:  # noqa: BLE001
                info[label] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue

            def run(a, w, extra, fn=fn, grid=grid, meta=meta, order=order):
                c = out_buf()
                IK.launch(fn, grid, IK.call_args(order, a, w, c, extra), meta, enable_fp_fusion=IK.NO_FP_FUSION)
                return c

            arms[label] = run
            info[label] = {
                "fused": True, "from": name, "cfg": info["triton"].get("cfg"), "grid": list(grid), "num_warps":
                info["triton"].get("num_warps"), "num_stages": info["triton"].get("num_stages"), "gen_path": path,
                "enable_fp_fusion": IK.NO_FP_FUSION, "epilogue": IK.EPI_SRC[key]
            }
    return arms, info


def mainloop_fp_fusion_delta(IK, src, order, grid, args):
    """What `enable_fp_fusion=False` does to the GEMM, as opposed to the epilogue.

    Arm 3 turns the flag off so its inserted roundings survive -- with contraction on, LLVM folds
    them away. That is only fair if the flag leaves the GEMM alone, so this compiles the SAME
    kernel BOTH ways -- the template with its epilogue stripped to a bare store, so the only thing
    left is the mainloop -- and diffs the PTX.

    It reports what changed rather than just that something did, because "the mainloop's text
    moved" and "the mainloop's arithmetic moved" are very different findings and the next reader
    should not have to re-run this to tell them apart. Comments and blank lines are dropped first;
    what is compared is instructions.
    """
    bare = IK.patch_epilogue(src, "none", (), None, drop_extra_loads=True)
    fn, _p = IK.load(bare, "fpfuse_probe", CACHE)
    meta = IK.launch_meta(bare)
    _ = order
    got = []
    for flag in (True, False):
        k = IK.launch(fn, grid, args, meta, enable_fp_fusion=flag)
        if k is None or "ptx" not in getattr(k, "asm", {}):
            return {"error": "no ptx"}
        body = [ln.strip() for ln in k.asm["ptx"].split("\n")]
        got.append([ln for ln in body if ln and not ln.startswith("//")])
    a, b = got
    ops = []
    for lines in (a, b):
        c = {}
        for ln in lines:
            op = ln.split()[0].split(":")[-1]
            if "." in op:
                c[op] = c.get(op, 0) + 1
        ops.append(c)
    changed = {
        k: [ops[0].get(k, 0), ops[1].get(k, 0)]
        for k in set(ops[0]) | set(ops[1])
        if ops[0].get(k, 0) != ops[1].get(k, 0)
    }
    diff = [[x, y] for x, y in zip(a, b) if x != y]
    return {
        "unchanged": a == b, "n_lines": [len(a), len(b)], "n_lines_differing": len(diff), "instruction_counts_changed":
        changed, "first_differing": diff[:6]
    }


def tile_sweep(torch, M, N, K, epi, dtype_name, srcs, info, ref_digests, a0, w0, extra0, flush, rounds, reps, params):
    """Re-tune the tile for the exact epilogue -- and for the other two epilogues on the same grid.

    Inductor picked 128x128 / 4 warps while looking at `tl.sigmoid`. The correctly rounded divide
    needs more live registers, so that tile is the wrong one for it. Sweeping only arm 3 would be
    unfair, so the same sweep is run over arm 2's own source (`triton_tuned`) and over the
    approximate patch, and all three are reported.
    """
    import measure
    import inductor_kernel as IK
    from artifact import digest
    from cases import EPI_OPERANDS, out_dtype_for

    name = info["triton"]["src_name"]
    src = IK.extract(srcs["triton"])[name]
    operands = EPI_OPERANDS.get(epi, ())
    order = IK.bind(srcs["triton"], src, name, operands)
    odt = out_dtype_for(epi, a0.dtype)
    space = IK.tile_space(M, N, K, src)
    # what arm 2's own kernel produces, per BLOCK_K, so a re-tuned arm-2 config can be checked
    # against it. It is not byte-identical to eager, so eager cannot be its reference.
    self_ref = {}
    for bk in sorted({c[2] for c in space}):
        s2, grid2 = IK.retune(src, M, N, K, 128 if M >= 128 else M, 128 if N >= 128 else N, bk, 8, 4, 3)
        f2, _p2 = IK.load(s2, f"selfref_{bk}", CACHE)
        m2 = IK.launch_meta(s2)
        c2 = torch.empty(M, N, device="cuda", dtype=odt)
        IK.launch(f2, grid2, IK.call_args(order, a0, w0, c2, extra0), m2)
        torch.cuda.synchronize()
        self_ref[bk] = digest(torch, c2)
        del c2
    out = {}
    for label, key in (("ours_tuned", epi), ("ours_helpers_tuned", epi + "_helpers"),
                       ("ours_approx_tuned", epi + "_approx"), ("triton_tuned", None)):
        if key is not None and key not in IK.EPI_SRC:
            continue
        base = src if key is None else IK.patch_epilogue(src, key, operands, params)
        fuse = True if key is None else IK.NO_FP_FUSION
        fns, metas, n_bad, n_err = {}, {}, 0, 0
        for cfg in space:
            try:
                s2, grid = IK.retune(base, M, N, K, *cfg)
                fn, _p = IK.load(s2, f"{label}_{'_'.join(map(str, cfg))}", CACHE)
                meta = IK.launch_meta(s2)

                def run(a, w, extra, fn=fn, grid=grid, meta=meta, fuse=fuse):
                    c = torch.empty(M, N, device="cuda", dtype=odt)
                    IK.launch(fn, grid, IK.call_args(order, a, w, c, extra), meta, enable_fp_fusion=fuse)
                    return c

                got = run(a0, w0, extra0)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                n_err += 1
                continue
            if key is None:
                # arm 2 is not byte-identical to eager, so eager cannot be its gate. It must still
                # agree with its own untuned self at the same BLOCK_K -- a tile that read past the
                # end of an operand would not.
                if digest(torch, got) != self_ref[cfg[2]]:
                    n_bad += 1
                    continue
            elif not key.endswith("_approx") and digest(torch, got) != ref_digests[0]:
                n_bad += 1  # arm 3 must stay byte-identical to eager
                continue
            del got
            fns[cfg] = run
            metas[cfg] = (meta, grid)
        if not fns:
            out[label] = {"error": f"no usable config ({n_bad} byte-different, {n_err} failed)"}
            continue
        t = measure.best_ms(torch, {c: (lambda f=f: f(a0, w0, extra0))
                                    for c, f in fns.items()}, flush, rounds=rounds, reps=reps)
        best = min(t, key=lambda k: t[k])
        out[label] = {
            "fused": True, "from": name, "tile": list(best), "search_ms": t[best], "n_tried": len(space), "n_ok":
            len(fns), "n_byte_different": n_bad, "n_failed": n_err, "enable_fp_fusion": fuse, "top5":
            [[list(c), t[c]]
             for c in sorted(t, key=lambda k: t[k])[:5]], "epilogue": IK.EPI_SRC[key] if key else "inductor's own"
        }
        out[label]["_run"] = fns[best]
        print(
            f"    {label:18s} best {t[best]:.4f} ms at {best}   "
            f"({len(fns)}/{len(space)} usable, {n_bad} byte-different)", flush=True)
    return out


def tune_pointwise(torch, triton, arms, info, which, a, w, extra, flush, rounds, reps):
    """Inductor autotunes XBLOCK for its pointwise kernels at run time; do the same by hand."""
    import measure
    if which not in info or not info[which].get("tune") or "XBLOCK" in info[which]["tune"]:
        return
    best = info[which]["tune"]
    fns = {}
    for xb in (512, 1024, 2048, 4096):
        for nw in (4, 8):
            cfg = dict(XBLOCK=xb, num_warps=nw)
            try:
                best.clear()
                best.update(cfg)
                arms[which](a, w, extra)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"    {which} XBLOCK={xb} nw={nw} failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            fns[(xb, nw)] = (lambda b=cfg: (best.clear(), best.update(b), arms[which](a, w, extra))[-1])
    if not fns:
        raise RuntimeError(f"no working config for the {which} pointwise kernel")
    t = measure.best_ms(torch, fns, flush, rounds=rounds, reps=reps)
    key = min(t, key=lambda k: t[k])
    best.clear()
    best.update(XBLOCK=key[0], num_warps=key[1])
    info[which]["tuned"] = dict(best)


def do_time(M, N, K, epi, dtype_name, srcs, reps, rounds, draws, tune_ours, params):
    import torch
    import triton

    from artifact import digest, hot_cublas, make_inputs
    from cases import EPILOGUES, make_epi_args
    import measure
    import inductor_kernel as IK
    import ours as O

    if dtype_name != "fp16":
        # `EPI_SRC` spells its eager boundaries as `.to(tl.float16)`. A bf16 case rounds
        # somewhere else, so there is no arm 3 for it and measuring one would be measuring the
        # wrong function.
        raise RuntimeError(f"arm 3 is fp16-only; {dtype_name} has different rounding points")
    dtype = torch.float16
    seed = M * 1000003 + N * 10007 + K

    def inputs(r):
        a, w = make_inputs(torch, M, N, K, "fp16", r, seed)
        if dtype is torch.bfloat16:
            a, w = a.to(dtype), w.to(dtype)
        # odd draws spread the operand's exponents too, for the same reason `make_inputs` does
        return a, w, make_epi_args(epi, M, N, dtype, seed=seed + r, wide=(r % 2 == 1))

    arms, info = build_arms(M, N, K, epi, dtype_name, srcs, tune_ours, params)
    a0, w0, extra0 = inputs(0)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    print(f"  warming: {measure.warm(torch)}", flush=True)

    # tune the knobs Inductor tunes at run time, so no arm is handicapped by a bad hand-pick
    tune_pointwise(torch, triton, arms, info, "aten_e", a0, w0, extra0, flush, rounds, reps)
    ecfg = info["exact2k"]["tune"]
    fns = {}
    for blk in (1024, 2048, 4096, 8192):
        for nw in (4, 8):
            fns[(blk, nw)] = (lambda b=blk, n=nw: (ecfg.update(BLK=b, num_warps=n), arms["exact2k"]
                                                   (a0, w0, extra0))[-1])
    t = measure.best_ms(torch, fns, flush, rounds=rounds, reps=reps)
    ecfg.update(BLK=min(t, key=lambda k: t[k])[0], num_warps=min(t, key=lambda k: t[k])[1])

    # would Inductor have picked a different tile if its own search had seen the exact epilogue?
    if tune_ours:
        ref0 = digest(torch, EPILOGUES[epi](torch.mm(a0, w0), *extra0, **params))
        sweep = tile_sweep(torch, M, N, K, epi, dtype_name, srcs, info, [ref0, None], a0, w0, extra0, flush, rounds,
                           reps, params)
        for label, rec in sweep.items():
            run = rec.pop("_run", None)
            info[label] = rec
            if run is not None:
                arms[label] = run

    # bytes: `draws` independent draws, odd ones with the exponents spread over fp16's range
    res = {}
    for name, run in arms.items():
        rec = dict(info.get(name, {}))
        n_ok = n_wide = n_ok_wide = 0
        worst = None
        for r in range(draws):
            a, w, extra = inputs(r)
            try:
                got = run(a, w, extra)
                same = digest(torch, got) == digest(torch, EPILOGUES[epi](torch.mm(a, w), *extra, **params))
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                same = False
                got = None
            n_ok += int(same)
            if r % 2 == 1:
                n_wide += 1
                n_ok_wide += int(same)
            if not same and worst is None and got is not None:
                ref = EPILOGUES[epi](torch.mm(a, w), *extra, **params)
                d = got.to(torch.float32) != ref.to(torch.float32)
                worst = {
                    "draw": r, "n_differ": int(d.sum()), "n_total": int(ref.numel()), "max_abs": float(
                        (got.to(torch.float32) - ref.to(torch.float32)).abs().max())
                }
                del ref, d
            del a, w, extra, got
        rec.update(draws=draws, bit_ok=n_ok, wide_draws=n_wide, bit_ok_wide=n_ok_wide, bit_exact=n_ok == draws)
        if worst:
            rec["first_mismatch"] = worst
        res[name] = rec
    for name, rec in info.items():  # an arm that was never built still has to say why
        res.setdefault(name, rec)
    torch.cuda.empty_cache()

    # time: all arms round-robin in one process, plus the two floors
    timed = {k: (lambda run=run: run(a0, w0, extra0)) for k, run in arms.items() if "error" not in res[k]}
    from bitequiv.cublas_match import ltapi as L
    try:
        h = hot_cublas(L, torch, a0, w0, "fp16", dtype)
        timed["_hot_cublas_gemm"] = h.run
    except Exception as e:  # noqa: BLE001
        print("  hot_cublas failed:", str(e)[:120])
    if "triton" in srcs and info.get("triton", {}).get("fused"):
        name = info["triton"]["src_name"]
        src = IK.extract(srcs["triton"])[name]
        from cases import EPI_OPERANDS
        order = IK.bind(srcs["triton"], src, name, EPI_OPERANDS.get(epi, ()))
        try:
            # the floor is the GEMM with no epilogue AT ALL, so the extra operands' loads go too
            bare = IK.patch_epilogue(src, "none", (), None, drop_extra_loads=True)
            fn, _p = IK.load(bare, "mainloop_" + name, CACHE)
            meta = IK.launch_meta(bare)
            grid = tuple(info["triton"]["grid"])

            def mainloop(fn=fn, grid=grid, meta=meta, order=order):
                c = torch.empty(M, N, device="cuda", dtype=dtype)
                IK.launch(fn, grid, IK.call_args(order, a0, w0, c, extra0), meta)
                return c

            mainloop()
            torch.cuda.synchronize()
            timed["_mainloop_no_epilogue"] = mainloop
            args0 = IK.call_args(order, a0, w0, torch.empty(M, N, device="cuda", dtype=dtype), extra0)
            res["_fp_fusion"] = mainloop_fp_fusion_delta(IK, src, order, grid, args0)
        except Exception as e:  # noqa: BLE001
            print("  mainloop-only failed:", str(e)[:120])
    for fn in timed.values():
        fn()
    torch.cuda.synchronize()
    ms = measure.best_ms(torch, timed, flush, rounds=rounds, reps=reps)
    for k, v in ms.items():
        res.setdefault(k, {})["ms"] = v
    _ = O  # imported for EPI ids via build_arms
    return res


# --------------------------------------------------------------------------------------------


def parse_case(tok):
    """`M,N,K,epi` plus optional `key=value` fields for the epilogue's scalars.

    `4096,4096,16,silu` is the old form and still works. `16384,2048,32,lora,scale=0.5` pins
    alpha/rank, which is a different kernel -- the constant is baked into the generated text --
    so it also has to be part of the compile cache key.
    """
    from cases import params_for
    parts = [t.strip() for t in tok.split(",")]
    M, N, K, epi = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
    over = {}
    for f in parts[4:]:
        k, v = f.split("=")
        over[k.strip()] = float(v)
    return M, N, K, epi, params_for(epi, over)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="", help="semicolon-separated M,N,K,epi[,scale=..][,lim=..]")
    p.add_argument("--cases-file", default="", help="a file of the same, one per line")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--draws", type=int, default=12)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--tune-ours", action="store_true")
    p.add_argument("--tag", default="")  # stamped on each record, and what --skip-done keys on
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--extract", default="")  # internal
    args = p.parse_args()

    from cases import params_tag

    if args.extract:
        M, N, K, epi, params = parse_case(args.cases)
        dst = os.path.join(CACHE, f"{M}x{N}x{K}_{epi}{params_tag(epi, params)}_{args.dtype}_{args.extract}.py")
        do_extract(M, N, K, epi, args.dtype, args.extract, dst, params)
        return 0

    # Resuming: a case already measured under this tag is skipped, so a run that dies part way
    # through -- or gets reaped -- picks up where it stopped instead of redoing hours of it.
    have = set()
    if args.skip_done and os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("tag") == args.tag:
                    have.add((r["M"], r["N"], r["K"], r["epi"], r["dtype"], r.get("params_tag", "")))

    rows = []
    toks = [t.strip() for t in args.cases.split(";") if t.strip()]
    if args.cases_file:
        toks += [ln.split("#")[0].strip() for ln in open(args.cases_file) if ln.split("#")[0].strip()]
    if not toks:
        p.error("give --cases or --cases-file")
    for i, tok in enumerate(toks):
        M, N, K, epi, params = parse_case(tok)
        ptag = params_tag(epi, params)
        if (M, N, K, epi, args.dtype, ptag) in have:
            print(f"\n=== [{i + 1}/{len(toks)}] {M}x{N}x{K} {epi}{ptag} {args.dtype} -- already done, skipping ===",
                  flush=True)
            continue
        print(f"\n=== [{i + 1}/{len(toks)}] {M}x{N}x{K} {epi}{ptag} {args.dtype} ===", flush=True)
        srcs = {}
        for which, _b, _e in COMPILES:
            dst = os.path.join(CACHE, f"{M}x{N}x{K}_{epi}{ptag}_{args.dtype}_{which}.py")
            if not os.path.exists(dst):
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__), "--cases", tok, "--extract", which, "--dtype", args.dtype
                ]
                t0 = time.time()
                cp = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=ROOT), timeout=args.timeout,
                                    capture_output=True, text=True)
                if not os.path.exists(dst):
                    print(f"  compile {which:9s} FAILED\n{(cp.stderr or cp.stdout)[-700:]}", flush=True)
                    continue
                print(f"  compile {which:9s} {time.time() - t0:.0f}s", flush=True)
            srcs[which] = open(dst).read()
        if "triton" not in srcs:
            print("  no TRITON source, skipping")
            continue
        try:
            res = do_time(M, N, K, epi, args.dtype, srcs, args.reps, args.rounds, args.draws, args.tune_ours, params)
        except Exception as e:  # noqa: BLE001
            # One shape that runs out of memory or trips a compile bug must not take the rest of
            # the sweep with it. Record the failure and move on.
            print(f"  FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            log(
                ATTEMPTS, {
                    "step": "three_way", "tag": args.tag, "M": M, "N": N, "K": K, "epi": epi, "params": params, "dtype":
                    args.dtype, "error": f"{type(e).__name__}: {str(e)[:300]}"
                })
            continue
        rec = {
            "M": M, "N": N, "K": K, "epi": epi, "params": params, "params_tag": ptag, "dtype": args.dtype, "tag":
            args.tag, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "draws": args.draws, "reps": args.reps, "rounds":
            args.rounds, **res
        }
        for name in ("exact2k", "aten_e", "triton", "triton_e", "ours", "ours_divrn", "ours_helpers", "ours_cheap",
                     "ours_fast", "ours_approx", "triton_tuned", "ours_tuned", "ours_helpers_tuned",
                     "ours_approx_tuned"):
            r = res.get(name) or {}
            if "ms" in r:
                print(
                    f"  {name:12s} {r['ms']:.4f} ms   bytes {r.get('bit_ok')}/{r.get('draws')} "
                    f"(wide {r.get('bit_ok_wide')}/{r.get('wide_draws')})"
                    f"{'  FUSED' if r.get('fused') else ''}{'  ' + r['error'][:80] if r.get('error') else ''}",
                    flush=True)
            elif r:
                print(f"  {name:12s} -            {r.get('error', r.get('skip', ''))[:100]}", flush=True)
        for name in ("_hot_cublas_gemm", "_mainloop_no_epilogue"):
            if name in res:
                print(f"  {name:12s} {res[name]['ms']:.4f} ms", flush=True)
        o = (res.get("ours") or {}).get("ms")
        if o:
            for arm in ("exact2k", "aten_e", "triton", "triton_e", "triton_tuned"):
                t = (res.get(arm) or {}).get("ms")
                rec[f"r_{arm}_over_ours"] = (t / o) if t else None
            ot = (res.get("ours_tuned") or {}).get("ms")
            if ot:
                for arm in ("exact2k", "aten_e", "triton", "triton_e", "triton_tuned"):
                    t = (res.get(arm) or {}).get("ms")
                    rec[f"r_{arm}_over_ours_tuned"] = (t / ot) if t else None
            print(
                f"  ==> at Inductor's own tile:  2/3 = {rec.get('r_triton_over_ours'):.3f}"
                f"   1/3 = {rec.get('r_exact2k_over_ours'):.3f}", flush=True)
            if rec.get("r_triton_over_ours_tuned"):
                print(
                    f"  ==> ours re-tuned:           2/3 = {rec['r_triton_over_ours_tuned']:.3f}"
                    f"   1/3 = {rec['r_exact2k_over_ours_tuned']:.3f}"
                    f"   2tuned/3tuned = {rec.get('r_triton_tuned_over_ours_tuned') or float('nan'):.3f}", flush=True)
        rows.append(rec)
        log(OUT, rec)
        log(ATTEMPTS, {"step": "three_way", **rec})

    good = [r for r in rows if (r.get("ours") or {}).get("bit_exact")]
    if good:
        gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
        print(f"\n  {len(good)}/{len(rows)} cases byte-identical on all {args.draws} draws")
        for suffix in ("ours", "ours_tuned"):
            print(f"  -- against {suffix} --")
            for arm, label in (("exact2k", "1/3   bit-exact two-kernel"), ("aten_e", "1i/3  Inductor ATEN two-kernel"),
                               ("triton", "2/3   Inductor fused"), ("triton_e", "2e/3  Inductor fused + emulate"),
                               ("triton_tuned", "2t/3  Inductor fused, same tile sweep")):
                v = [r[f"r_{arm}_over_{suffix}"] for r in good if r.get(f"r_{arm}_over_{suffix}")]
                if v:
                    print(f"  {label:40s} geomean {gm(v):.4f}   per case {', '.join(f'{x:.3f}' for x in v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
