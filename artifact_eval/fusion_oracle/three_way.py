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

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> .venv/bin/python three_way.py --cases 4096,4096,16,silu
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
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


# --------------------------------------------------------------------------------------------
# phase A: compile in a subprocess, save the generated source
# --------------------------------------------------------------------------------------------


def do_extract(M, N, K, epi, dtype_name, which, dst):
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
        fn = torch.compile(lambda x, y, *rest: EPILOGUES[epi](x @ y, *rest), mode="max-autotune-no-cudagraphs")
        _o, codes = run_and_get_code(fn, a, w, *extra)
    torch.cuda.synchronize()
    with open(dst, "w") as f:
        f.write("\n\n#### NEXT ####\n\n".join(codes))
    print("EXTRACTED " + dst)


# --------------------------------------------------------------------------------------------
# phase B: rebuild every arm from source and time them together
# --------------------------------------------------------------------------------------------


def build_arms(M, N, K, epi, dtype_name, srcs, tune_ours):
    """name -> (callable(a, w, extra) -> output tensor, info dict). Nothing here compiles torch."""
    import torch
    import triton

    import inductor_kernel as IK
    import ours as O

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    odt = O.FP8 if epi == "fp8cast" else dtype
    epi_id = O.EPI_ID[epi]
    needs_r = epi_id in O.EPI_NEEDS_R
    arms, info = {}, {}

    def out_buf():
        return torch.empty(M, N, device="cuda", dtype=odt)

    # ---- the Inductor arms, straight out of output_code.py -----------------------------------
    for which in srcs:
        code = srcs[which]
        kern = IK.extract(code)
        tem = {k: v for k, v in kern.items() if "_tem_" in k}
        poi = {k: v for k, v in kern.items() if "_tem_" not in k}
        info[which] = {"n_template": len(tem), "n_pointwise": len(poi), "kernels": list(kern)}
        if tem and not poi:  # one fused template: the whole op is that kernel
            name, src = next(iter(tem.items()))
            meta = IK.launch_meta(src)
            grid = IK.grid_of(code, name)
            fn, path = IK.load(src, f"{which}_{name}", CACHE)
            info[which].update(fused=True, gen_path=path, cfg=meta.get("constexpr",
                                                                       {}), num_warps=meta.get("num_warps"),
                               num_stages=meta.get("num_stages"), grid=list(grid), src_name=name)
            n_extra = len(meta["args"]) - 2 - 1  # arg_A, arg_B, ... , out_ptr

            def run(a, w, extra, fn=fn, grid=grid, meta=meta, n_extra=n_extra):
                c = out_buf()
                IK.launch(fn, tuple(grid), (a, w) + tuple(extra[:n_extra]) + (c, ), meta)
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
            info[which].update(fused=False, src_name=name, gen_path=path, arg_names=names, inplace=inplace, tune=best)

            def run(a, w, extra, fn=fn, names=names, nel=nel, best=best, inplace=inplace):
                xb, nw = best.get("XBLOCK", 1024), best.get("num_warps", 4)
                mm = torch.mm(a, w)
                c = mm if inplace else out_buf()
                nxt = list(extra)
                call = []
                for n in names:
                    if n.startswith("in_out_ptr"):
                        call.append(mm)
                    elif n == "in_ptr0":
                        call.append(mm)
                    elif n.startswith("in_ptr"):
                        call.append(nxt.pop(0))
                    elif n.startswith("out_ptr"):
                        call.append(c)
                    elif n == "xnumel":
                        call.append(nel)
                fn[(triton.cdiv(nel, xb), )](*call, XBLOCK=xb, num_warps=nw)
                return c

            arms[which] = run
        else:
            info[which] = {"skip": f"{len(tem)} template + {len(poi)} pointwise kernels, not handled"}

    # ---- arm 1: cuBLAS then our own bit-exact epilogue kernel ---------------------------------
    ecfg = {"BLK": 2048, "num_warps": 4}

    def exact2k(a, w, extra):
        return O.launch_epi(torch.mm(a, w), odt, epi_id, extra[0] if needs_r else None, **ecfg)

    arms["exact2k"] = exact2k
    info["exact2k"] = {"fused": False, "tune": ecfg}

    # ---- arm 3: arm 2's source with only the epilogue arithmetic replaced ---------------------
    if "triton" in srcs and info.get("triton", {}).get("fused"):
        code = srcs["triton"]
        name = info["triton"]["src_name"]
        src = IK.extract(code)[name]
        meta = IK.launch_meta(src)
        grid = tuple(info["triton"]["grid"])
        n_extra = len(meta["args"]) - 3
        for label, key in (("ours", epi), ("ours_divrn", epi + "_divrn"), ("ours_helpers", epi + "_helpers"),
                           ("ours_approx", epi + "_approx")):
            if key not in IK.EPI_SRC:
                continue
            try:
                fn, path = IK.load(IK.patch_epilogue(src, key), f"{label}_{name}", CACHE)
            except Exception as e:  # noqa: BLE001
                info[label] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue

            def run(a, w, extra, fn=fn, grid=grid, meta=meta, n_extra=n_extra):
                c = out_buf()
                IK.launch(fn, grid, (a, w) + tuple(extra[:n_extra]) + (c, ), meta)
                return c

            arms[label] = run
            info[label] = {
                "fused": True, "from": name, "cfg": info["triton"].get("cfg"), "grid": list(grid), "num_warps":
                info["triton"].get("num_warps"), "num_stages": info["triton"].get("num_stages"), "gen_path": path,
                "epilogue": IK.EPI_SRC[key]
            }
    return arms, info


def tile_sweep(torch, M, N, K, epi, dtype_name, srcs, info, ref_digests, a0, w0, extra0, flush, rounds, reps):
    """Re-tune the tile for the exact epilogue -- and for the other two epilogues on the same grid.

    Inductor picked 128x128 / 4 warps while looking at `tl.sigmoid`. The correctly rounded divide
    needs more live registers, so that tile is the wrong one for it. Sweeping only arm 3 would be
    unfair, so the same sweep is run over arm 2's own source (`triton_tuned`) and over the
    approximate patch, and all three are reported.
    """
    import measure
    import inductor_kernel as IK
    from artifact import digest

    name = info["triton"]["src_name"]
    src = IK.extract(srcs["triton"])[name]
    meta0 = IK.launch_meta(src)
    n_extra = len(meta0["args"]) - 3
    odt = a0.dtype if epi != "fp8cast" else __import__("ours").FP8
    space = IK.tile_space(M, N, K, src)
    # what arm 2's own kernel produces, per BLOCK_K, so a re-tuned arm-2 config can be checked
    # against it. It is not byte-identical to eager, so eager cannot be its reference.
    self_ref = {}
    for bk in sorted({c[2] for c in space}):
        s2, grid2 = IK.retune(src, M, N, K, 128 if M >= 128 else M, 128 if N >= 128 else N, bk, 8, 4, 3)
        f2, _p2 = IK.load(s2, f"selfref_{bk}", CACHE)
        m2 = IK.launch_meta(s2)
        c2 = torch.empty(M, N, device="cuda", dtype=odt)
        IK.launch(f2, grid2, (a0, w0) + tuple(extra0[:n_extra]) + (c2, ), m2)
        torch.cuda.synchronize()
        self_ref[bk] = digest(torch, c2)
        del c2
    out = {}
    for label, key in (("ours_tuned", epi), ("ours_helpers_tuned", epi + "_helpers"),
                       ("ours_approx_tuned", epi + "_approx"), ("triton_tuned", None)):
        if key is not None and key not in IK.EPI_SRC:
            continue
        base = src if key is None else IK.patch_epilogue(src, key)
        fns, metas, n_bad, n_err = {}, {}, 0, 0
        for cfg in space:
            try:
                s2, grid = IK.retune(base, M, N, K, *cfg)
                fn, _p = IK.load(s2, f"{label}_{'_'.join(map(str, cfg))}", CACHE)
                meta = IK.launch_meta(s2)

                def run(a, w, extra, fn=fn, grid=grid, meta=meta):
                    c = torch.empty(M, N, device="cuda", dtype=odt)
                    IK.launch(fn, grid, (a, w) + tuple(extra[:n_extra]) + (c, ), meta)
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
            len(fns), "n_byte_different": n_bad, "n_failed": n_err, "top5":
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


def do_time(M, N, K, epi, dtype_name, srcs, reps, rounds, draws, tune_ours):
    import torch
    import triton

    from artifact import digest, hot_cublas, make_inputs
    from cases import EPILOGUES, make_epi_args
    import measure
    import inductor_kernel as IK
    import ours as O

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    seed = M * 1000003 + N * 10007 + K

    def inputs(r):
        a, w = make_inputs(torch, M, N, K, "fp16", r, seed)
        if dtype is torch.bfloat16:
            a, w = a.to(dtype), w.to(dtype)
        return a, w, make_epi_args(epi, M, N, dtype, seed=seed + r)

    arms, info = build_arms(M, N, K, epi, dtype_name, srcs, tune_ours)
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
        ref0 = digest(torch, EPILOGUES[epi](torch.mm(a0, w0), *extra0))
        sweep = tile_sweep(torch, M, N, K, epi, dtype_name, srcs, info, [ref0, None], a0, w0, extra0, flush, rounds,
                           reps)
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
                same = digest(torch, got) == digest(torch, EPILOGUES[epi](torch.mm(a, w), *extra))
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                same = False
                got = None
            n_ok += int(same)
            if r % 2 == 1:
                n_wide += 1
                n_ok_wide += int(same)
            if not same and worst is None and got is not None:
                ref = EPILOGUES[epi](torch.mm(a, w), *extra)
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
        src = IK.extract(srcs["triton"])[info["triton"]["src_name"]]
        try:
            fn, _p = IK.load(IK.patch_epilogue(src, "none"), "mainloop_" + info['triton']['src_name'], CACHE)
            meta = IK.launch_meta(src)
            grid = tuple(info["triton"]["grid"])
            n_extra = len(meta["args"]) - 3

            def mainloop(fn=fn, grid=grid, meta=meta, n_extra=n_extra):
                c = torch.empty(M, N, device="cuda", dtype=dtype)
                IK.launch(fn, grid, (a0, w0) + tuple(extra0[:n_extra]) + (c, ), meta)
                return c

            mainloop()
            torch.cuda.synchronize()
            timed["_mainloop_no_epilogue"] = mainloop
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, help="semicolon-separated M,N,K,epi")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--draws", type=int, default=12)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--tune-ours", action="store_true")
    p.add_argument("--extract", default="")  # internal
    args = p.parse_args()

    if args.extract:
        M, N, K, epi = args.cases.split(",")
        dst = os.path.join(CACHE, f"{M}x{N}x{K}_{epi}_{args.dtype}_{args.extract}.py")
        do_extract(int(M), int(N), int(K), epi, args.dtype, args.extract, dst)
        return 0

    rows = []
    for tok in args.cases.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        M, N, K, epi = tok.split(",")
        print(f"\n=== {M}x{N}x{K} {epi} {args.dtype} ===", flush=True)
        srcs = {}
        for which, _b, _e in COMPILES:
            dst = os.path.join(CACHE, f"{M}x{N}x{K}_{epi}_{args.dtype}_{which}.py")
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
        res = do_time(int(M), int(N), int(K), epi, args.dtype, srcs, args.reps, args.rounds, args.draws, args.tune_ours)
        rec = {
            "M": int(M), "N": int(N), "K": int(K), "epi": epi, "dtype": args.dtype, "when":
            time.strftime("%Y-%m-%d %H:%M:%S"), "draws": args.draws, "reps": args.reps, "rounds": args.rounds, **res
        }
        for name in ("exact2k", "aten_e", "triton", "triton_e", "ours", "ours_divrn", "ours_helpers", "ours_approx",
                     "triton_tuned", "ours_tuned", "ours_helpers_tuned", "ours_approx_tuned"):
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
