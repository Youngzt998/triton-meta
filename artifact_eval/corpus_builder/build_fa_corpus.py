"""Build the FLASH-ATTENTION slice of the local_evaluation corpus -- MAXIMUM config space.

We RECONSTRUCT flash-attention from the canonical, runnable Triton FA kernel ``_attn_fwd``
in ``python/tutorials/06-fused-attention.py`` (the three jit functions below are copied
VERBATIM from that file, with ONE minimal generalization -- a ``BF16`` constexpr so the
non-fp8 path can emit bf16 as well as fp16, see the marked lines in ``_attn_fwd``). We drive
it directly with ``.warmup`` at each config and emit the standard corpus format so later
checker experiments read cached PTX + empirical keys (no recompile, no re-fuzz).

The corpus is a cache: whatever FA config an experiment later needs is pulled from disk. So
this builds the FULL cross-product of every meaningful, SUPPORTED axis (not a small slice):

  GENUINE-SPLITTER axes (distinct kernel instances -- must stay bit-distinct, test SOUNDNESS):
    * causal      {False, True}           -> STAGE 1 vs 3
    * HEAD_DIM    {64, 128, 256}
    * dtype       {f16, bf16, fp8e5m2}    -- one <dtype> subdir each

  TILING / autotune axes (bit-neutral candidates -- test RECOVERY):
    * num_warps   {1, 2, 4, 8, 16}
    * BLOCK_M     {16, 32, 64, 128, 256}
    * BLOCK_N     {16, 32, 64, 128, 256}  (kept <= HEAD_DIM: kernel static_assert)
    * num_stages  {1, 2, 3, 4, 5, 6, 7}

  Fixed shape: S = N_CTX = 2048, Z = 1, H = 2 (empirical_key depends on S, so one S keeps
  configs comparable).

DROPPED axes (exposed by the kernel but NOT usable on this Triton build -- verified by probe):
  * warp_specialize=True -> crashes TritonGPUAutomaticWarpSpecialization / LoadMMASpecialization
      ("'ttng.tmem_alloc' op operation destroyed but still has uses") for EVERY fwd config
      tried. Swept warp_specialize=False only.
  * num_ctas=2 (clusters) -> crashes the MLIR pass pipeline for this FA kernel. Swept
      num_ctas=1 only.
  * fp8: the tutorial kernel's fp8 code path is float8_e5m2 (tl.float8e5), NOT the e4m3 the
      task named, so we sweep e5m2 (what the kernel actually supports) and label it fp8e5m2.

Config VALIDITY prune (mirrors upstream ``prune_invalid_configs`` so the grid matches configs
the real autotuner would pick -- not arbitrary): BLOCK_N <= HEAD_DIM (static_assert),
BLOCK_M <= N_CTX, and BLOCK_M >= BLOCK_N when causal.

Corpus layout (same as build_local_eval.py):
  corpus/flash_attention/<dtype>/<cid>.{ptx,ttgir,json}
  json = {kernel, dtype, config, size, seeds, ok, empirical_key, ptx, note}
  empirical_key = sha1 of concatenated per-seed output sha1 digests (same formula as
  build_local_eval.py). The fuzzed output is the attention result tensor ``o``.

Backward compatibility: the 48 pre-existing f16 json (40 ok + 8 infeasible) were keyed by the
OLD cid ``sha1(("flash_attention",(num_warps,BLOCK_M,BLOCK_N,num_stages)))``. ``_cid`` below
reproduces that EXACT cid for the historical default sub-cube (f16, non-causal, HEAD_DIM=128,
S=2048, ws=False, ctas=1) so those files are skipped as-is on resume; every config differing
in a new axis gets a fresh distinct cid.

Resumable (skip existing json whose seeds match) + timeout/oomd-safe (SIGALRM -> os._exit(3)
so systemd Restart resumes past a wedged config). Deterministic compiler crashes and
CUDA-context-poisoning errors are marked infeasible (never retried forever); only genuine env
errors (GPU busy, libtriton relink) are treated as transient and retried. The checker is NOT
run here. Shardable via --shard/--nshards (cid-hash split) like build_local_eval.py.
"""
import argparse
import hashlib
import itertools
import json
import os
import signal
import sys

os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
# Repo root, two levels up from artifact_eval/corpus_builder/. `bitequiv` is not installed,
# so it has to be on the path; keep this in step with PYTHONPATH in the artifact README.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# Where the corpus is written. Same name and same default as the `checker.corpus` step reads,
# so building it and grading it cannot end up pointed at two different directories.
CORPUS = os.environ.get("CHECKER_CORPUS") or os.path.expanduser("~/bitwise-equiv/local_evaluation/corpus")
DEVICE = "cuda"

# ========================================================================================= #
# Kernel: copied VERBATIM from python/tutorials/06-fused-attention.py (OpenAI kernel team).
# Only the forward path is needed. The SOLE change vs upstream is the `BF16` constexpr in
# `_attn_fwd` (marked below) so the non-fp8 path can pick bf16 as well as fp16; everything
# else -- including `_attn_fwd_inner` and its `dtype == tl.float8e5` branches -- is unchanged.
# ========================================================================================= #


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              BF16: tl.constexpr,  # <-- only change vs upstream: select bf16 for the non-fp8 path
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    if FP8_OUTPUT:
        dtype = tl.float8e5
    elif BF16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# ========================================================================================= #
# Corpus builder around the copied kernel.
# ========================================================================================= #
_Z, _H, _N_CTX = 1, 2, 2048
_SM_SCALE = 0.5
_IS_HOPPER = False  # this box is Blackwell (cuda:10x)

# Genuine-splitter axes.
_DTYPES = ("f16", "bf16", "fp8e5m2")
_CAUSAL = (False, True)
_HEAD_DIM = (64, 128, 256)
# Tiling axes.
_NUM_WARPS = (1, 2, 4, 8, 16)
_BLOCK_M = (16, 32, 64, 128, 256)
_BLOCK_N = (16, 32, 64, 128, 256)
_NUM_STAGES = (1, 2, 3, 4, 5, 6, 7)
# Dropped axes -- fixed (see module docstring for why).
_WARP_SPECIALIZE = False
_NUM_CTAS = 1

# Deterministic, per-config failures -> mark infeasible (ok=false), do NOT retry.
_PERMANENT_TYPES = ("CompilationError", "AssertionError", "ValueError", "OutOfResources", "KeyError", "TypeError",
                    "NotImplementedError", "IndexError", "RuntimeError")
# Genuine env/contention failures -> retry later (write no json). Matched in the message.
_TRANSIENT_MARKERS = ("invalid elf", "busy or unavailable", "no cuda-capable device", "cannot find ptxas", "nvml",
                      "out of memory")
# CUDA-context-poisoning failures -> mark infeasible AND hard-exit so systemd restarts with a
# fresh context (a sticky CUDA error would otherwise fail every later config in this process).
_POISON_MARKERS = ("illegal memory access", "misaligned address", "device-side assert", "an illegal instruction",
                   "unspecified launch failure")
_TIMEOUT_S = 240
_current = {"jpath": None, "rec": None}


def _classify(e):
    """Return 'transient' (retry), 'poison' (infeasible + hard-exit), or 'infeasible'."""
    msg = (str(e) or repr(e)).lower()
    if any(m in msg for m in _POISON_MARKERS):
        return "poison"
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return "transient"
    if type(e).__name__ in _PERMANENT_TYPES:
        return "infeasible"
    return "infeasible"  # unknown (e.g. MLIR pass crash) -> infeasible so the build always finishes


def _on_alarm(signum, frame):
    jp, rec = _current["jpath"], _current["rec"]
    if jp and rec is not None:
        rec.update(ok=False, error=f"timeout: compile/run wedged > {_TIMEOUT_S}s")
        try:
            json.dump(rec, open(jp + ".tmp", "w"))
            os.replace(jp + ".tmp", jp)
        except Exception:  # noqa: BLE001
            pass
    print(f"  TIMEOUT on {_current.get('jpath')} -> marked skip, exiting for systemd restart", flush=True)
    os._exit(3)


signal.signal(signal.SIGALRM, _on_alarm)


def _to_bytes(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _qkv(dtype, head_dim, seed):
    """Per-seed q/k/v. f16 generation is byte-identical to the original build so the existing
    f16 empirical_keys stay comparable. bf16/fp8e5m2 mirror the tutorial's own construction."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (_Z, _H, _N_CTX, head_dim)

    def base():
        return torch.randn(shape, generator=g, dtype=torch.float32) * 0.5

    if dtype == "f16":
        q = base().to(DEVICE, torch.float16).contiguous()
        k = base().to(DEVICE, torch.float16).contiguous()
        v = base().to(DEVICE, torch.float16).contiguous()
    elif dtype == "bf16":
        q = base().to(DEVICE, torch.bfloat16).contiguous()
        k = base().to(DEVICE, torch.bfloat16).contiguous()
        v = base().to(DEVICE, torch.bfloat16).contiguous()
    elif dtype == "fp8e5m2":
        q = base().to(DEVICE, torch.float16).to(torch.float8_e5m2)
        k = base().to(DEVICE, torch.float16).to(torch.float8_e5m2)
        v = base().to(DEVICE, torch.float16)
        v = v.permute(0, 1, 3, 2).contiguous()  # tutorial's fp8 v layout: transposed-contiguous
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    else:
        raise ValueError(f"bad dtype {dtype}")
    return q, k, v


def _alloc(nbytes, alignment, stream):
    return torch.empty(nbytes, device=DEVICE, dtype=torch.int8)


def _tensors(dtype, head_dim, seed):
    q, k, v = _qkv(dtype, head_dim, seed)
    o = torch.empty_like(q)
    M = torch.empty((_Z, _H, _N_CTX), device=DEVICE, dtype=torch.float32)
    return q, k, v, o, M


# We pass RAW tensors (not host TensorDescriptor) as desc_q/k/v/o: the kernel's
# _maybe_make_tensor_desc then builds DEVICE-side descriptors. Raw tensors are plain pointer
# args, so the compiled-kernel launch (ck[grid]) does not hit the host-descriptor ABI
# signature mismatch. Needs the global scratch allocator (set below).
def _warmup(cfg, dtype, seed):
    triton.set_allocator(_alloc)
    hd, bm, bn = cfg["HEAD_DIM"], cfg["BLOCK_M"], cfg["BLOCK_N"]
    q, k, v, o, M = _tensors(dtype, hd, seed)
    stage = 3 if cfg["causal"] else 1
    grid = (triton.cdiv(_N_CTX, bm), _Z * _H, 1)
    ck = _attn_fwd.warmup(_SM_SCALE, M, _Z, _H, q, k, v, o, _N_CTX, HEAD_DIM=hd, BLOCK_M=bm, BLOCK_N=bn,
                          FP8_OUTPUT=(dtype == "fp8e5m2"), BF16=(dtype == "bf16"), STAGE=stage,
                          warp_specialize=_WARP_SPECIALIZE, IS_HOPPER=_IS_HOPPER, grid=grid, num_warps=cfg["num_warps"],
                          num_stages=cfg["num_stages"], num_ctas=_NUM_CTAS)
    return ck, grid


def _compile(cfg, dtype):
    return _warmup(cfg, dtype, 0)[0]


def _run_seed(cfg, dtype, ck, seed):
    """Reuse the already-compiled kernel across seeds (compile once, launch many)."""
    triton.set_allocator(_alloc)
    hd, bm = cfg["HEAD_DIM"], cfg["BLOCK_M"]
    q, k, v, o, M = _tensors(dtype, hd, seed)
    grid = (triton.cdiv(_N_CTX, bm), _Z * _H, 1)
    ck[grid](_SM_SCALE, M, _Z, _H, q, k, v, o, _N_CTX)
    torch.cuda.synchronize()
    return _to_bytes(o)  # fuzz the attention output


def _cid(cfg, dtype):
    # Backward-compat: reproduce the original 48 f16 json's cid for the historical default
    # sub-cube so they are skipped as-is; any config differing in a new axis gets a new cid.
    is_default = (dtype == "f16" and cfg["causal"] is False and cfg["HEAD_DIM"] == 128 and cfg["S"] == 2048
                  and cfg["warp_specialize"] is False and cfg["num_ctas"] == 1)
    if is_default:
        legacy = (cfg["num_warps"], cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["num_stages"])
        return hashlib.sha1(repr(("flash_attention", legacy)).encode()).hexdigest()[:16]
    return hashlib.sha1(repr(("flash_attention", dtype, tuple(sorted(cfg.items())))).encode()).hexdigest()[:16]


def _valid(cfg):
    """Mirror upstream prune_invalid_configs so the grid matches configs the real autotuner
    would pick: BLOCK_N <= HEAD_DIM (static_assert), BLOCK_M <= N_CTX, BLOCK_M >= BLOCK_N if causal."""
    if cfg["BLOCK_N"] > cfg["HEAD_DIM"]:
        return False
    if cfg["BLOCK_M"] > _N_CTX:
        return False
    if cfg["causal"] and cfg["BLOCK_M"] < cfg["BLOCK_N"]:
        return False
    return True


def _all_configs():
    cfgs = []
    for causal, hd, nw, bm, bn, ns in itertools.product(_CAUSAL, _HEAD_DIM, _NUM_WARPS, _BLOCK_M, _BLOCK_N,
                                                        _NUM_STAGES):
        cfg = {
            "num_warps": nw, "BLOCK_M": bm, "BLOCK_N": bn, "num_stages": ns, "causal": causal, "HEAD_DIM": hd, "S":
            _N_CTX, "warp_specialize": _WARP_SPECIALIZE, "num_ctas": _NUM_CTAS
        }
        if _valid(cfg):
            cfgs.append(cfg)
    return cfgs


def build(seeds, shard, nshards):
    cfgs = _all_configs()
    print(
        f"flash_attention MAX corpus: {len(cfgs)} valid configs (before dtype split) x "
        f"{len(_DTYPES)} dtypes; seeds={seeds} shard={shard}/{nshards} S={_N_CTX} Z={_Z} H={_H}", flush=True)
    built = infeasible = skipped = 0
    for dtype in _DTYPES:
        outdir = os.path.join(CORPUS, "flash_attention", dtype)
        os.makedirs(outdir, exist_ok=True)
        for cfg in cfgs:
            cid = _cid(cfg, dtype)
            if int(cid, 16) % nshards != shard:
                continue
            jpath = os.path.join(outdir, cid + ".json")
            if os.path.exists(jpath):
                try:
                    cached_seeds = json.load(open(jpath)).get("seeds")
                except Exception:  # noqa: BLE001
                    cached_seeds = None
                if cached_seeds == seeds:
                    skipped += 1
                    continue
                os.remove(jpath)  # stale seed count -> rebuild (empirical_key is seed-count dependent)
            stage = 3 if cfg["causal"] else 1
            rec = {
                "kernel":
                "flash_attention", "dtype":
                dtype, "config":
                dict(cfg), "size": [_Z, _H, _N_CTX, cfg["HEAD_DIM"]], "seeds":
                seeds, "note":
                f"FA fwd (06-fused-attention _attn_fwd, +bf16 generalization), "
                f"STAGE={stage} dtype={dtype}; output=o"
            }
            _current["jpath"], _current["rec"] = jpath, rec
            signal.alarm(_TIMEOUT_S)
            try:
                ck = _compile(cfg, dtype)
                open(os.path.join(outdir, cid + ".ptx"), "w").write(ck.asm.get("ptx", ""))
                if ck.asm.get("ttgir"):
                    open(os.path.join(outdir, cid + ".ttgir"), "w").write(ck.asm["ttgir"])
                hs = [hashlib.sha1(_run_seed(cfg, dtype, ck, s)).hexdigest() for s in range(seeds)]
                rec.update(ok=True, empirical_key=hashlib.sha1("".join(hs).encode()).hexdigest(), ptx=cid + ".ptx")
                signal.alarm(0)
                built += 1
            except Exception as e:  # noqa: BLE001
                signal.alarm(0)
                kind = _classify(e)
                tail = (str(e).splitlines() or ["?"])[-1][:90]
                if kind == "transient":
                    print(f"  transient {type(e).__name__} on {cid} {dtype} {cfg}: {tail} -- retry later", flush=True)
                    continue
                rec.update(ok=False, error=f"{type(e).__name__}: {tail}")
                infeasible += 1
                if kind == "poison":
                    tmp = jpath + ".tmp"
                    json.dump(rec, open(tmp, "w"))
                    os.replace(tmp, jpath)
                    print(
                        f"  POISON {type(e).__name__} on {cid} {dtype} {cfg}: {tail} -- "
                        f"marked infeasible, exiting for fresh CUDA context", flush=True)
                    os._exit(3)
            tmp = jpath + ".tmp"
            json.dump(rec, open(tmp, "w"))
            os.replace(tmp, jpath)
            if (built + infeasible) % 50 == 0:
                print(f"  {dtype}: built={built} infeasible={infeasible} skipped={skipped}", flush=True)
        print(f"DONE flash_attention/{dtype}: built={built} infeasible={infeasible} skipped={skipped}", flush=True)
    print(f"ALL_DONE_fa built={built} infeasible={infeasible} skipped={skipped}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50, help="fuzz seeds per config (keeps existing 40 valid)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    build(args.seeds, args.shard, args.nshards)


if __name__ == "__main__":
    main()
