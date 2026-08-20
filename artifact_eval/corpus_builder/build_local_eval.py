"""Build the local_evaluation corpus: per config, compile PTX/TTGIR + run the empirical fuzzer once,
save to disk so checker experiments read cached artifacts (no recompile, no re-fuzz).

Resumable: one atomic JSON per config, skipped if it exists -> survives a GPU kill / server reboot
(just relaunch). Shardable: --shard i --nshards n splits configs (by hash) across parallel builders.
--dtype all loops the kernel's valid dtypes.

Config space per kernel:
  * GEMM family (registry below): a rich hand-authored axis cross-product (bit-FREE axes swept,
    bit-RELEVANT axes fixed -- input_precision / fp_fusion held constant).
  * everything else (reductions): the eval framework's own max_config_space() for that kernel+dtype.

Corpus layout:  corpus/<kernel>/<dtype>/<cid>.{ptx,ttgir,json}
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

from bitequiv.evaluation import evaluate as E  # noqa: E402
from bitequiv.evaluation.eval_kernels import resolve_kernels  # noqa: E402

# Where the corpus is written. Same name and same default as the `checker.corpus` step reads,
# so building it and grading it cannot end up pointed at two different directories.
CORPUS = os.environ.get("CHECKER_CORPUS") or os.path.expanduser("~/bitwise-equiv/local_evaluation/corpus")

# Rich GEMM layout sweep (bit-FREE): power-of-2 num_warps (1/2/16 -> v2 mma.sync, 4/8 -> v5 tcgen05);
# block_m {8,16,32}=v2 + {64,128,256}=v5; num_ctas incl 4; num_stages incl 3.
_LAYOUT = {
    "num_warps": [1, 2, 4, 8, 16], "gemm_block_m": [8, 16, 32, 64, 128, 256], "gemm_block_n": [64, 128, 256],
    "gemm_block_k": [16, 32, 64, 128], "gemm_num_ctas": [1, 2, 4], "num_stages": [1, 2, 3]
}
# Smaller layout for the fusion variants (still spans v2/v5 + num_warps).
_VARIANT = {
    "num_warps": [1, 2, 4, 8], "gemm_block_m": [32, 64, 128], "gemm_block_n": [64, 128], "gemm_block_k": [32, 64],
    "gemm_num_ctas": [1, 2], "num_stages": [1, 2]
}
# GEMM+reduction epilogue: the reduction-order recovery is the point -> sweep reduction_ordering.
_REDUCE_EPI = {
    "num_warps": [1, 2, 4, 8, 16], "gemm_block_m": [32, 64, 128], "gemm_block_n": [64, 128, 256], "reduction_ordering":
    ["inner_tree", "unordered"], "num_stages": [1, 2, 3]
}

AXES = {
    "gemm": _LAYOUT,
    "gemm_bias_relu_fp_fusion": _VARIANT,
    "gemm_kgroup": {**_VARIANT, "gemm_num_splits": [1, 2, 3, 4, 6, 8]},
    "gemm_reduce_sum": _REDUCE_EPI,
    "gemm_softmax": _REDUCE_EPI,
}
# bit-RELEVANT axes held constant for the GEMM family (so they are not put in one equivalence group).
COMMON_FIXED = {"input_precision": "ieee", "enable_fp_fusion": True}

_PERMANENT = ("CompilationError", "AssertionError", "ValueError", "OutOfResources", "KeyError", "TypeError")

# Per-config wall-clock cap. A real compile+fuzz at 256^3 is a few seconds; a pathological config that
# wedges ptxas (observed: 26 min, 102% CPU, no progress) blows past this. On timeout we mark the config
# skip-on-resume and HARD-EXIT so systemd Restart relaunches (its cgroup cleanup reaps the wedged ptxas)
# and the build resumes past the bad config -> the corpus always makes progress, never stalls on one config.
_TIMEOUT_S = 180
_current = {"jpath": None, "rec": None}


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
    os._exit(3)  # non-zero -> systemd Restart=on-failure resumes; cgroup kill reaps the wedged ptxas


signal.signal(signal.SIGALRM, _on_alarm)


def build_configs(spec, kernel):
    """The config set to build for (spec, kernel). GEMM family -> rich axis cross-product; else the
    eval's own max_config_space()."""
    if kernel in AXES:
        base = spec.max_config_space()[0]
        keys = list(AXES[kernel])
        out = []
        for vals in itertools.product(*[AXES[kernel][k] for k in keys]):
            over = dict(zip(keys, vals))
            over.update(COMMON_FIXED)
            out.append(base._replace(**over))
        return out
    return list(spec.max_config_space())


def cid_of(config):
    return hashlib.sha1(repr(tuple(sorted(config._asdict().items()))).encode()).hexdigest()[:16]


def _slp_hang(config):
    """f16 GEMM configs whose per-thread output tile is huge wedge LLVM's SLP vectorizer in-process
    (llvm.cc optimize_module, ~26 min, 102% CPU, UNINTERRUPTIBLE by SIGALRM — a C++ spin). See
    ~/gemm_compile_hang. Trigger = block_m*block_n/(num_warps*32) >= 1024 on f16 (bf16/f32/fp8 only
    slow, ~<=78s, so they are left to finish). Skip these (mark infeasible) — they are un-cacheable."""
    bm, bn = config.gemm_block_m, config.gemm_block_n
    if config.dtype != "f16" or bm is None or bn is None:
        return False
    return (bm * bn) / ((config.num_warps or 1) * 32) >= 1024


def build_one(kernel, dtype, shard, nshards, seeds, size):
    specs = E._materialize(resolve_kernels(kernel), dtype)
    if not specs:
        print(f"[{kernel}/{dtype}] no spec (dtype not valid) -- skip", flush=True)
        return
    spec = specs[0]
    sz = tuple(size) if size else tuple(getattr(spec, "precision_size", (256, 256, 256)))
    outdir = os.path.join(CORPUS, kernel, dtype)
    os.makedirs(outdir, exist_ok=True)
    cfgs = build_configs(spec, kernel)
    mine = [c for c in cfgs if int(cid_of(c), 16) % nshards == shard]
    print(f"[{kernel}/{dtype} shard {shard}/{nshards}] {len(mine)}/{len(cfgs)} configs seeds={seeds} size={sz}",
          flush=True)
    built = infeasible = skipped = 0
    for c in mine:
        cid = cid_of(c)
        jpath = os.path.join(outdir, cid + ".json")
        if os.path.exists(jpath):
            # Only skip a config already built at the SAME seed count. empirical_key hashes the
            # per-seed digests, so it is seed-COUNT-dependent; a stale json from an earlier build with a
            # different --seeds gives a key that is not comparable to the rest of the corpus (it shows up
            # as a phantom over-merge). If the cached seed count differs, rebuild it.
            try:
                cached_seeds = json.load(open(jpath)).get("seeds")
            except Exception:
                cached_seeds = None
            if cached_seeds == seeds:
                skipped += 1
                continue
            os.remove(jpath)
        if _slp_hang(c):  # un-cacheable: wedges LLVM SLP for ~26 min, SIGALRM can't stop a C++ spin
            rec = {
                "kernel": kernel, "dtype": dtype, "config": c._asdict(), "size": list(sz), "seeds": seeds, "ok": False,
                "error": "SLP-vectorizer hang (f16 elems/thread>=1024); see ~/gemm_compile_hang"
            }
            json.dump(rec, open(jpath + ".tmp", "w"))
            os.replace(jpath + ".tmp", jpath)
            infeasible += 1
            continue
        rec = {"kernel": kernel, "dtype": dtype, "config": c._asdict(), "size": list(sz), "seeds": seeds}
        _current["jpath"], _current["rec"] = jpath, rec
        signal.alarm(_TIMEOUT_S)  # _on_alarm marks skip + hard-exits if this config wedges compile/run
        try:
            ck = spec.compile(c, sz)
            open(os.path.join(outdir, cid + ".ptx"), "w").write(ck.asm.get("ptx", ""))
            if ck.asm.get("ttgir"):
                open(os.path.join(outdir, cid + ".ttgir"), "w").write(ck.asm["ttgir"])
            hs = [hashlib.sha1(spec.run(c, ck, s, sz)).hexdigest() for s in range(seeds)]
            rec.update(ok=True, empirical_key=hashlib.sha1("".join(hs).encode()).hexdigest(), ptx=cid + ".ptx")
            signal.alarm(0)
            built += 1
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            name = type(e).__name__
            if name not in _PERMANENT:
                print(f"  transient {name} on {cid}: {str(e).splitlines()[-1][:55]} -- retry later", flush=True)
                continue
            rec.update(ok=False, error=f"{name}: {str(e).splitlines()[-1][:90]}")
            infeasible += 1
        tmp = jpath + ".tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, jpath)
        if (built + infeasible) % 50 == 0:
            print(f"  {kernel}/{dtype}: built={built} infeasible={infeasible} skipped={skipped}", flush=True)
    print(f"DONE {kernel}/{dtype} shard {shard}: built={built} infeasible={infeasible} skipped={skipped}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--dtype", default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--size", type=int, default=0, help="cube side; 0 = kernel's precision_size")
    args = ap.parse_args()
    size = (args.size, args.size, args.size) if args.size else None
    if args.dtype == "all":
        specs = resolve_kernels(args.kernel)
        spec0 = specs[0] if isinstance(specs, (list, tuple)) else specs
        dtypes = list(getattr(spec0, "valid_dtypes", None) or ("f32", ))
    else:
        dtypes = [args.dtype]
    for dt in dtypes:
        build_one(args.kernel, dt, args.shard, args.nshards, args.seeds, size)


if __name__ == "__main__":
    main()
