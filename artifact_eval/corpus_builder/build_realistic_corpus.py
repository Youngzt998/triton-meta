"""Build the REALISTIC-INDUCTOR slice of the local_evaluation corpus.

This is a missing group of the corpus that build_local_eval.py does NOT cover:
the realistic torch-Inductor-style single-kernel reductions in
``bitequiv.evaluation.realistic_inductor_kernels`` (A_rms_norm_fwd, C..H,
I_bias_grad_dim0, J_epilogue_colsum_dim0, and the LC_* loop-carried kernels).
These are NOT in eval_kernels.resolve_kernels, so build_local_eval.py cannot
build them. They have their OWN compile+fuzz driver already: the ``_g1_cases``
+ ``_lc_cases`` harness at the bottom of realistic_inductor_kernels.py (the same
one D113724314 used, default 12 fuzz seeds). We REUSE it here and just emit the
standard corpus format so future checker experiments read cached artifacts
(no recompile, no re-fuzz).

Corpus layout (same as build_local_eval.py):
  corpus/<kernel>/<dtype>/<cid>.{ptx,ttgir,json}
  json = {kernel, dtype, config, size, seeds, ok, empirical_key, ptx, ...}
  empirical_key = sha1 of the concatenated per-seed output sha1 digests -- the
  SAME formula build_local_eval.py uses, so keys are comparable corpus-wide.

Resumable: one atomic JSON per config, skipped if it exists -> survives a GPU
kill / reboot (just relaunch). Timeout/oomd-safe: a wedged config trips SIGALRM
and HARD-EXITs so systemd Restart relaunches and resumes past it.

The checker is NOT run here -- run_local_eval.py reads <cid>.ptx later and
recomputes the checker key at read time.
"""
import argparse
import hashlib
import json
import os
import signal
import sys

os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
# Repo root, two levels up from artifact_eval/corpus_builder/. `bitequiv` is not installed,
# so it has to be on the path; keep this in step with PYTHONPATH in the artifact README.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# The existing driver: same case list D113724314 fuzzed (GROUP 1/1b + LC loop-carried).
from bitequiv.evaluation.realistic_inductor_kernels import _g1_cases, _lc_cases  # noqa: E402

# Where the corpus is written. Same name and same default as the `checker.corpus` step reads,
# so building it and grading it cannot end up pointed at two different directories.
CORPUS = os.environ.get("CHECKER_CORPUS") or os.path.expanduser("~/bitwise-equiv/local_evaluation/corpus")

# Same permanent-vs-transient split as build_local_eval.py: a transient error (a
# busy/OOM GPU shared with another session) is retried on the next run rather than
# recorded as an infeasible config.
_PERMANENT = ("CompilationError", "AssertionError", "ValueError", "OutOfResources", "KeyError", "TypeError")
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
    os._exit(3)  # non-zero -> systemd Restart=on-failure resumes; cgroup kill reaps the wedged process


signal.signal(signal.SIGALRM, _on_alarm)


def _dtype_of(name):
    """Realistic kernels have an intrinsic dtype encoded in the name suffix; default f32."""
    if name.endswith("_f16"):
        return "f16"
    if name.endswith("_bf16"):
        return "bf16"
    return "f32"


def _cid(name, cfg):
    return hashlib.sha1(repr((name, tuple(cfg))).encode()).hexdigest()[:16]


def _asm_of(ck):
    """PTX (+ optional TTGIR) for one config. Every realistic case returns a single CompiledKernel."""
    cks = ck if isinstance(ck, (list, tuple)) else [ck]
    return cks[0].asm.get("ptx", ""), cks[0].asm.get("ttgir")


def build(seeds):
    cases = _g1_cases() + _lc_cases()
    total = sum(len(c[1]) for c in cases)
    print(f"realistic corpus: {len(cases)} kernels, {total} configs, seeds={seeds}", flush=True)
    for name, cfgs, compile_fn, run_fn, label_fn, _nondet, note in cases:
        dt = _dtype_of(name)
        outdir = os.path.join(CORPUS, name, dt)
        os.makedirs(outdir, exist_ok=True)
        built = infeasible = skipped = 0
        for cfg in cfgs:
            cid = _cid(name, cfg)
            jpath = os.path.join(outdir, cid + ".json")
            if os.path.exists(jpath):
                skipped += 1
                continue
            rec = {
                "kernel": name, "dtype": dt, "config": {"label": label_fn(cfg), "cfg": list(cfg)}, "size": None,
                "seeds": seeds, "note": note
            }
            _current["jpath"], _current["rec"] = jpath, rec
            signal.alarm(_TIMEOUT_S)
            try:
                ck = compile_fn(cfg)
                ptx, ttgir = _asm_of(ck)
                open(os.path.join(outdir, cid + ".ptx"), "w").write(ptx)
                if ttgir:
                    open(os.path.join(outdir, cid + ".ttgir"), "w").write(ttgir)
                hs = [hashlib.sha1(run_fn(cfg, s)).hexdigest() for s in range(seeds)]
                rec.update(ok=True, empirical_key=hashlib.sha1("".join(hs).encode()).hexdigest(), ptx=cid + ".ptx")
                signal.alarm(0)
                built += 1
            except Exception as e:  # noqa: BLE001
                signal.alarm(0)
                nm = type(e).__name__
                if nm not in _PERMANENT:
                    print(f"  transient {nm} on {name}/{cid}: {str(e).splitlines()[-1][:60]} -- retry later",
                          flush=True)
                    continue
                rec.update(ok=False, error=f"{nm}: {str(e).splitlines()[-1][:90]}")
                infeasible += 1
            tmp = jpath + ".tmp"
            json.dump(rec, open(tmp, "w"))
            os.replace(tmp, jpath)
        print(f"DONE {name}/{dt}: built={built} infeasible={infeasible} skipped={skipped}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12, help="fuzz seeds per config (D113724314 used 12)")
    args = ap.parse_args()
    build(args.seeds)
    print("ALL_DONE_realistic", flush=True)


if __name__ == "__main__":
    main()
