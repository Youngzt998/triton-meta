"""Experiment runner for compiler-equivalence fuzzing.

This file only *describes how to start* a run. The equivalence logic lives in
``fuzzer.py``; the compile plumbing lives in ``compile_utils.py``.

Example:

    python -m eq_fuzzing.runner \
        --level ttgir --mode candidate-vs-reference \
        --kernels eq_fuzzing/kernels/basic.py \
        --passes eq_fuzzing/passes/ttgir_opt.txt \
        -R 1000 --target cuda:90 \
        --checkpoint eq_fuzzing/checkpoints/run1.json

Pause with Ctrl-C; a checkpoint is written. Resume with the same command plus
``--resume``. On a bitwise mismatch the run stops (unless ``--keep-going``) and
saves everything needed to inspect it under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from eq_fuzzing import compile_utils as cu
from eq_fuzzing.fuzzer import EquivalenceFuzzer

CHECKPOINT_EVERY = 25

# All relative paths on the command line are resolved against the triton repo
# root (this file lives at <repo>/tv/eq_fuzzing/), so runs work from any dir.
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


# --------------------------------------------------------------------------- #
# loading kernels & passes
# --------------------------------------------------------------------------- #
def load_kernels(path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("eq_fuzzing._kernelset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "KERNELS"):
        raise AttributeError(f"{path} must define a top-level KERNELS dict")
    return mod.KERNELS


def read_passes(path: Optional[str]) -> List[str]:
    if not path:
        return []
    flags = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            flags.append(line)
    return flags


def parse_triton_opts(s: Optional[str]) -> Dict[str, Any]:
    """Parse ``"num_warps=8,num_stages=3"`` into a typed options dict."""
    opts: Dict[str, Any] = {}
    if not s:
        return opts
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        k, v = k.strip(), v.strip()
        try:
            opts[k] = int(v)
        except ValueError:
            opts[k] = v
    return opts


def _sanitize(flag: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", flag.lstrip("-"))[:60]


# --------------------------------------------------------------------------- #
# task descriptors
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    spec_name: str
    task_id: str
    # For IR levels: the extra triton-opt flags for each side.
    ref_flags: Optional[List[str]] = None
    cand_flags: Optional[List[str]] = None
    # For triton level: the options dict for each side.
    ref_opts: Optional[Dict[str, Any]] = None
    cand_opts: Optional[Dict[str, Any]] = None

    @property
    def key(self) -> str:
        return f"{self.spec_name}::{self.task_id}"


def build_tasks(spec_name: str, level: str, mode: str,
                passes_a: List[str], passes_b: List[str],
                opts_a: Dict[str, Any], opts_b: Dict[str, Any]) -> List[Task]:
    if level == "triton":
        if mode == "candidate-vs-reference":
            return [Task(spec_name, "cand_vs_ref", ref_opts={}, cand_opts=opts_a)]
        if mode == "pairwise":
            return [Task(spec_name, "pairwise", ref_opts=opts_a, cand_opts=opts_b)]
        raise ValueError(f"mode {mode!r} is not supported at the triton level")

    # IR levels (ttir / ttgir)
    if mode == "candidate-vs-reference":
        return [Task(spec_name, "cand_vs_ref", ref_flags=[], cand_flags=passes_a)]
    if mode == "pairwise":
        return [Task(spec_name, "pairwise", ref_flags=passes_a, cand_flags=passes_b)]
    if mode == "ablation":
        tasks = []
        for i, p in enumerate(passes_a):
            minus = passes_a[:i] + passes_a[i + 1:]
            tasks.append(Task(spec_name, f"ablate_{i}_{_sanitize(p)}",
                              ref_flags=passes_a, cand_flags=minus))
        return tasks
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# producing the compile-ready IR for one side
# --------------------------------------------------------------------------- #
def produce_side_ttgir(root_ext: str, root_text: str, level: str,
                       extra_flags: List[str], target, num_warps: int) -> str:
    """Return compile-ready TTGIR text for one side of a comparison."""
    if level == "ttgir":
        if not extra_flags:
            return root_text  # unoptimized root, straight-lowered
        return cu.run_triton_opt(root_text, "ttgir", extra_flags)
    # level == "ttir": always end in convert-triton-to-tritongpu -> ttgir
    flags = list(extra_flags)
    if not any("convert-triton-to-tritongpu" in f for f in flags):
        flags = flags + [cu.convert_flag(target, num_warps)]
    return cu.run_triton_opt(root_text, "ttir", flags)


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"tasks": {}}


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def save_artifacts(out_dir: Path, task: Task, level: str, ref_text: str,
                   cand_text: str, result, meta: Dict[str, Any]) -> Path:
    d = out_dir / task.key.replace("::", "__")
    d.mkdir(parents=True, exist_ok=True)
    if ref_text is not None:
        (d / "ref.ttgir").write_text(ref_text)
    if cand_text is not None:
        (d / "cand.ttgir").write_text(cand_text)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    to_cpu = lambda dd: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in dd.items()}
    torch.save(to_cpu(result.fail_input), d / "input.pt")
    out_ref, out_cand = result.fail_outputs
    torch.save(to_cpu(out_ref), d / "out_ref.pt")
    torch.save(to_cpu(out_cand), d / "out_cand.pt")
    return d


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Triton compiler equivalence fuzzer")
    ap.add_argument("--level", choices=["triton", "ttir", "ttgir"], required=True)
    ap.add_argument("--mode",
                    choices=["candidate-vs-reference", "ablation", "pairwise"],
                    default="candidate-vs-reference")
    ap.add_argument("--kernels", default="tv/eq_fuzzing/kernels/basic.py",
                    help="python file exposing KERNELS (relative to triton root)")
    ap.add_argument("--passes", help="txt file of triton-opt flags (candidate / A)")
    ap.add_argument("--passes-b", help="txt file of triton-opt flags (B, pairwise)")
    ap.add_argument("--triton-opts", help='e.g. "num_warps=8,num_stages=3" (triton level)')
    ap.add_argument("--triton-opts-b", help="second option set (triton level, pairwise)")
    ap.add_argument("-R", type=int, default=1000, help="launches to declare equivalence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="cuda:90")
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--only", help="comma-separated subset of kernel names")
    ap.add_argument("--checkpoint", default="tv/eq_fuzzing/checkpoints/run.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first mismatch")
    ap.add_argument("--out-dir", default="tv/eq_fuzzing/inequiv")
    args = ap.parse_args(argv)

    # resolve every path against the triton repo root (unless already absolute)
    args.kernels = resolve_path(args.kernels)
    args.passes = resolve_path(args.passes)
    args.passes_b = resolve_path(args.passes_b)
    args.checkpoint = resolve_path(args.checkpoint)
    args.out_dir = resolve_path(args.out_dir)

    kernels = load_kernels(args.kernels)
    names = list(kernels.keys())
    if args.only:
        wanted = [n.strip() for n in args.only.split(",")]
        names = [n for n in names if n in wanted]

    passes_a = read_passes(args.passes)
    passes_b = read_passes(args.passes_b)
    opts_a = parse_triton_opts(args.triton_opts)
    opts_b = parse_triton_opts(args.triton_opts_b)

    # ordered task list (deterministic; resume overlays status)
    tasks: List[Task] = []
    for n in names:
        tasks.extend(build_tasks(n, args.level, args.mode, passes_a, passes_b, opts_a, opts_b))

    ckpt_path = Path(args.checkpoint)
    state = load_checkpoint(ckpt_path) if args.resume else {"tasks": {}}
    state.setdefault("tasks", {})
    state["config"] = {
        "level": args.level, "mode": args.mode, "seed": args.seed, "R": args.R,
        "target": args.target, "num_warps": args.num_warps,
        "kernels": args.kernels, "passes": args.passes, "passes_b": args.passes_b,
    }

    fuzzer = EquivalenceFuzzer(R=args.R, seed=args.seed, target=args.target)
    target = cu.parse_target(args.target)
    root_cache: Dict[str, Tuple[str, str]] = {}
    out_dir = Path(args.out_dir)

    def get_root(spec) -> Tuple[str, str]:
        if spec.name not in root_cache:
            root_cache[spec.name] = cu.make_root_ir(spec, args.level, target, args.num_warps)
        return root_cache[spec.name]

    print(f"[eq_fuzzing] level={args.level} mode={args.mode} R={args.R} "
          f"tasks={len(tasks)} kernels={names}")

    stopped = False
    try:
        for task in tasks:
            spec = kernels[task.spec_name]
            tstate = state["tasks"].setdefault(task.key, {"status": "pending", "iters_done": 0})
            if tstate["status"] in ("passed", "failed"):
                print(f"  skip {task.key} ({tstate['status']})")
                continue

            # build the two sides
            ref_text = cand_text = None
            if args.level == "triton":
                kernel1 = (spec.fn, task.ref_opts)
                kernel2 = (spec.fn, task.cand_opts)
                compile_level = "triton"
            else:
                root_ext, root_text = get_root(spec)
                ref_text = produce_side_ttgir(root_ext, root_text, args.level,
                                              task.ref_flags, target, args.num_warps)
                cand_text = produce_side_ttgir(root_ext, root_text, args.level,
                                               task.cand_flags, target, args.num_warps)
                kernel1, kernel2 = ref_text, cand_text
                compile_level = "ttgir"

            def on_iter(it, _ts=tstate):
                _ts["iters_done"] = it + 1
                if (it + 1) % CHECKPOINT_EVERY == 0:
                    save_checkpoint(ckpt_path, state)

            print(f"  run  {task.key} (from iter {tstate['iters_done']}) ...", flush=True)
            result = fuzzer.check_eq(kernel1, kernel2, compile_level, spec,
                                     start_iter=tstate["iters_done"], on_iter=on_iter)

            if result.equal:
                tstate["status"] = "passed"
                tstate["iters_done"] = result.iters_run
                print(f"       {result}")
            else:
                tstate["status"] = "failed"
                tstate["iters_done"] = result.iters_run
                meta = {"task": task.key, "level": args.level, "mode": args.mode,
                        "ref_flags": task.ref_flags, "cand_flags": task.cand_flags,
                        "ref_opts": task.ref_opts, "cand_opts": task.cand_opts,
                        "seed": args.seed, "first_fail_iter": result.first_fail_iter}
                d = save_artifacts(out_dir, task, args.level, ref_text, cand_text, result, meta)
                print(f"       {result}  -> artifacts in {d}")
                save_checkpoint(ckpt_path, state)
                if not args.keep_going:
                    stopped = True
                    break
            save_checkpoint(ckpt_path, state)
    except KeyboardInterrupt:
        save_checkpoint(ckpt_path, state)
        print(f"\n[eq_fuzzing] interrupted; checkpoint saved to {ckpt_path}. "
              f"Re-run with --resume to continue.")
        return 130

    save_checkpoint(ckpt_path, state)
    passed = sum(1 for t in state["tasks"].values() if t["status"] == "passed")
    failed = sum(1 for t in state["tasks"].values() if t["status"] == "failed")
    print(f"[eq_fuzzing] done. passed={passed} failed={failed} "
          f"{'(stopped at first mismatch)' if stopped else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
