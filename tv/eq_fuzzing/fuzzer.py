"""The equivalence fuzzer.

The core interface is:

    fuzzer = EquivalenceFuzzer(R=1000, target="cuda:90")
    result = fuzzer.check_eq(kernel1, kernel2, level, spec)

``level`` is one of ``"triton"``, ``"ttir"``, ``"ttgir"``.

  * ``triton``: ``kernel1`` / ``kernel2`` are ``(jit_fn_ignored, options_dict)``
    -- the kernel comes from ``spec.fn``; only the compile options differ.
  * ``ttir`` / ``ttgir``: ``kernel1`` / ``kernel2`` are IR strings (already
    produced by the runner, e.g. via ``triton-opt``).

``check_eq`` launches both variants on the same random inputs up to ``R`` times
and compares outputs bitwise. All ``R`` matches => "empirically equivalent".
First mismatch => stop and report it, with the failing input saved.

This class is meant to be the *improvable* part: today inputs are plain random
draws; a smarter input strategy can subclass and override ``gen_inputs``.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from . import compile_utils as cu


@dataclass
class EqResult:
    equal: bool
    level: str
    iters_run: int
    first_fail_iter: Optional[int] = None
    fail_input: Optional[Dict[str, Any]] = None
    fail_outputs: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None

    def __str__(self) -> str:
        if self.equal:
            return f"EQUAL ({self.level}, {self.iters_run} launches)"
        return f"NOT-EQUAL ({self.level}, first mismatch at iter {self.first_fail_iter})"


class EquivalenceFuzzer:

    def __init__(self, R: int, seed: int = 0, target: str = "cuda:90",
                 device: str = "cuda", bitwise: bool = True, nan_equal: bool = True):
        self.R = R
        self.seed = seed
        self.target_str = target
        self.target = cu.parse_target(target)
        self.device = device
        self.bitwise = bitwise
        self.nan_equal = nan_equal

    # ----- input generation (override this for a smarter fuzzer) ----------- #
    def gen_inputs(self, spec, iteration: int) -> Dict[str, Any]:
        gen = torch.Generator(device=self.device).manual_seed(self.seed + iteration)
        return spec.gen_inputs(gen, self.device)

    # ----- comparison ----------------------------------------------------- #
    def _compare(self, a: Any, b: Any) -> bool:
        if not torch.is_tensor(a):
            return a == b
        if self.bitwise:
            return cu.tensors_bitwise_equal(a, b, nan_equal=self.nan_equal)
        return torch.allclose(a, b)

    # ----- variant construction ------------------------------------------- #
    def _build_variant(self, kernel, level: str, spec, workdir: Path, label: str) -> cu.CompiledVariant:
        if level == "triton":
            _, options = kernel
            return cu.build_variant_from_triton(spec, options, self.target, label)
        # ttir / ttgir: kernel is an IR string
        return cu.build_variant_from_ir(kernel, level, spec, self.target, workdir, label)

    # ----- the main loop -------------------------------------------------- #
    def check_eq(self, kernel1, kernel2, level: str, spec, *, start_iter: int = 0,
                 on_iter: Optional[Callable[[int], None]] = None) -> EqResult:
        with tempfile.TemporaryDirectory() as d:
            workdir = Path(d)
            var1 = self._build_variant(kernel1, level, spec, workdir, "ref")
            var2 = self._build_variant(kernel2, level, spec, workdir, "cand")

            for it in range(start_iter, self.R):
                inputs = self.gen_inputs(spec, it)
                grid = spec.grid(inputs)
                in1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
                in2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
                var1.run(in1, grid)
                var2.run(in2, grid)
                torch.cuda.synchronize()

                ok = all(self._compare(in1[a], in2[a]) for a in spec.out_args)
                if not ok:
                    return EqResult(
                        equal=False, level=level, iters_run=it + 1, first_fail_iter=it,
                        fail_input=inputs,
                        fail_outputs=({a: in1[a] for a in spec.out_args},
                                      {a: in2[a] for a in spec.out_args}))
                if on_iter is not None:
                    on_iter(it)

        return EqResult(equal=True, level=level, iters_run=self.R)
