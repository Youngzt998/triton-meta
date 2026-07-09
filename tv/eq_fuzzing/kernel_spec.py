"""The description of one kernel under test.

A kernel-set file (see kernels/basic.py) exposes a dict ``KERNELS`` mapping a
name to a ``KernelSpec``. The runner and fuzzer use the spec to build the root
IR, generate random inputs, launch, and compare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class KernelSpec:
    name: str
    fn: Any                       # the @triton.jit function
    # arg name -> triton type string, for NON-constexpr args, in call order.
    signature: Dict[str, str]
    # constexpr arg name -> value (baked into the IR).
    constexprs: Dict[str, Any]
    # (generator, device) -> {arg_name: tensor-or-scalar}; must allocate outputs.
    gen_inputs: Callable[[Any, str], Dict[str, Any]]
    # inputs dict -> launch grid tuple.
    grid: Callable[[Dict[str, Any]], Tuple[int, ...]]
    # arg names whose buffers are compared after launch.
    out_args: List[str]
    num_warps: int = 4
