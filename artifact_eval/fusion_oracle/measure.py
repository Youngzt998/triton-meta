"""Timing that survives the clock ramp.

`artifact.graph_ms` is the primitive -- capture once, replay with L2 flushed, take the median --
and it is right. What it cannot see is that this GPU idles at 120 MHz and takes seconds of load to
reach 2070 MHz. A search that compiles one config, times it, compiles the next, times it, spends
most of its wall clock compiling on the host with the GPU asleep, so each config is timed at
whatever clock the ramp happened to be at. Measured: the same config came out 0.0294 ms in a
326-second run and 0.0497 ms in an 8-second run.

Two fixes, both needed:

  warm   spin a memory-bound kernel until the clock stops rising, before timing anything.
  rounds time every candidate in round-robin and keep each one's best. Drift then has to hit one
         config in every round to fool the ranking, instead of hitting it once.
"""
from __future__ import annotations

import time

from artifact import graph_ms


def warm(torch, seconds=6.0):
    """Pull the SM clock up and report where it landed."""
    x = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            x.mul_(1.0000001).add_(1e-8)
        torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()
    try:
        return torch.cuda.clock_rate()  # MHz, if this torch has it
    except Exception:  # noqa: BLE001
        return None


def best_ms(torch, fns, flush, rounds=3, reps=15):
    """`fns` is a dict name -> callable. Returns name -> best median over `rounds` round-robins."""
    out = {k: None for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            t = graph_ms(torch, fn, flush, reps=reps)
            if t is not None and (out[k] is None or t < out[k]):
                out[k] = t
    return out
