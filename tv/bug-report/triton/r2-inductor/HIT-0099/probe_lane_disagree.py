#!/usr/bin/env python
"""HIT-0099 probe: the tt.reduce result is NOT the same in every lane.

Both arms of the report are re-run with one edit: the value stored into the
i32 output is replaced by the *broadcast rstd* of the row
(`rsqrt(m2/2048 + 1e-5)`), bit-cast to i32.  rstd is one number per row, so
every column of a row must print the same value.

    source /home/youngzt/fuzz/r2-inductor/env.sh
    python probe_lane_disagree.py 1009

Observed on 2026-08-19 (row 99, out_ptr3 has ptr_off=5, so slot position
p = kernel column r0 + 5):

    reference : NaN in every column of the row
    candidate : 0.0 at r0 = 3, 7, 11, 15, 19, 23 and NaN everywhere else

0.0 is rsqrt(+inf); NaN is rsqrt(NaN).  So lanes that are supposed to hold
copies of one reduce result hold +inf in some and NaN in others.
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
HOME = Path('/home/youngzt/fuzz/r2-inductor')
sys.path.insert(0, str(HOME))
os.environ.setdefault('TRITON_ALLOW_NON_CONSTEXPR_GLOBALS', '1')
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/hit0099-cache')
import torch
from fz import specs as specmod, worker as W

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1009
d = json.loads((HERE / 'repro.json').read_text())
spec = specmod.spec_from_json(d['spec'])
vr = W.get_variant((HERE / 'probe-rstd-ref.ttgir').read_text(), 'ref', spec.cfg)
vc = W.get_variant((HERE / 'probe-rstd-cand.ttgir').read_text(), 'cand', spec.cfg)
pool, dts = W.get_pool(spec)
args = W.build_args(spec, pool)
pin = set(W.pin_zero_slots(spec, pool.slot_bytes // 8))
pool.prepare(dts, 'wide', seed, pin)
pool.reset(); W.launch(vr, args, spec.grid); torch.cuda.synchronize(); A = pool.buf.clone()
pool.reset(); W.launch(vc, args, spec.grid); torch.cuda.synchronize(); B = pool.buf.clone()

def slot(buf, i):
    o = pool.slot_off(i)
    return buf[o:o + pool.slot_bytes].view(torch.float32)

fa, fb = slot(A, 4), slot(B, 4)
diff = (fa.view(torch.int32) != fb.view(torch.int32))
print('differing elements:', int(diff.sum()))
rows = sorted({int(j) // 2048 for j in diff.nonzero().flatten()})
print('rows touched:', rows)
PTR_OFF = 5                      # spec.json: ptr_off[4] == 5 elements
for x0 in rows[:2]:
    print(f'--- row {x0}: broadcast rstd, one value per column ---')
    for r0 in range(25):
        p = x0 * 2048 + PTR_OFF + r0
        print(f'   r0={r0:2d}  ref={float(fa[p])!r:>8}  cand={float(fb[p])!r}')
