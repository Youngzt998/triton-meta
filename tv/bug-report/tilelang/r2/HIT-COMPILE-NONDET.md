# HIT-COMPILE-NONDET: TileLang compiles the same input to different CUDA on different runs — at least four passes, at least two independent causes

Verdict: **CONFIRMED**
Arch:    generic (the passes are target-neutral; all measurements here are
         H100 sm_90a, so the *rate* is only measured on one target)
Line:    tilelang r2 (TileLang 0.1.12, pypi wheel, H100 sm_90a, nvcc 12.8.93)
Culprit: **four** passes, not one:
         `MergeSharedMemoryAllocations`, `ThreadSync.shared.dyn`,
         `LegalizeSafeMemoryAccess`, `Simplify` (the `#3` instance).
Scope:   this report owns the **scope**. `r2/HIT-0043` owns the one fully
         traced instance and is not repeated here — read it first for the
         `ThreadSync` mechanism and for the case where the lottery changes the
         numerical answer.

Summary: `tilelang.lower` is not a function of its input. Handing the compiler
one unchanged `IRModule` and compiling it repeatedly yields different CUDA
source text, drawn at random, run after run in the same process. `HIT-0043`
recorded one path (`tl.ThreadSync("shared.dyn")`, 38/12 over 50 applications on
one FA kernel). It is much wider than that. Measured here: **7 of the 14
generated kernel families and 4 of 150 sampled captured kernels** produce more
than one CUDA source for one input; the two worst produce **33 and 29 distinct
sources in 48 lowerings** (independently re-measured at 15/16 and 14/16, and by
the reviewer at 15/16 and 12/16). Four distinct passes are responsible, and
they do **not** share one cause: three of them fail through the arithmetic
analyzer (a fact is proved on some runs and not on others), while the fourth,
`MergeSharedMemoryAllocations`, uses no solver at all and instead iterates
`std::unordered_map`s **keyed on node addresses**, so the order in which it
emits the merged buffer's aliases changes from run to run.

The two causes differ in how much they can hurt. Cause A's `ThreadSync` half
changes the program: on `HIT-0043`'s point one draw drops a barrier at a
genuine cross-thread write→read boundary and the answers differ. Cause B, on
every point where it could be re-measured, does **not** change the program: the
byte offsets it assigns are identical in every draw and the whole difference is
*which name* a scratch pointer is given, among several names bound to the same
address (§1.3). So cause B breaks reproducible builds and output-hash caches
and nothing else; cause A can break the answer. You cannot tell which kind you
have without looking.

Labels: compile-time-nondeterminism, reproducible-builds, multiple-root-causes,
scope-report

---

## 1. Root cause in the source

There is no single root cause. There are at least two, and the four passes
split cleanly between them.

### 1.1 What was measured, and how the passes were pinned

The stock CUDA pipeline is re-expressed as a plain list of steps in
`tlfz/steps.py`, so the `IRModule` can be hashed immediately before and
immediately after every single pass. Lower the whole pipeline `R` times and,
for each pass, build the map

```
input IR hash  ->  { output IR hashes }
```

A pass is nondeterministic **exactly when one input hash maps to more than one
output hash**. Passes downstream of a diverging pass legitimately see different
inputs, so keying on the input is what stops them being blamed. This is
`HIT-0043` §1.1's method run over every pass instead of three; run on
`HIT-0043`'s own instance it returns `ThreadSync.shared.dyn` and nothing else,
which is the check that the method is right.

**The reviewer re-ran that control from scratch** (2026-08-19, §2b's script with
`kid` swapped for `HIT-0043`'s spec, 16 lowerings): one nondeterministic
(pass, input) pair and no other, `55:ThreadSync.shared.dyn`, input `6549ecf3` →
`adbdd531` ×14, `d960d9ae` ×2. Those are the same three hashes `HIT-0043` §1.1
printed, from a separate process on a separate day. The attribution method is
sound and everything below rests on a control that holds.

19 unstable points were probed this way at R=12–16:

| culprit pass | points | pipeline slot |
|---|---|---|
| `ThreadSync.shared.dyn` | 11 | 55 |
| `LegalizeSafeMemoryAccess` | 5 | 19 |
| `MergeSharedMemoryAllocations` | 4 | 51–52 |
| `Simplify#3` | 1 | 21 |

Two points have more than one culprit at once — `gen_rr_53f17794e18f4b` has
**three** (`LegalizeSafeMemoryAccess`, `Simplify#3`, `ThreadSync.shared.dyn`)
and `gen_copy_68f6d2b39b00f0` has two. One point
(`gen_gemm_0b0ee91756f729`) showed no pass-level divergence at R=16 even though
it had shown two CUDA sources at n=8; its rate is simply low.

### 1.2 Cause A — an arithmetic proof that does not always succeed

`ThreadSync.shared.dyn`, `LegalizeSafeMemoryAccess` and `Simplify` all reach
`arith::Analyzer`. In this TVM fork that class carries a `z3_prover` member:

* `src/transform/thread_storage_sync.cc:353` —
  `analyzer_->z3_prover.CountSatisfyingValues(iv->var, extent)`, and again at
  `:468`; `:1685-1688` uses `analyzer.z3_prover.CanProve`.
* `src/transform/legalize_safe_memory_access.cc:167` and `:205` —
  `analyzer_->CanProve(index < shape_dim)` / `CanProve(index >= 0)`.
* `src/transform/simplify.cc:364`, `:420`, `:549` — `analyzer_->CanProve(...)`
  and `analyzer_->Simplify(...)`.

Every divergence these three produce has the same shape: **the compiler proves
a fact on some runs and fails to prove it on others**, and emits different code
accordingly.

`ThreadSync.shared.dyn` — a shared-memory barrier is emitted or omitted. On
`gen_strided_b0554a0ef12f2a`, 40 lowerings, one input `64c04905`, two outputs
28/12, and the whole difference is one line:

```
         for i in T.unroll(16):
             As_1[i*1024 + tx*4 : +4] = As_1[...] + As_1[...]
+        T.tvm_storage_sync("shared.dyn")
         for i in T.unroll(32):
             O_1[...] = As_1[i//2*1024 + tx*4 + i%2*2 : +2]
```

(On *this* kernel each thread only reads back what it wrote, so the omission is
harmless. On `HIT-0043`'s FA kernel the same defect drops a barrier between a
128-thread `stmatrix` and a one-thread `tma_store`, and the answers differ.
Same pass, same shape, different consequence.)

`LegalizeSafeMemoryAccess` — a bounds guard is added or not. On
`gen_ielem_ff479647c52c97`, 40 lowerings, one input `2ad0e996`, two outputs
16/24:

```
- Bs[...] = T.if_then_else(bx*128 + (i*32+tx)%128 < 191, B[...], T.int8(0))
+ Bs[...] = T.if_then_else(bx*128 + (i*32+tx)%128 < 191,
+              T.if_then_else(bx*128 + i%4*32 + tx < 191, B[...], T.int8(0)),
+              T.int8(0))
```

The inner guard is redundant given the outer one. On 16 runs in 40 the analyzer
proved that and left it out; on the other 24 it did not.

`Simplify#3` — a guard is folded to a shorter equivalent form or not. On
`gen_rr_53f17794e18f4b`, 10 lowerings, one input `8a588180`, two outputs 2/6:

```
- ... and i % 2 * 32 + tx < 16 and i % 2 < 1 ...
+ ... and i % 2 * 32 + tx < 16 and i % 2 * 32 + tx < 16 ...
```

With `tx` in `[0,32)`, `i%2*32 + tx < 16` already implies `i%2 == 0`, so the two
conjunctions are equivalent; sometimes the simplifier derives the short form and
sometimes it repeats the term.

**What is proved and what is inferred.** Proved: each of these three passes maps
one byte-identical input IR to two different output IRs, and the difference is
always a proof-dependent guard or barrier. Inferred: that the varying quantity
is the analyzer's answer, and that the Z3 layer is where it varies. `HIT-0043`
§1.3 already argued that for `ThreadSync` by elimination and flagged that
`CountSatisfyingValues` could not be read because `3rdparty/tvm` is not checked
out in this source copy — that is still true here
(`tilelang-src/3rdparty/tvm` is empty), so **I could not read
`arith::Analyzer::CanProve` either, and I have not proved that the other two
passes go through Z3 at all.** They may share `HIT-0043`'s exact cause or they
may have their own. I state that as unresolved rather than guessing.

### 1.3 Cause B — iteration in address order (`MergeSharedMemoryAllocations`)

This one is definitely **not** cause A: `src/transform/merge_shared_memory_allocations.cc`
contains **zero** references to `analyzer`, `CanProve` or `z3`. It is the
classic pattern instead — the pass reads containers keyed on node addresses, so
what it emits depends on where the allocator happened to put the nodes.

**What the pass actually does differently between runs.** On the captured
kernel `cap_c5c2818305b35efa` (`sparse_mla_fwd`) the pass maps **one** input
(`e30b5815`) to 13–14 different outputs in 16 applications. The reviewer dumped
the IR on both sides of the pass over 8 fresh lowerings (7 distinct outputs) and
compared them field by field. The result is narrower than this report first
claimed:

* the **byte offsets are identical in all 7 outputs** — `Q_shared` 0,
  `Q_tail_shared` 65536, `K_tail_shared` 73728, `KV_shared` 90112,
  `S_shared` 221184, five `workspace` buffers at 229376 and one at 230400;
  the merged size is the same too;
* the *only* thing that changes is which of the five equal-offset `workspace`
  aliases each `tl::AllReduce` site is written with:

```
-  ... "tl::AllReduce<tl::SumOp,256,128,0,...>::run", sumexp_i_1[i], ...workspace_4...
+  ... "tl::AllReduce<tl::SumOp,256,128,0,...>::run", sumexp_i_1[i], ...workspace_3...
-  ... "tl::AllReduce<tl::MaxOp,256,128,0,...>::run", m_i_clear_1[i], ...workspace_3...
+  ... "tl::AllReduce<tl::MaxOp,256,128,0,...>::run", m_i_clear_1[i], ...workspace_2...
```

and in the generated CUDA those five names are five spellings of one address:

```c
void* workspace   = ((void*)((char*)buf_dyn_shmem + 229376));
void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 229376));
void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 229376));
void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 229376));
void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 229376));
void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 230400));
```

Two distinct CUDA sources for this kernel differ in **3 lines out of 497**, and
each of the three is a `workspace_i` → `workspace_j` rename between pointers
that hold the same value. The sixth workspace, the one with its own offset, is
referenced by the same site in every draw. Measured the same way on the other
two reproducible cause-B points, `cap_c84d73cfbb167409` (aliases at 58368,
one at 59392) and `cap_09d61c7623de9654`: same picture, 4 changed lines each,
all of them renames among equal-address aliases. **So on every cause-B point
that can be re-run, the different sources are the same program with the scratch
pointers renamed.** (The fourth cause-B point, `gen_rr_cedf40c305ae7c`, has no
spec recorded anywhere in the line's files, so it could not be checked.)

**Where that renaming comes from.** `MakeAliasBindings`, `:512-528`, is the step
that turns the offset table into the emitted `Bind` statements, and it is the
one place whose output order is not fixed by the program:

```cpp
    std::vector<AliasInfo> aliases;
    for (const auto &pair : buffer_byte_offsets_) {      // :514  address order
      ...
      aliases.push_back(AliasInfo{pair.first, pair.second});
    }
    std::sort(aliases.begin(), aliases.end(),
              [](const AliasInfo &lhs, const AliasInfo &rhs) {
                ... if (offsets differ) return lhs_offset->value < rhs_offset->value;
                return lhs.var->name_hint < rhs.var->name_hint;   // :526
              });
```

`buffer_byte_offsets_` is `std::unordered_map<const VarNode *, PrimExpr>`
(`:1593`) — keyed on the **address** of the buffer var, so `:514` fills the
vector in address order. The sort is then supposed to make the order canonical,
but its key is *(offset, name)* and this kernel declares **six shared buffers
that are all called `workspace`**; five of them also share the offset 229376.
For those five the comparator returns false both ways, `std::sort` is not
stable, and the tie is settled by the address order it was handed. A different
order of the five identical `Bind` statements gives each `workspace` var a
different printed suffix, which is exactly the observed diff.

The same tie sits in the two loops the file calls deterministic —
`PlanSequentialLayout` at `:556` and `PlanMemory` at `:1376-1386`:

```cpp
    // Sort allocations deterministically by name.
    std::sort(sorted_vars.begin(), sorted_vars.end(),
              [](const VarNode *a, const VarNode *b) {
                return a->name_hint < b->name_hint;
              });
```

Sorting by `name_hint` alone is not a total order once two allocations share a
name, and the vector being sorted was filled by walking `shmem_allocs_`, another
`VarNode *`-keyed `unordered_map` (`:559`, `:1380`). On this kernel that tie
covers six buffers. It does not show up in the offsets here because the five
tied buffers get the same offset whatever order they are packed in — but the
"deterministic by name" comment is only true for kernels whose shared buffers
have distinct names.

**What is proved and what is not.**

* Proved: one byte-identical input IR, 13–14 different output IRs from this one
  pass; the offsets and the merged size are the same in all of them; the whole
  difference is a permutation of equal-address alias names.
* Proved: the variation needs *fresh* node addresses. Applying the same pass
  object 50 times to the **same in-memory module** — same nodes, same addresses
  — gives **1 distinct output, 50 times**. The same experiment on
  `ThreadSync.shared.dyn` at `HIT-0043`'s point gives **2** outputs (44/6), so
  the experiment is a working detector and the single answer for
  `MergeSharedMemoryAllocations` means something. Together with "no solver in
  the file", that rules out an RNG, a timer and Z3, and leaves allocation order.
* Not proved: that `MakeAliasBindings`'s tie is the *only* address-ordered step
  that fires. The earlier "reorder kill points" loop at `:1197` also walks an
  address-keyed map (`event_map_`, `:1600`) and this report first blamed it. It
  may still vary; what the measurement shows is that it has **no visible effect
  here**, because every draw produces the same live ranges' worth of offsets. Two
  further reasons to doubt the original story: each buffer is pushed into exactly
  one event's `gen` list (single `touched` set, `:1098-1110`), so the "first
  match wins" search at `:1218` has only one match to find; and `PlanMemory`
  consumes the kill sets with `end_index[var] = std::max(...)` (`:1365`), which
  does not care about their order. Settling it needs what this report could not
  do: print the traversal order, or rebuild with the loops made ordered and check
  the nondeterminism is gone.

### 1.4 So: one cause or several?

**Several.** `MergeSharedMemoryAllocations` cannot share `HIT-0043`'s cause — it
never calls the analyzer. That is two independent causes at minimum. Whether
`LegalizeSafeMemoryAccess` and `Simplify` share `ThreadSync`'s exact Z3 defect
or only its shape is **undetermined** for the reason in §1.2, so the true count
is two, three or four. Saying "one shared cause" would be wrong; saying
"exactly four causes" would be unearned.

---

## 2. Reproduction

Both commands were run on 2026-08-19 and their output is pasted below.
`TILELANG_DISABLE_CACHE=1` is mandatory and `env.sh` sets it. Nothing here
needs a GPU beyond target detection — `tilelang.lower(...,
enable_device_compile=False)` runs the whole device pipeline and stops before
nvcc.

### (a) the same input compiles to many different CUDA sources

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
cd /home/youngzt/fuzz-tilelang-r2
python - <<'EOF'
import collections
from tlfz import core, steps as S
S.install()
for item in [{"kind": "cap", "kid": "cap_c5c2818305b35efa"},
             {"kind": "cap", "kid": "cap_c84d73cfbb167409"}]:
    k = core.load_item(item)
    c = collections.Counter(h[:8] for h in core.source_hashes(k, {}, 16))
    print(item["kid"], "->", len(c), "distinct CUDA sources in 16 lowerings", dict(c))
EOF
```

Observed:

```
cap_c5c2818305b35efa -> 15 distinct CUDA sources in 16 lowerings
cap_c84d73cfbb167409 -> 14 distinct CUDA sources in 16 lowerings
```

Re-run by the reviewer, same command, fresh process:

```
cap_c5c2818305b35efa -> 15 distinct CUDA sources in 16 lowerings
cap_c84d73cfbb167409 -> 12 distinct CUDA sources in 16 lowerings
```

### (b) which pass is not a function of its input

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
cd /home/youngzt/fuzz-tilelang-r2
python - <<'EOF'
import collections, hashlib
import tilelang
from tlfz import core, steps as S

trace = []
def traced(mod, target):                       # hash the IR around every pass
    cu = tilelang.cuda.transform
    for i, (name, p) in enumerate(S._apply_order(S.build_steps(target))):
        hi = hashlib.sha256(str(mod).encode()).hexdigest()[:8]
        if p is None:
            if S.module_has_tma(mod):
                mod = cu.FuseMBarrierArriveExpectTx()(mod)
        else:
            mod = p(mod)
        trace.append((f"{i:02d}:{name}", hi,
                      hashlib.sha256(str(mod).encode()).hexdigest()[:8]))
    return mod

S.cuda_body = traced
S.install()
k = core.load_item({"kind": "cap", "kid": "cap_c5c2818305b35efa"})
cfg, tgt = core._merged_cfg(k, core.set_variant({})), core.target()
m = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
src = collections.Counter()
for _ in range(16):
    trace.clear()
    with tilelang.transform.PassContext(opt_level=3, config=cfg), tgt:
        art = tilelang.lower(k.prim, target=tgt, enable_host_codegen=False,
                             enable_device_compile=False)
    src[hashlib.sha256(art.kernel_source.encode()).hexdigest()[:8]] += 1
    for name, hi, ho in trace:
        m[name][hi][ho] += 1
print("distinct CUDA sources:", len(src))
for name, ins in m.items():                    # one input -> two outputs = the culprit
    for hi, outs in ins.items():
        if len(outs) > 1:
            print("NONDETERMINISTIC PASS", name, "input", hi, "->", dict(outs))
EOF
```

Observed:

```
distinct CUDA sources: 14
NONDETERMINISTIC PASS 51:MergeSharedMemoryAllocations input e30b5815 ->
  {'33b21160': 1, '99ec4ad4': 2, '3a0ca09d': 1, '4f492c62': 1, '028a290f': 1,
   '87be522d': 2, 'eb84cde8': 1, '520d37f8': 1, '06191ec5': 1, '4e4209df': 1,
   '6e4dc3ce': 1, 'fa830147': 1, 'b651e6bd': 1, 'f30d73a2': 1}
```

One input IR hash. Fourteen output IR hashes. One pass. Everything before slot
51 is stable.

Reviewer's re-run of the same script: `distinct CUDA sources: 13`, and the one
line printed is `NONDETERMINISTIC PASS 51:MergeSharedMemoryAllocations input
e30b5815 -> {13 output hashes}` — the same slot, the same input hash, nothing
else in the pipeline.

Swap the `kid` for any of the ids in §5 to reproduce the other passes:
`gen_ielem_ff479647c52c97` for `LegalizeSafeMemoryAccess`,
`gen_strided_b0554a0ef12f2a` for `ThreadSync.shared.dyn`,
`gen_rr_53f17794e18f4b` for all three of cause A at once (build the item with
`{"kind": "gen", "spec": ...}` — the specs are in §5). Both were re-run at 20
lowerings and both come back with exactly one culprit and the input hash this
report quotes:

```
ielem   -> 2 distinct CUDA sources; 19:LegalizeSafeMemoryAccess  input 2ad0e996 -> {'2ad0e996': 11, 'cdd8e14b': 9}
strided -> 2 distinct CUDA sources; 55:ThreadSync.shared.dyn     input 64c04905 -> {'9f428498': 11, 'a8413e73': 9}
```

### (c) does the pass need fresh node addresses? (the cause-A / cause-B split)

Apply one pass object over and over to the **same** in-memory module, so every
IR node keeps the address it already has. Cause A does not care; cause B goes
quiet.

```bash
source /home/youngzt/fuzz-tilelang-r2/env.sh
cd /home/youngzt/fuzz-tilelang-r2
python - <<'EOF'
import collections, hashlib
import tilelang
from tlfz import core, steps as S

CASES = [("MergeSharedMemoryAllocations",
          {"kind": "cap", "kid": "cap_c5c2818305b35efa"}),
         ("ThreadSync.shared.dyn",
          {"kind": "gen", "spec": {"fam": "fa", "p": {"B": 2, "H": 4, "SQ": 385, "SKV": 127,
            "D": 32, "bM": 64, "bN": 64, "th": 128, "st": 1, "causal": True,
            "dt": "bfloat16", "pol": "Square"},
            "pc": {"tl.ptxas_register_usage_level": 10}}})]

ST = {"want": None, "cap": {}}

def traced(mod, target):
    cu = tilelang.cuda.transform
    for i, (name, p) in enumerate(S._apply_order(S.build_steps(target))):
        if name == ST["want"] and "pre" not in ST["cap"]:
            ST["cap"]["pre"], ST["cap"]["pass"] = mod, p   # freeze the input nodes
        if p is None:
            if S.module_has_tma(mod):
                mod = cu.FuseMBarrierArriveExpectTx()(mod)
        else:
            mod = p(mod)
    return mod

S.cuda_body = traced          # install() captures cuda_body once, so set it first
S.install()

for want, item in CASES:
    ST["want"] = want
    ST["cap"] = cap = {}
    k = core.load_item(item)
    cfg, tgt = core._merged_cfg(k, core.set_variant({})), core.target()
    with tilelang.transform.PassContext(opt_level=3, config=cfg), tgt:
        tilelang.lower(k.prim, target=tgt, enable_host_codegen=False,
                       enable_device_compile=False)
    pre, p = cap["pre"], cap["pass"]
    outs = collections.Counter()
    with tilelang.transform.PassContext(opt_level=3, config=cfg), tgt:
        for _ in range(50):                            # same module, same addresses
            outs[hashlib.sha256(str(p(pre)).encode()).hexdigest()[:8]] += 1
    print(want, "->", len(outs), "distinct outputs in 50 applications", dict(outs))
EOF
```

Observed (reviewer, 50 applications each; the script above, run whole):

```
MergeSharedMemoryAllocations -> 1 distinct outputs in 50 applications {'5e02d0b3': 50}
ThreadSync.shared.dyn -> 2 distinct outputs in 50 applications {'d960d9ae': 7, 'adbdd531': 43}
```

and in two earlier separate processes, one case each: `{'430f8dad': 50}` and
`{'adbdd531': 44, 'd960d9ae': 6}`. The frozen module is a fresh object in each
process, so the single hash on the `Merge` line is a different value every time
— that is the point: it is *one* value per process, 50 times.

The `ThreadSync` line is the positive control (`HIT-0043` measured 38/12 the
same way): the experiment does detect a nondeterministic pass. So the single
answer on `MergeSharedMemoryAllocations` is evidence, not blindness — that pass
needs the addresses to move before it changes its mind.

---

## 3. Where it reproduces

**Any target, in principle; measured on H100 sm_90a only.**

None of the four passes is target-gated. `MergeSharedMemoryAllocations` is a
generic shared-memory liveness/merge pass; `ThreadSync`, `Simplify` and
`LegalizeSafeMemoryAccess` are generic TIR passes. No Hopper-only construct
(wgmma, TMA, mbarrier, cluster, warp specialization) appears in the diverging
code for three of the four — the `MergeSharedMemoryAllocations` case is a
`tl::AllReduce` over a `workspace` buffer, the `LegalizeSafeMemoryAccess` case
is an `if_then_else`-guarded global load, the `Simplify` case is a loop guard.
`HIT-0043`'s specific *consequence* is Hopper-only (`stmatrix` + `tma_store`),
but that is the consequence, not the defect.

What I have **not** shown: that the rate is the same on another GPU, another
CUDA version or another build of TileLang. Everything here is one H100, TileLang
0.1.12 from the pypi wheel, nvcc 12.8.93. If cause B really is address-order,
the rate would be expected to depend on the allocator and on ASLR, so treat the
rates as this-machine numbers.

---

## 4. Plain-language explanation

A compiler is supposed to be a function: same input, same output. TileLang is
not. Compile the same kernel twice and you can get two different CUDA files.

Two separate reasons.

**First reason.** Several passes ask a solver questions like "is this index
always inside the array?" or "how many threads reach this point?". The solver
does not always give the same answer for the same question. When it answers
"yes" the compiler leaves out a bounds check or a wait; when it answers "don't
know" it puts one in. So the output flips between runs. Usually both versions
compute the same numbers. Sometimes — see `HIT-0043` — the thing that gets left
out is a wait that was actually needed, and then the two versions disagree.

**Second reason, unrelated to the first.** The pass that packs shared-memory
buffers into one block keeps its bookkeeping in hash tables whose keys are the
*memory addresses* of program nodes, so it walks them in whatever order the
memory allocator produced that run. Before it writes the buffer names out it
sorts them, which is meant to make the order fixed — but it sorts by
(offset, name), and this kernel has six scratch buffers that are all called
`workspace`, five of them at the same offset. Equal keys, so the sort leaves
them in the order it got them, which is the address order. Each run therefore
hands the six the names out differently.

The effect is smaller than it looks. The five names all point at the *same*
address, so the two CUDA files are the same program with some pointers renamed:
identical offsets, identical shared-memory size, 3 changed lines out of 497.
Nothing computes anything different. What breaks is only "same source in, same
file out".

The file even shows the author was thinking about this: two loops in it sort
"deterministically by name". The catch is that a name sort is not deterministic
when several buffers share a name, which is exactly this kernel.

---

## 5. Scope: the numbers

All lowerings use the stock pipeline with no toggles. "Distinct sources" =
distinct `sha256` of `art.kernel_source`.

### 5.1 How many kernel families

Generated families, 30 fresh (shape, config) draws each, 8 lowerings per draw:

| family | unstable / probed | | family | unstable / probed |
|---|---|---|---|---|
| bcast | 0/30 | | ielem | 0/28 |
| cast | 0/30 | | layernorm | 0/30 |
| **copy** | **1/30** | | **rr** | **1/30** |
| cumsum | 0/30 | | softmax | 0/30 |
| elem | 0/30 | | **strided** | **3/30** |
| **fa** | **1/25** | | transpose | 0/30 |
| **gemm** | **2/30** | | | |
| **gemv** | **1/30** | | **total** | **9 / 413** = 2.2% |

(420 draws attempted, 7 failed to lower at all and are excluded.)

Captured kernels: 150 of the 519 usable, 8 lowerings each → **4 unstable
(2.7%)**: `cap_09d61c7623de9654`, `cap_9e5bbec266901edb`,
`cap_c5c2818305b35efa`, `cap_c84d73cfbb167409`.

The campaign's own live gate (6 samples per job, ~2000 jobs) has independently
logged 8 points in `run/unstable.jsonl`, adding the `ielem` family.

**Union: 7 of the 14 generated families — `copy`, `fa`, `gemm`, `gemv`,
`ielem`, `rr`, `strided` — plus captured kernels.** The 7 clean families
(`bcast`, `cast`, `cumsum`, `elem`, `layernorm`, `softmax`, `transpose`) were
clean *on 30 draws at 8 lowerings each*; that is not the same as "cannot happen"
and I do not claim it.

### 5.2 How many sources each produces

The 17 known-unstable points, re-probed with 48 fresh lowerings:

| point | family | distinct sources / 48 |
|---|---|---|
| `cap_c5c2818305b35efa` | captured | **33** |
| `cap_c84d73cfbb167409` | captured | **29** |
| `gen_rr_cedf40c305ae7c` | rr | 10 |
| `gen_rr_53f17794e18f4b` | rr | 6 |
| `gen_gemm_5c04c2487c1165` | gemm | 5 |
| `gen_copy_68f6d2b39b00f0` | copy | 4 |
| 10 further points | fa, gemm, gemv, ielem, rr, strided | 2 each |
| `gen_strided_b6b911131cdd9a` | strided | 1 (did not recur in this block) |

The two captured kernels were re-measured independently at 15/16 and 14/16
(§2a), and again by the reviewer at 15/16 and 12/16. **Six distinct sources in
six lowerings is no longer the strongest statement available; thirty-three in
forty-eight is.**

The split is by cause, not by luck. Every point with more than 5 sources is a
`MergeSharedMemoryAllocations` point — a permutation has many outcomes. The
cause-A points top out at 2–6, because a proof that either succeeds or fails is
a coin, and a handful of independent coins.

Read the two columns differently, though. A cause-A point with 2 sources has
**two different programs**; a cause-B point with 33 sources has, as far as every
measurement in §1.3 goes, **one program written 33 ways**. The big numbers in
this table are the harmless ones.

### 5.3 How much the 8-lowering probe missed

98 instances (7 per family) that agreed 8 times out of 8 were re-probed with 32
fresh lowerings: **0 turned out unstable**.

That bounds the miss rate for *moderate*-rate points. For a point whose minority
outcome has probability `p`, surviving the 8-sample filter has probability
`p^8 + (1-p)^8`, and then agreeing 32 more times has probability
`p^32 + (1-p)^32`. At `p = 0.16` (the rate this campaign measures for
`HIT-0043`'s point through the real compile path) that second number is 0.004 —
so a point like that would essentially always have been caught by the deeper
probe. Seeing 0 of 98 says there is no reservoir of moderate-rate unstable
points hiding behind the n=8 screen. It says **nothing** about points with
`p < 0.02`; those would pass both screens, and `gen_strided_b6b911131cdd9a` —
2 sources in a block of 8 and 2 in a block of 16, but only 1 in a fresh block
of 48 — is a real example of how noisy the low-rate tail is.
**2.2% / 2.7% are lower bounds.**

### 5.4 Does the lottery change the answer?

Compile `R` times, group the compiled kernels by source hash, run one of each
group on the same input, compare bit for bit:

| point | compiles | distinct sources | outputs |
|---|---|---|---|
| `gen_rr_53f17794e18f4b` | 12 | 4 | all 4 **bit-identical** |
| `cap_c5c2818305b35efa` | 12 | 10 | all 10 **bit-identical** |
| `gen_fa_ca4121f27fc1c5` (`HIT-0043`) | — | 2 | **differ**: 3327/98560 elements, `max_abs 25165824.0`, `ulp_max 33758` |

So the honest statement is: on the points measured here the different sources
agree, and the one known case where they disagree is the one `HIT-0043` traced.
Two `normal` draws on two kernels is a small sample — this is not evidence that
the lottery is generally safe.

For the `cap_c5c2818305b35efa` row there is now a reason rather than luck: §1.3
shows its 10 sources are one program with the scratch pointers renamed, so they
*must* agree. That reason covers the three reproducible cause-B points and
nothing else. It says nothing about the cause-A points, where the two sources
really are two programs.

### 5.5 The pass behind each point

19 points, 12–16 lowerings each:

| point | family | sources | culprit pass(es) |
|---|---|---|---|
| `cap_09d61c7623de9654` | captured | 14/16 | `MergeSharedMemoryAllocations` |
| `cap_c5c2818305b35efa` | captured | 14/16 | `MergeSharedMemoryAllocations` |
| `cap_c84d73cfbb167409` | captured | 13/16 | `MergeSharedMemoryAllocations` |
| `gen_rr_cedf40c305ae7c` | rr | 7/16 | `MergeSharedMemoryAllocations` |
| `gen_rr_53f17794e18f4b` | rr | 5/16 | `LegalizeSafeMemoryAccess`, `Simplify#3`, `ThreadSync.shared.dyn` |
| `gen_copy_68f6d2b39b00f0` | copy | 4/16 | `LegalizeSafeMemoryAccess`, `ThreadSync.shared.dyn` |
| `gen_gemm_5c04c2487c1165` | gemm | 4/16 | `ThreadSync.shared.dyn` |
| `gen_fa_3a8e3db142683d` | fa | 2/16 | `LegalizeSafeMemoryAccess` |
| `gen_gemv_755969fbda178e` | gemv | 2/16 | `LegalizeSafeMemoryAccess` |
| `gen_ielem_ff479647c52c97` | ielem | 2/16 | `LegalizeSafeMemoryAccess` |
| `gen_gemv_3847e42070b1e3` | gemv | 2/16 | `ThreadSync.shared.dyn` |
| `gen_rr_25494d603d0032` | rr | 2/16 | `ThreadSync.shared.dyn` |
| `gen_strided_1319c31c244951` | strided | 2/16 | `ThreadSync.shared.dyn` |
| `gen_strided_b0554a0ef12f2a` | strided | 2/16 | `ThreadSync.shared.dyn` |
| `gen_strided_b6b911131cdd9a` | strided | 2/16 | `ThreadSync.shared.dyn` |
| `gen_strided_bf661da36a9b68` | strided | 2/16 | `ThreadSync.shared.dyn` |
| `gen_strided_c2074f058c3b51` | strided | 2/16 | `ThreadSync.shared.dyn` |
| `gen_fa_ca4121f27fc1c5` | fa | 2/12 | `ThreadSync.shared.dyn` (= `HIT-0043`) |
| `gen_gemm_0b0ee91756f729` | gemm | 1/16 | none at this depth |

Specs for the generated points, for §2's `{"kind": "gen", "spec": ...}` form:

```json
gen_rr_53f17794e18f4b   {"fam":"rr","p":{"M":513,"N":64,"bM":128,"bN":256,"th":32,"st":0,"kind":"max"},"pc":{"tl.ptxas_register_usage_level":2,"tl.config_index_bitwidth":32}}
gen_ielem_ff479647c52c97 {"fam":"ielem","p":{"M":512,"N":191,"bM":128,"bN":128,"th":32,"dt":"int8"},"pc":{"tl.ptxas_register_usage_level":8}}
gen_strided_b0554a0ef12f2a {"fam":"strided","p":{"M":128,"N":512,"pad":2,"bM":128,"bN":128,"th":256,"dt":"float32"},"pc":{"tl.ptxas_register_usage_level":8}}
gen_gemv_755969fbda178e {"fam":"gemv","p":{"M":384,"K":1888,"bM":128,"bK":64,"th":512,"st":0,"dt":"float16"},"pc":{"tl.ptxas_register_usage_level":10,"tl.config_index_bitwidth":64}}
gen_rr_25494d603d0032   {"fam":"rr","p":{"M":384,"N":64,"bM":128,"bN":128,"th":32,"st":0,"kind":"max"},"pc":{"tl.ptxas_register_usage_level":0,"tl.config_index_bitwidth":64}}
gen_strided_bf661da36a9b68 {"fam":"strided","p":{"M":514,"N":768,"pad":17,"bM":64,"bN":128,"th":64,"dt":"float16"},"pc":{"tl.ptxas_register_usage_level":8,"tl.config_index_bitwidth":64}}
```

The remaining generated specs are in the sweep records; the four captured ids
need no spec. **Gap:** those sweep records are not in the line's files — a
`grep` for `gen_rr_cedf40c305ae7c` across `fuzz-tilelang-r2/` finds nothing — so
the 13 points without a spec above, including the biggest cause-B generated
point, cannot be rebuilt from this report. The four captured points can, and
they are what §1.3 rests on.

Reviewer re-ran four rows of this table from scratch at 16–20 lowerings each and
got the same culprit and the same input hash every time:
`cap_c5c2818305b35efa` → `MergeSharedMemoryAllocations` (input `e30b5815`),
`gen_ielem_ff479647c52c97` → `LegalizeSafeMemoryAccess` (input `2ad0e996`),
`gen_strided_b0554a0ef12f2a` → `ThreadSync.shared.dyn` (input `64c04905`),
`gen_fa_ca4121f27fc1c5` → `ThreadSync.shared.dyn` (input `6549ecf3`). 4 of 4.

---

## 6. What it means for a user

**Reproducible builds are broken.** Building the same source twice does not give
the same binary. Anything that checks build reproducibility, signs artifacts, or
diffs two builds to prove "no change" will report a change that is not there.

**Content-addressed kernel caches are unreliable.** TileLang's own cache is
keyed on the input, so it hides the problem by handing back whichever draw was
cached first — which means the code you ship is one arbitrary sample of the
lottery, chosen by whichever machine warmed the cache. Turn the cache off (as
this campaign must, `TILELANG_DISABLE_CACHE=1`) and the variation is visible
immediately. A cache keyed on the *output* would miss constantly.

**Every bug report against TileLang is weakened, including this campaign's.**
"Compile this and run it" may not give the reporter's binary. Measured on this
line: of the 28 reports audited at 12 lowerings per arm, **4 sit on a point where
an arm does not compile to the same CUDA twice** (`HIT-0043`, `HIT-0078`,
`HIT-0093`, `HIT-0098`) — `run/ref_stability_audit.txt`. Two of the campaign's
early reports (`HIT-0044`, `HIT-0045`) turned out to be the lottery alone:
byte-identical CUDA on both arms and a difference recorded anyway. A
differential fuzzer that changes one compiler setting and compares outputs
cannot tell "the setting did it" from "the compiler disagreed with itself"
unless it screens for this first — which is why R5 now runs a compile-determinism
gate on every point and again before every report.

**Performance numbers: not on the evidence here.** An earlier draft of this
report said `MergeSharedMemoryAllocations` gives a different packing, hence a
different shared-memory footprint, hence different occupancy. **Measurement
does not support that.** On all three cause-B points that can be re-run the byte
offsets and the merged size are identical in every draw (§1.3); only names move.
A benchmark that recompiles between runs will get the same footprint. The risk
is not zero in general — the same tie also sits in the loop that assigns the
offsets, so a kernel with same-named buffers of *different* sizes could get a
different packing — but no such case was found, and none should be claimed
until one is.

**Correctness is at risk on the `ThreadSync` path specifically.** Most draws are
value-equivalent, but `HIT-0043` shows the pass omitting a barrier at a real
128-writer→1-reader boundary. That kernel is wrong about one compile in four,
and nothing in the source, the flags or the environment tells you which compile
you got. A test suite that passes is not evidence the shipped binary is the one
that passed.

---

## 7. SMT reachability analysis

1. **Root cause in semantic terms.** Two different things, and they need
   different answers.

   *Cause B* (`MergeSharedMemoryAllocations`): the constructs are
   `T.alloc_buffer(..., scope="shared.dyn")`, the merged allocation, the
   `Bind(var, handle_add_byte_offset(merged, k))` alias statements, and the
   `T.tvm_access_ptr(..., workspace_k, 0, 256, 2)` operands of `tl::AllReduce`.
   The semantic content of the pass is a *storage assignment*: a map from
   logical buffers to offsets inside one merged allocation, valid iff two
   buffers sharing an offset have disjoint live ranges. What varies run to run
   is **not** that map — §1.3 measures it identical every time — but the order
   in which the alias statements are written out, which decides the *name* each
   buffer var is printed with. Semantically the outputs are the same program up
   to renaming: alpha-equivalent, and in fact equal as soon as names are
   resolved to offsets. There is no semantic error at all in the outputs seen
   here; the error is that the *text-producing function* is not single-valued.

   *Cause A* (`ThreadSync`, `LegalizeSafeMemoryAccess`, `Simplify`): the
   constructs are `T.tvm_storage_sync("shared.dyn"[, id, count])`,
   `T.if_then_else(bound_cond, BufferLoad, 0)` and loop guards. The semantic
   content is a *proof obligation* — "this index is in bounds", "this many
   threads reach here" — and the pass emits or omits a guard depending on
   whether the obligation discharged. The obligation's truth value is fixed by
   the program; the compiler's knowledge of it is not.

2. **Is the difference a pure input→output function difference?** No, and for
   an unusual reason. In the normal case a checker gets one input program and
   one output program and asks whether they agree on all inputs. Here **there
   is no single output program to check**: the compiler emits a different one
   each time it is asked. Worse, in the cases measured in §5.4 the different
   outputs *are* input→output equivalent to each other — so a functional
   equivalence check between any two of them returns "equivalent" and reports
   no problem, which is the correct answer to the question asked and the wrong
   answer to the question that matters. And in `HIT-0043`'s case, where they are
   not equivalent, the difference is a missing happens-before edge, which a
   value-semantics check cannot see either.

3. **Fundamental verdict.** Two verdicts, one per cause.

   *Cause B* — **`SMT-decidable in principle`, but it answers the wrong
   question, and here it answers it trivially.** Each individual compile is a
   deterministic input→output function and an adequate model of shared memory as
   a second address space with offsets would let a checker verify that *this*
   compile is correct. It would pass all 33 of them — and, since the 33 turn out
   to be one program with renamed pointers, any two of them are equal after the
   very first step of any checker, name resolution. Nondeterminism is a property
   of the *compiler as a function*, not of any one translation, and equivalence
   checking is defined on pairs of programs. The only way an SMT-based tool
   catches this is operationally: run it on every compile and additionally hash
   the output, i.e. use it as a per-build gate rather than a one-off offline
   check. That is a harness property, not a modelling capability. A plain
   `sha256` of the emitted source catches cause B at zero cost and an SMT checker
   catches it never; that is the honest ranking for this cause.

   *Cause A* — **`needs a concurrency / memory-ordering model`** for the
   `ThreadSync` half, exactly as `HIT-0043` §5 argues, and **`SMT-decidable in
   principle`** for the `LegalizeSafeMemoryAccess` / `Simplify` half: whether
   a dropped bounds guard changes the result *is* a value-semantics question
   about one program instance, and a model that carries buffer shapes and
   `if_then_else` would expose it. In the instances measured here the guard that
   varies is redundant, so both outputs are equivalent and the checker correctly
   says so; the defect only becomes a bug if the analyzer ever fails in the
   other direction and drops a guard that was load-bearing. **I did not find
   such a case and do not claim one exists.**

4. **Minimum semantic vocabulary required.** To check any single compile of
   cause B: shared memory as a second address space; a buffer as a byte range
   inside one merged allocation, with the offset as part of the model, not
   abstracted away; live ranges as intervals over the statement sequence; and
   the invariant "two buffers may share an offset only if their intervals are
   disjoint". For the `ThreadSync` half of cause A, `HIT-0043` §5.4 lists it and
   I do not repeat it: per-thread visibility, a barrier as a happens-before edge
   over a *sub-range* of threads (`tvm_storage_sync(scope, id, count)`),
   `stmatrix` as 128 thread-indexed writes, `tma_store` as one bulk read,
   `tl_shuffle_elect` as "exactly one thread". For the guard half: buffer shapes
   as symbolic bounds and `if_then_else` as a total function with the guard as
   its condition — no concurrency needed.

5. **Would the abstraction hide it?** The FP axiom profile is irrelevant to all
   four passes: not one of them changes an arithmetic expression. Under both
   exact bit-to-bit and reassoc-allowed profiles the outputs of cause B are
   equal, and the outputs of the guard half of cause A are equal. What hides
   cause A's `ThreadSync` half is the *sequential-execution* abstraction, not
   the FP one. Cause B is hidden by something even more basic than the storage
   model: **treating names as names**. Every IR-level checker resolves a
   variable to the thing it is bound to, and the moment `workspace_3` and
   `workspace_4` both resolve to `merged + 229376` the 33 outputs collapse to one
   term. Keeping offsets concrete does not help either, since the offsets are
   what agree. Nothing an equivalence checker can be built to look at
   distinguishes these outputs; only the byte text does.

6. **Inherent cost.** Cheap in both cases, if the model exists. Cause B is an
   interval-disjointness check over a handful of buffers with no arithmetic —
   trivial. Cause A's guard half is a bounds implication over affine index
   expressions — small, and it is UNSAT (proving equivalence) in every case
   measured, which is the cheap direction. Cause A's barrier half is a
   race-freedom query over one buffer, one write set, one read set, again with
   no arithmetic. The expensive part of every one of these is building the
   model, not solving it. The genuinely hard part is none of the above: it is
   that catching nondeterminism requires running the checker on *every* compile,
   so the cost that matters is per-build latency, not per-query difficulty.

---

## 8. Confidence

**Very high** that the compiler is nondeterministic and that the scope is as
described. **High** that cause B is address-order inside
`MergeSharedMemoryAllocations` and that its effect is renaming, not repacking.
**Medium** on which line of that pass does it. **Medium** on how many distinct
causes there really are.

Very high on the phenomenon: these are direct measurements with a byte-identical
input, repeated in fresh processes and inside one process, at four different
depths (8, 12, 16, 48 lowerings), on generated kernels and on captured
production kernels, and independently re-measured — `cap_c5c2818305b35efa`
gave 7/8, 10/12, 14/16, 14/16, 15/16, 33/48 and (reviewer) 15/16, 13/16 and 7/8
distinct sources in nine separate blocks run by six different scripts.
`HIT-0043`'s method reproduces `HIT-0043`'s answer on `HIT-0043`'s instance —
re-run by the reviewer in a fresh process, same three hashes — which is the
control that says the pass-attribution method is sound.

High on cause B being address-order in this pass: the pass calls no solver
whatsoever; it reads two `VarNode *`-keyed and one `Object *`-keyed
`unordered_map` (`:1589`, `:1593`, `:1600`); and holding the addresses fixed —
50 applications of the pass object to one in-memory module — makes it
deterministic, while the same experiment on `ThreadSync` still flips. That last
one is the measurement that turns "address-keyed container, therefore probably
address order" into evidence.

Medium on *which* line: the effect that reaches the output is the alias
emission order, and the tie that lets it move is
`MakeAliasBindings`'s `(offset, name_hint)` sort at `:519-528` over the vector
filled at `:514` — the six buffers named `workspace` tie on both keys. That is a
reading of the code plus a matching measurement, not an instrumented trace. The
`:1197` kill-point loop, which an earlier draft blamed, walks an address-keyed
map too and may also vary, but it has no visible effect here: the offsets are
identical in every output. What would settle both: rebuild with the loops made
ordered (sort by `seq` position, and break the name tie with something total)
and check the source hash stops moving; or print the traversal order at `:514`
and `:1197` across runs.

Medium on the count of causes: two is proved (cause B cannot be cause A). Three
or four is possible and I could not test it, because `arith::Analyzer` lives in
`3rdparty/tvm`, which is not checked out in this source copy, so I could not
read `CanProve` and could not confirm whether `LegalizeSafeMemoryAccess` and
`Simplify` reach Z3 at all. What would settle it: build with the submodule
present and instrument `arith::Analyzer::CanProve` to log its answer, or set a
fixed Z3 random seed and resource limit and see which of the four passes become
stable.

Two things I want to flag rather than bury. First, the 2.2% / 2.7% instability
rates are **lower bounds** measured at 8 lowerings; §5.3 bounds the miss for
moderate-rate points but not for the low-rate tail, and one point in this
report's own list flipped between blocks. Second, §5.4's "the outputs agree" is
two kernels on one input draw each. It is enough to show the lottery is often
text-only. It is nowhere near enough to say it is safe.

---

## 9. Relationship to `r2/HIT-0043`

`HIT-0043` owns the one fully traced instance: the `ThreadSync.shared.dyn`
mechanism down to `thread_storage_sync.cc:298`'s `return Stmt()`, the exact two
dropped `tl::__sync_thread_partial(3, 128)` lines, the 38/12 rate over 50
in-process applications, the proof that the two kernels compute different
values, and the concurrency-model SMT argument. None of that is repeated here.

This report owns the scope: that `ThreadSync` is one of at least four
nondeterministic passes, that it accounts for 11 of 19 probed points and not the
rest, that the widest source counts come from a completely different pass with a
completely different cause, and that the phenomenon reaches 7 of 14 kernel
families and 2.7% of the captured corpus.

Read `HIT-0043` for the mechanism. Read this for how much of the compiler it is.

---

## 10. Reviewer verification, 2026-08-19

Kept. Everything below was re-run by the reviewer in fresh processes against the
same wheel (TileLang 0.1.12, H100 sm_90a); the source was read against the
matching checkout (`tilelang-src`, `git log -1` = `2d63708c [Release] Bump
version to 0.1.12`), so the line numbers apply to the wheel.

**Held up.**

* §2a: 15/16 and 12/16 distinct CUDA sources.
* §2b: 13 distinct sources, one culprit, `51:MergeSharedMemoryAllocations`,
  input `e30b5815`, nothing else in the pipeline.
* The self-validation: on `HIT-0043`'s own instance the method returns
  `55:ThreadSync.shared.dyn` and nothing else, with `HIT-0043`'s own three
  hashes. Everything in this report rests on that control and it holds.
* Two more §5.5 rows re-run from their specs: `gen_ielem_ff479647c52c97` →
  `LegalizeSafeMemoryAccess`, `gen_strided_b0554a0ef12f2a` →
  `ThreadSync.shared.dyn`, both with the input hash quoted here.
* The `:1600` `event_map_`, `:556` "sort deterministically by name" and the
  cause-A citations (`thread_storage_sync.cc:353/:468/:1685`,
  `legalize_safe_memory_access.cc:167/:205`, `simplify.cc:364/:420/:549`) are
  all where the report says they are. `3rdparty/tvm` is indeed empty, so
  `arith::Analyzer::CanProve` cannot be read and the "undetermined" verdict on
  the cause-A count is the right call.

**Corrected.** The first draft pinned cause B on the kill-point reorder loop
(`:1197`/`:1218`) moving live ranges and permuting the buffer→offset assignment.
Measurement says otherwise: over 7 distinct outputs of that pass the offsets and
the merged size are identical, and the whole difference is a rename among
aliases bound to the same address. §1.3, §4, §5.2, §5.4, §6 and §7 were rewritten
to say that, and the "different packing → different occupancy" performance claim
was withdrawn. The new candidate step is the `(offset, name_hint)` sort in
`MakeAliasBindings`, which ties for this kernel's six identically-named
`workspace` buffers.

**Added.** The fixed-address experiment (§2c): the pass is deterministic when
the node addresses are held fixed, while `ThreadSync` is not, which is what
upgrades "address-keyed container" from a guess to evidence.
