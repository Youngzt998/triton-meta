"""checker.corpus -- grade the static equivalence checker over a corpus of compiled kernels.

This is PR #2781's measurement, made runnable by someone else. It is not a new experiment and
nothing here was designed from scratch: the sweep, the arithmetic and the column set follow
`build_checker_table.py`, the script that produced the diff's table, and the numbers it is
compared against are copied verbatim into `artifact_eval/prior_results.txt`.

The corpus is an INPUT, not part of this repository -- 11 GB, and regenerated rather than
archived. `artifact_eval/corpus_builder/` holds the scripts that build it and says what it costs.
Without one this step prints how to get one and exits 0.

CPU only. The checker reads text; nothing here touches a GPU. Building the corpus does.

    CHECKER_CORPUS          where the corpus is         (default ~/bitwise-equiv/local_evaluation/corpus)
    CHECKER_CORPUS_CAP      configs per group, 0 = all  (default 0 -- the diff graded with no cap)
    CHECKER_CORPUS_KERNELS  comma-separated kernels     (default every kernel in the corpus)
    CHECKER_CORPUS_WORKERS  checker process pool size   (default 16)
    CHECKER_CORPUS_CHECKER  module:function             (default the forward PTX checker)
    CHECKER_CORPUS_FRESH    1 = regrade groups already in cache/checker.corpus.jsonl
    CHECKER_CORPUS_RECYCLE  files before the pool is replaced  (default 5000)
    CHECKER_CORPUS_STALL    seconds of no result before the pool is called dead (default 600)

They are environment variables rather than flags because `artifact.py`'s CLI is shared by every
step; `gemm.perf.static` does the same.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from ._common import CACHE, writer

NAME = "checker.corpus"
ORDER = 80
DESCRIPTION = "grade the equivalence checker over the compiled corpus: classes, over-merges"
IMPLEMENTED = True

DEFAULT_CORPUS = os.path.expanduser("~/bitwise-equiv/local_evaluation/corpus")
DEFAULT_CHECKER = "bitequiv.ptx.forward.interp:forward_module_descriptor"

# Which groups the diff counts as the MMA side. Reproducing its split exactly: on the corpus this
# rule picks out 14 of the 93 groups holding 41,768 of the 51,152 configs, which are the two
# numbers its "MMA total" row reports. Everything else is the reduction side.
MMA_KERNELS = ("flash_attention", "LC_splitk_gemm_f16")

TABLES = {
    "checker.corpus": {
        "doc":
        "One row per (kernel, dtype) group of the corpus, plus three rows holding the totals. "
        "For each group: run the checker on every configuration's compiled IR and group the "
        "configurations by the answer, then group the same configurations by `empirical_key` -- "
        "the hash of the bytes they actually returned when the kernel was launched, recorded "
        "when the corpus was built. Two numbers matter and they point in opposite directions. "
        "`over_merges` is the soundness violation and MUST be 0: the checker called two "
        "configurations equal and the hardware disagreed. `over_splits` is tuning freedom the "
        "checker gave up to stay safe. Compare `checker_cls` against the `after` column of "
        "`prior_results.txt`.",
        "cols": [
            ("scope", "str", "`group` for a (kernel, dtype) row, `total` for a summed row. Do not "
             "sum the totals back in"),
            ("kernel", "str", "corpus kernel directory, or reduction/mma/everything on a total row"),
            ("dtype", "str", "element dtype: f16, bf16, f32, fp8 or fp8e5m2"),
            ("configs", "int", "configurations graded: those that compiled and launched"),
            ("infeasible", "int", "configurations that failed to build; recorded by the builder, excluded here"),
            ("checker_cls", "int", "equivalence classes the checker produced. Fewer is more tuning "
             "freedom proven equal. This is the diff's `after` column"),
            ("checker_max", "int", "size of the largest checker class"),
            ("empirical_cls", "int", "classes the recorded outputs actually fall into. The ground "
             "truth, and the number checker_cls would reach if nothing were left to recover"),
            ("empirical_max", "int", "size of the largest empirical class: the recovery ceiling"),
            ("over_merges", "int", "pairs the checker called equal whose recorded bytes differ. "
             "The soundness violation. MUST be 0"),
            ("over_splits", "int", "pairs whose bytes match that the checker refused to merge. "
             "Safe, but recovery left on the table"),
            ("recovery", "str", "one word for the split side: perfect, over-split, fail-closed "
             "(every configuration its own class, nothing recovered) or unsound"),
            ("v2v5_checker", "str", "n/N: checker classes that mix mma v2 (mma.sync) with v5 "
             "(tcgen05/wgmma). Non-zero means the checker proved the two atoms equal"),
            ("v2v5_empirical", "str", "n/N: the same count on the empirical classes -- whether the "
             "hardware agrees they are equal"),
            ("seeds", "str", "random input draws per configuration behind empirical_key. More than "
             "one value in a group means its keys are not comparable and it needs rebuilding"),
            ("cap", "int", "configurations per group this run graded, 0 = all of them. Class counts "
             "under a cap are for the sample, not for the space"),
            ("checker", "str", "checker under test, as module:function"),
            ("corpus", "str", "the corpus directory these rows were read from"),
            ("seconds", "float", "wall clock spent grading this group"),
            ("notes", "str", "anything a reader has to know before trusting the row"),
        ],
    },
}

_CK = None  # per-worker checker, set by _init_worker
_CHUNK = 8  # files handed to a worker per task


def _init_worker(spec):
    global _CK
    import importlib
    mod, fn = spec.split(":")
    _CK = getattr(importlib.import_module(mod), fn)


def _checker_of(path):
    """Worker: (checker key, mma atom) for one compiled file, or None if it could not be read.

    Returning None rather than raising keeps one unreadable file from killing a sweep of 54,120.
    A configuration the checker cannot get through is therefore dropped, not merged with anything,
    and the group's row says how many were dropped -- a dropped configuration changes the class
    counts, so it is never silent.
    """
    try:
        text = open(path).read()
        atom = "v5" if ("tcgen05" in text or "wgmma" in text) else ("v2" if "mma.sync" in text else "other")
        return str(_CK(text)), atom
    except Exception:  # noqa: BLE001
        return None


def _checker_chunk(paths):
    """Worker: `_checker_of` over a slice of files, in order. What `Executor.map` does internally.

    Submitting the chunks by hand rather than through `map` is what lets the parent notice a pool
    that has stopped answering; see `_Pool._run`.
    """
    return [_checker_of(p) for p in paths]


class _PoolDied(Exception):
    """The worker pool stopped being able to grade: a worker died, or nothing came back."""


class _Pool:
    """A checker process pool that is REPLACED rather than repaired.

    Two decisions, and the first one is why this class exists at all.

    NO per-worker recycling. `ProcessPoolExecutor(max_tasks_per_child=N)` looks like the obvious
    way to bound worker memory, and it is what hung this step. When a worker retires the pool is
    supposed to spawn a replacement, but `_adjust_process_count` spawns one only if its
    idle-worker semaphore is empty, and that semaphore gains a permit for every task that
    finishes. A long run of results with no new submissions -- exactly one big group -- leaves
    hundreds of permits, so every retiring worker is replaced by nothing. The pool drains to zero
    with work still queued, and then no worker sentinel is left for the manager thread to watch:
    it waits, the parent waits on a future nobody will ever complete, and there is no exception,
    no message and no end. That is reproducible in twenty lines and does not need this checker.
    With recycling off, a worker only ever leaves by dying, which trips its sentinel and raises
    `BrokenProcessPool` at once.

    Memory is still bounded, by throwing away the WHOLE pool every `recycle` files. New processes
    reclaim strictly more than a new child does, and the boundary is a plain arithmetic count
    instead of a race between three unrelated numbers. Measured on this corpus the bound is
    insurance rather than a need: 300 flash-attention parses in one process held resident memory
    at 0.03 GB, and the largest file in the corpus peaks at 0.32 GB and gives it back.

    Every failure of a worker is turned into `_PoolDied`, which the caller reports against the
    group being graded. Two ways a pool can fail:
      * a worker dies -- `BrokenProcessPool`, raised within milliseconds of the death;
      * nothing at all comes back for `stall` seconds -- the backstop for a cause nobody has
        seen yet, so that the worst case is a message rather than silence.
    """

    def __init__(self, workers, checker, recycle, stall):
        self.workers, self.checker = workers, checker
        self.recycle, self.stall = recycle, stall
        self._ex, self._served, self.generations = None, 0, 0

    def _executor(self):
        if self._ex is None:
            ctx = mp.get_context("spawn")  # each worker imports the checker fresh
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_init_worker,
                                           initargs=(self.checker, ))
            self._served = 0
            self.generations += 1
        return self._ex

    def _stop(self, alive=True):
        """Drop the current pool. `alive=False` means do not wait on it -- it is dead or stuck."""
        ex, self._ex = self._ex, None
        if ex is None:
            return
        if not alive:
            for p in list(getattr(ex, "_processes", {}).values()):
                p.kill()
        ex.shutdown(wait=alive, cancel_futures=True)

    def _run(self, paths):
        """Grade `paths` on the current pool, in order. Raises `_PoolDied` if it cannot."""
        ex = self._executor()
        try:
            futures = [ex.submit(_checker_chunk, paths[i:i + _CHUNK]) for i in range(0, len(paths), _CHUNK)]
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=self.stall, return_when=FIRST_COMPLETED)
                if not finished:
                    raise _PoolDied(f"no configuration came back for {self.stall}s with "
                                    f"{len(pending) * _CHUNK} still queued")
            out = []
            for f in futures:
                out.extend(f.result())
            return out
        except BrokenProcessPool as exc:
            raise _PoolDied(f"a worker died ({exc})") from exc

    def grade(self, paths):
        """(checker key, mma atom) or None per path, in order, recycling the pool as it goes.

        A slice is retried once on a brand new pool, because the commonest reason a worker dies on
        a shared machine is that something else on it took the memory, and that passes.
        """
        out, i = [], 0
        while i < len(paths):
            room = self.recycle - self._served
            if room <= 0:
                self._stop()
                room = self.recycle
            batch = paths[i:i + room]
            try:
                out.extend(self._run(batch))
            except _PoolDied as first:
                print(f"    POOL DIED: {first}. Rebuilding it and retrying these "
                      f"{len(batch)} configurations.", flush=True)
                self._stop(alive=False)
                try:
                    out.extend(self._run(batch))
                except _PoolDied as second:
                    self._stop(alive=False)
                    raise _PoolDied(f"{second}, and the same on a fresh pool") from second
            self._served += len(batch)
            i += len(batch)
        return out

    def close(self):
        self._stop()


def _groups(corpus, only):
    """Every (kernel, dtype) group, smallest first.

    Smallest first because the step appends each row the moment it has it: a reader sees the
    cheap groups within seconds, and a kill costs whichever single group was in flight instead
    of the twenty that would still be waiting behind flash attention. `build_checker_table.py`
    ordered its sweep the same way and for the same reason. It is NOT what keeps the run alive
    -- that is `_Pool` -- and the run is correct in any order.

    No group is skipped. The five loop-carried scan and welford kernels that the older harness
    skipped for a checker memory blowup grade here in a few seconds each, and if one ever did
    exhaust a worker it would now cost that group a clear failure line rather than the run.
    """
    out = []
    for kernel in sorted(os.listdir(corpus)):
        kdir = os.path.join(corpus, kernel)
        if not os.path.isdir(kdir) or (only and kernel not in only):
            continue
        for dtype in sorted(os.listdir(kdir)):
            ddir = os.path.join(kdir, dtype)
            if os.path.isdir(ddir):
                out.append((len([fn for fn in os.listdir(ddir) if fn.endswith(".json")]), kernel, dtype))
    return [(kernel, dtype) for _, kernel, dtype in sorted(out)]


def _read_group(ddir, cap):
    """The configuration records of one group: (feasible records, ptx paths, infeasible, unreadable).

    A capped run takes a STRIDED sample rather than the first `cap`, so it spans the configuration
    space instead of one corner of it -- file names are configuration hashes, so the first N are an
    arbitrary but not a representative slice.
    """
    names = [fn for fn in sorted(os.listdir(ddir)) if fn.endswith(".json")]
    if cap and len(names) > cap:
        names = names[::max(1, len(names) // cap)][:cap]
    recs, paths, infeasible, unreadable = [], [], 0, 0
    for fn in names:
        try:
            r = json.load(open(os.path.join(ddir, fn)))
        except Exception:  # noqa: BLE001
            unreadable += 1
            continue
        if not r.get("ok"):
            infeasible += 1
            continue
        recs.append(r)
        paths.append(os.path.join(ddir, r["ptx"]))
    return recs, paths, infeasible, unreadable


def _grade(kernel, dtype, ddir, pool, cap):
    """One (kernel, dtype) row, and how many of its configurations had to be dropped.

    The grouping and the soundness arithmetic come from `equivalence_fuzzer`, the same standalone
    module `build_checker_table.py` called, so this produces the same numbers rather than a second
    opinion about what they mean. It is pure stdlib, hence importable without a GPU. `over_merges`
    with the two key sets swapped is the over-split count -- it is a symmetric question, "pairs one
    grouping merges that the other separates".

    Raises `_PoolDied` if the worker pool could not be kept alive for this group.
    """
    from bitequiv.evaluation import equivalence_fuzzer as fz
    started = time.time()
    recs, paths, infeasible, unreadable = _read_group(ddir, cap)

    graded = []
    for r, res in zip(recs, pool.grade(paths)):
        if res is None:
            unreadable += 1
            continue
        r["checker_key"], r["mma"] = res
        graded.append(r)

    row = {
        "scope": "group", "kernel": kernel, "dtype": dtype, "configs": len(graded), "infeasible": infeasible, "cap":
        cap, "seconds": round(time.time() - started, 1)
    }
    if not graded:
        note = "no configurations graded" + (f"; {unreadable} unreadable" if unreadable else "")
        return {**row, "recovery": "no-configs", "notes": note}, unreadable

    idx = list(range(len(graded)))
    ck = {i: graded[i]["checker_key"] for i in idx}
    emp = {i: graded[i]["empirical_key"] for i in idx}
    ck_cls, emp_cls = fz.partition(idx, ck), fz.partition(idx, emp)
    over_merges = fz.over_merges(idx, ck, emp)
    over_splits = fz.over_merges(idx, emp, ck)

    if over_merges:
        recovery = "unsound"
    elif not over_splits:
        recovery = "perfect"
    elif len(ck_cls) == len(graded) and len(emp_cls) < len(graded):
        recovery = "fail-closed"
    else:
        recovery = "over-split"

    seeds = sorted({r.get("seeds") for r in graded}, key=lambda x: (x is None, x))
    atoms = Counter(r["mma"] for r in graded)
    notes = []
    if over_merges:
        notes.append(f"OVER-MERGE: {over_merges} pairs -- a soundness bug, not a tuning trade-off")
    if len(seeds) > 1:
        notes.append(f"MIXED SEEDS {seeds}: empirical keys are seed-count dependent and so are not "
                     "comparable inside this group; rebuild it with one --seeds")
    if cap:
        notes.append(f"strided sample of {cap} per group: class counts are for the sample")
    if set(atoms) != {"other"}:
        notes.append("mma:" + ",".join(f"{k}={v}" for k, v in sorted(atoms.items())))
    if unreadable:
        notes.append(f"{unreadable} files unreadable")

    return {
        **row,
        "checker_cls": len(ck_cls),
        "checker_max": max(len(g) for g in ck_cls),
        "empirical_cls": len(emp_cls),
        "empirical_max": max(len(g) for g in emp_cls),
        "over_merges": over_merges,
        "over_splits": over_splits,
        "recovery": recovery,
        "v2v5_checker": f"{sum(1 for g in ck_cls if len({graded[i]['mma'] for i in g}) > 1)}/{len(ck_cls)}",
        "v2v5_empirical": f"{sum(1 for g in emp_cls if len({graded[i]['mma'] for i in g}) > 1)}/{len(emp_cls)}",
        "seeds": ",".join(str(s) for s in seeds),
        "notes": "; ".join(notes),
    }, unreadable


def _is_mma(row):
    return row["kernel"].startswith("gemm") or row["kernel"] in MMA_KERNELS


def _totals(rows, cap, checker, corpus):
    """The diff's total rows: the reduction side, the MMA side, and everything.

    Its fourth total, a 72-group subset of the reduction side, is deliberately not reproduced --
    that subset is the frozen `REDUCTION_BASELINE` list, which is not in this branch, and guessing
    at which 72 groups it names would produce a number that looks comparable and is not.
    """
    is_mma = _is_mma
    out = []
    for name, want in (("reduction", lambda r: not is_mma(r)), ("mma", is_mma), ("everything", lambda r: True)):
        part = [r for r in rows if r.get("configs") and want(r)]
        if not part:
            continue
        out.append({
            "scope": "total",
            "kernel": name,
            "dtype": "",
            "configs": sum(r["configs"] for r in part),
            "infeasible": sum(r["infeasible"] for r in part),
            "checker_cls": sum(r["checker_cls"] for r in part),
            "empirical_cls": sum(r["empirical_cls"] for r in part),
            "over_merges": sum(r["over_merges"] for r in part),
            "over_splits": sum(r["over_splits"] for r in part),
            "recovery": "unsound" if any(r["over_merges"] for r in part) else "sound",
            "cap": cap,
            "checker": checker,
            "corpus": corpus,
            "notes": f"sum over {len(part)} groups",
        })
    return out


def _cached():
    """Every record the cache already holds, in order."""
    path = os.path.join(CACHE, f"{NAME}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def _key(row):
    return (row.get("corpus"), row.get("cap"), row.get("checker"))


def _already_done(rows, cap, checker, corpus):
    """Groups the cache already holds for this (corpus, cap, checker), so a killed run resumes.

    A row taken under a different cap or a different checker is a different measurement and does
    not count as done.
    """
    return {(r["kernel"], r["dtype"]) for r in rows if r.get("scope") == "group" and _key(r) == (corpus, cap, checker)}


def _resum(path):
    """Rewrite the cache so every (corpus, cap, checker) carries exactly one set of total rows.

    Totals are the one thing a streaming append gets wrong: grading the corpus in several sittings
    would leave one partial total per sitting, and a reader has no way to tell which is the real
    one. Group rows are never recomputed -- they are appended as they are measured and are what
    makes the run kill-safe. Only the totals are recomputed, over every group row of that key.

    Group rows are put back in name order, so which order the sweep happened to grade them in
    never shows up in the exported table.
    """
    rows = _cached()
    groups = [r for r in rows if r.get("scope") == "group"]
    groups.sort(key=lambda r: (r.get("kernel") or "", r.get("dtype") or "", str(_key(r))))
    out = list(groups)
    for corpus, cap, checker in dict.fromkeys(_key(r) for r in groups):
        part = [r for r in groups if _key(r) == (corpus, cap, checker)]
        out += _totals(part, cap, checker, corpus)
    with open(path + ".tmp", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return [r for r in out if r["scope"] == "total"]


def _no_corpus(corpus):
    builder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus_builder")
    print(f"""
[{NAME}] no corpus at {corpus} -- nothing was measured.

The corpus is an input to this step, not a file in the repository: 11 GB of compiled kernels and
their recorded outputs, 54,120 PTX files in 93 (kernel, dtype) groups. It is regenerated, not
archived. Two ways forward.

  Point at one you already have:

      export CHECKER_CORPUS=/path/to/corpus

  Or build one. The scripts are in the artifact:

      {builder}/README.md

  It needs a GPU throughout -- every configuration is compiled and then launched on every seed --
  and on the machine these numbers came from it took the order of a day and 11 GB of disk.
  `build_local_eval.py` alone gives the reduction and GEMM groups for about 6 GB and a shorter
  build; flash attention is the expensive half. Every builder resumes, so a kill costs nothing.

This step grades whatever groups the corpus has, so a partial one still produces a table -- the
rows it can fill.
""")


def run(args, env):
    corpus = os.environ.get("CHECKER_CORPUS") or DEFAULT_CORPUS
    if not os.path.isdir(corpus):
        _no_corpus(corpus)
        return

    cap = int(os.environ.get("CHECKER_CORPUS_CAP", 0))
    checker = os.environ.get("CHECKER_CORPUS_CHECKER", DEFAULT_CHECKER)
    workers = int(os.environ.get("CHECKER_CORPUS_WORKERS", 16))
    recycle = max(1, int(os.environ.get("CHECKER_CORPUS_RECYCLE", 5000)))
    stall = max(1, int(os.environ.get("CHECKER_CORPUS_STALL", 600)))
    only = {k for k in os.environ.get("CHECKER_CORPUS_KERNELS", "").split(",") if k}
    fresh = bool(os.environ.get("CHECKER_CORPUS_FRESH"))

    groups = _groups(corpus, only)
    if not groups:
        print(f"[{NAME}] {corpus} has no <kernel>/<dtype> directories" + (f" matching {sorted(only)}" if only else "") +
              " -- nothing to grade.")
        return
    done = set() if fresh else _already_done(_cached(), cap, checker, corpus)

    print(f"\n[{NAME}] {len(groups)} groups in {corpus}")
    print(f"  checker  {checker}")
    print(f"  cap      {cap or 'none -- every configuration in every group'}")
    print(f"  workers  {workers}   (CPU only; this step never touches a GPU)")
    print(f"  pool     replaced every {recycle} files; a group that goes {stall}s without a result "
          f"is reported, not waited on")
    if done:
        print(f"  resume   {len(done)} groups already graded at this cap; CHECKER_CORPUS_FRESH=1 to redo them")
    print("  Groups are graded smallest first, so flash attention and the big GEMM sweeps -- the\n"
          "  slow half -- come last. A full uncapped run is tens of minutes at this worker count.\n")

    out = writer(NAME)
    rows, failed, dropped = [], [], 0
    pool = _Pool(workers, checker, recycle, stall)
    try:
        for kernel, dtype in groups:
            if (kernel, dtype) in done:
                print(f"  {kernel}/{dtype}: already graded, skipping")
                continue
            try:
                row, lost = _grade(kernel, dtype, os.path.join(corpus, kernel, dtype), pool, cap)
            except _PoolDied as exc:
                # Nothing is written for this group, so re-running the step retries it. Carrying
                # on is worth more than stopping: every other group is still measurable.
                failed.append((kernel, dtype, str(exc)))
                print(f"  {kernel}/{dtype}: NOT GRADED -- {exc}", flush=True)
                continue
            row.update(checker=checker, corpus=corpus)
            rows.append(row)
            dropped += lost
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"  {kernel}/{dtype}: configs={row['configs']} "
                f"checker={row.get('checker_cls', '-')} empirical={row.get('empirical_cls', '-')} "
                f"over_merges={row.get('over_merges', '-')} -> {row['recovery']}"
                f"{'  | ' + row['notes'] if row['notes'] else ''}", flush=True)
    finally:
        pool.close()
        out.close()

    for row in _resum(os.path.join(CACHE, f"{NAME}.jsonl")):
        if _key(row) != (corpus, cap, checker):
            continue
        print(f"  TOTAL {row['kernel']:11} configs={row['configs']} checker={row['checker_cls']} "
              f"empirical={row['empirical_cls']} over_merges={row['over_merges']}   ({row['notes']})")

    unsound = [r for r in rows if r.get("over_merges")]
    print(f"\n  {len(rows)} groups graded this run in {pool.generations} worker pools. "
          f"over_merges is the gate and must be 0.")
    if unsound:
        print("  OVER-MERGES FOUND -- the checker certified configurations the recorded bytes separate:")
        for r in unsound:
            print(f"    {r['kernel']}/{r['dtype']}: {r['over_merges']} pairs")
    else:
        print("  0 over-merges in every group graded this run.")

    if dropped:
        print(f"  {dropped} configurations were dropped because the checker could not read them. A\n"
              "  dropped configuration is in nobody's class, so the counts above are for what was\n"
              "  graded. The per-group notes say where they were.")
    if failed:
        print(f"\n  {len(failed)} GROUPS WERE NOT GRADED -- their worker pool died twice:")
        for kernel, dtype, why in failed:
            print(f"    {kernel}/{dtype}: {why}")
        print("  Nothing was written for them, so re-running this step grades them and nothing else.\n"
              "  A worker dies when the machine takes its memory away: check what else is running,\n"
              "  then lower CHECKER_CORPUS_WORKERS.")
    if cap:
        print(f"  Graded under cap={cap}, so every class count above is for a sample of at most {cap}\n"
              f"  configurations per group and is NOT comparable to the uncapped prior result in\n"
              f"  prior_results.txt. Unset CHECKER_CORPUS_CAP for the full table.")
