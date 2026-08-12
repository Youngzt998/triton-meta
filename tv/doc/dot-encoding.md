# Encoding `tt.dot` (contraction) in SMT

> **Status: proposed design note — NOT an adopted decision.** Question B1
> (`tt.dot` fidelity) is still open. If we adopt this, it must be written into
> `tile-smt-design.md` / `roadmap.md` as an explicit decision.
>
> **On novelty:** none of this is an original technique — see §8 and the
> citation list. It is a recombination of standard abstraction-refinement
> methods, instantiated for tile-level tensor IR.

## 1. The problem

`tt.dot(A, B, C)` computes `D[m,n] = C[m,n] + Σ_{k<K} A[m,k]·B[k,n]`.

The naive encoding — unfold all three dimensions into scalar multiply-adds —
costs `M×N×K` terms. For a routine Triton tile (`BLOCK_M=128, BLOCK_N=128,
BLOCK_K=64`) that is **~1.05 million** uninterpreted-function applications for a
*single* `tt.dot`, before any loop unrolling multiplies it further. Not viable.

Under Abstract FP the multiply-adds are uninterpreted functions with no
associativity, so the fold also has a *fixed order* — the encoding is both huge
and rigid.

**Goal: keep all three dimensions symbolic.** Each technique below removes one
source of unfolding; together they reduce a `tt.dot` to O(1) formula size.

## 2. Do not unfold K — a symbolic reduction term

Instead of folding `K-1` multiply-adds, introduce a canonical term
`Dot(A,B,m,n,K)` defined by recurrence axioms:

```
Dot(A,B,m,n,0)   = 0
Dot(A,B,m,n,j+1) = fp_add( Dot(A,B,m,n,j), fp_mul(A[m,j], B[j,n]) )
```

`K` stays symbolic; the solver instantiates the recurrence only as far as it
needs. This is the same mechanism as loop `summarize` mode (`roadmap.md` §M1) —
not a separate machine.

**The hook already exists in the code** but was never wired up
(`tv/semantics/AbstractFp.cpp`, `getDotFn`):

```cpp
z3::sort arr = ctx.array_sort(ctx.bv_sort(32), sort());
z3::sort doms[3] = {arr, arr, ctx.bv_sort(32)};
dotFn.emplace(ctx.function(("fp_dot_" + suffix).c_str(), 3, doms, sort()));
```

i.e. `fp_dot_<ty> : (Array(BV32,ty), Array(BV32,ty), BV32) -> ty` — two whole
operand arrays plus a symbolic `K`, returning one scalar. That signature is
exactly this design.

## 3. Do not unfold M and N — one lambda over the output index

```
D = λ idx. Dot(A, B, row(idx), col(idx), K)
```

The whole result tile is a single term. This is the same trick already used by
`Memory::store` (one `z3::lambda` per masked-tile store instead of an
element-wise store chain).

## 4. Only one output element is ever needed — the witness

Equivalence is already checked at a **symbolic witness index**
(`Equivalence.cpp`: `select(m1,w) != select(m2,w)`), never by comparing whole
arrays. So only `D[w]` for that one symbolic `w` has to be built. Combined with
§3 the lambda never has to be materialised.

This is Skolemisation of the goal `∀a. m1[a] = m2[a]` — standard first-order
logic, not a new idea; the *engineering* observation is that it avoids the
`unknown` that array extensionality returns on lambda-defined arrays.

## 5. Uninterpreted first, refine lazily

Keep `fp_dot(A,B,K)` **fully uninterpreted** by default and add the §2 recurrence
axioms only when needed.

Why this is strong: an SMT solver's **congruence closure** gives
`A₁=A₂ ∧ B₁=B₂ ∧ K₁=K₂ ⟹ dot₁=dot₂` for free, with zero axioms. In translation
validation the common case is that the pass under test *did not touch the dot's
inputs at all* — so the common case costs nothing.

Only when the two sides' `dot` terms differ syntactically do we pay for
refinement. This is standard abstraction-refinement (CEGAR / theory lemmas on
demand), and the closest precedent is UF-abstraction of bit-vector
multiplication with refinement [Bryant et al. 2007].

## 6. `dot` is not a new primitive — it is `reduce ∘ map`

```
contract(K) = reduce_k( map(mul, A, B) )
```

Defining it as a composition means it inherits the loop-summarization machinery,
the FP axiom profiles, and the split/merge lemmas automatically, instead of
needing its own parallel set. It also matches the abstract tensor operation set
(`tile-smt-design.md` §"Abstract tensor operation set"), where `contract(K)` is
already a listed category.

## 7. Separate index reasoning from arithmetic reasoning

Most real compiler miscompiles in tensor code are **indexing / layout** bugs
(which element ended up where), not **arithmetic** bugs. So:

- index math → **exact** integer reasoning (cheap, decidable);
- the arithmetic → **uninterpreted**.

The most common bug class is then caught by the cheapest encoding. This split is
the classic control-vs-datapath separation [Burch & Dill 1994], and is what
polyhedral equivalence checking does for affine programs
[Verdoolaege et al. 2012].

## 8. Interaction with the FP axiom profiles

Splitting the contraction (split-K, k-loop tiling) needs

```
Dot(0,K) = fp_add( Dot(0,j), Dot(j,K) )
```

which requires **associativity of `fp_add`**. Therefore:

| FP axiom profile | split-K provable? | correct? |
|---|---|---|
| **exact / bit-to-bit** | no | ✅ yes — split-K really does change the bits |
| **reassoc-allowed** | yes | ✅ yes — it is a legal reordering |

So the split lemma is *gated* by the profile (`tile-smt-design.md` §"FP axiom
profiles"), and the exact/reassoc pair distinguishes "deliberate performance
reordering" from "miscompile" for dot exactly as it does for reductions.

**`inputPrecision`.** `tt.dot` carries `tf32 / tf32x3 / ieee / bf16x3`. `tf32`
truncates its inputs, so the same dot at a different precision is *genuinely* a
different function. Modelling each as its own uninterpreted family
(`fp_dot_tf32_`, `fp_dot_ieee_`, …) is cheap and correctly reports a precision
change as not-equivalent.

## 9. Honest cost

Under **exact** profile with an uninterpreted `dot`, two dots that are truly
equal but *differently tiled* will report NOT EQUIVALENT — a false alarm. That
is the price of the abstraction. The three-way verdict plus profile switching is
what makes this usable rather than misleading.

## 10. Novelty assessment

**This is not an original technique.** Its parts are:

| Part | Prior art |
|---|---|
| Uninterpreted functions to abstract arithmetic (multiplication is undecidable) | Burch & Dill 1994; Pnueli et al. 1998 |
| Start uninterpreted, refine on spurious counterexample | Clarke et al. 2000 (CEGAR); Bryant et al. 2007 (specifically for BV multiplication) |
| A fold/aggregate over an array as a first-class term with symbolic length | Daca et al. 2016; Kincaid et al. 2015 (recurrence-based summaries) |
| Exact index reasoning + uninterpreted operators for equivalence | Verdoolaege et al. 2009/2012 |
| `matmul = reduction over a map` | standard in tensor IRs (Halide, TVM, MLIR `linalg.generic`) |
| Witness element instead of array extensionality | Skolemisation (textbook) |

Plausibly incremental, and small:
1. abstracting the **whole contraction** (both operand *arrays* + symbolic `K`
   in one UF application) rather than the usual scalar `mul(a,b)`, so all three
   dimensions collapse at once;
2. pairing that with the witness so exactly one output element is ever built;
3. gating the split lemma on an explicit FP axiom profile, making split-K
   provable *iff* reassociation is permitted.

For a paper, this section should be written as *"an instantiation of known
abstraction–refinement methods for tile-level tensor IR"*, **not** as a new
technique. The contribution lies elsewhere (the domain gap, the three-size-axis
analysis, the FP-profile diagnostic, the measured corpus data).

---

## Citations

> ⚠️ **UNVERIFIED — do not paste into a paper as-is.** These were written from
> memory while this machine had no web access (GitHub, dl.acm.org and most
> sites return 403). Author names and titles are the reliable part; **years and
> venues must each be checked** before use. Confidence is marked per entry.

| # | Reference | Used for | Confidence |
|---|---|---|---|
| 1 | J. R. Burch, D. L. Dill. *Automatic Verification of Pipelined Microprocessor Control.* CAV 1994. | UF abstraction of the datapath; control/datapath separation (§5, §7) | high |
| 2 | A. Pnueli, M. Siegel, E. Singerman. *Translation Validation.* TACAS 1998. | origin of translation validation; UF-based equivalence | high |
| 3 | E. Clarke, O. Grumberg, S. Jha, Y. Lu, H. Veith. *Counterexample-Guided Abstraction Refinement.* CAV 2000. | the refine-on-failure loop (§5) | high |
| 4 | R. Bryant, D. Kroening, J. Ouaknine, S. Seshia, O. Strichman, B. Brady. *Deciding Bit-Vector Arithmetic with Abstraction.* TACAS 2007. | **closest precedent**: UF-abstract multiplication, refine on spurious counterexample (§5) | medium-high |
| 5 | S. Verdoolaege, G. Janssens, M. Bruynooghe. *Equivalence Checking of Static Affine Programs Using Widening to Handle Recurrences.* CAV 2009; extended in ACM TOPLAS 2012. | **closest prior work overall**: exact index reasoning + uninterpreted operators + recurrences (§7) | medium |
| 6 | P. Daca, T. A. Henzinger, A. Kupriyanov. *Array Folds Logic.* CAV 2016. | aggregates/folds over arrays as first-class terms (§2) | medium-low (year uncertain) |
| 7 | Z. Kincaid, J. Breck, A. F. Boroujeni, T. Reps. *Compositional Recurrence Analysis* (FMCAD 2015) and *Non-linear Reasoning for Invariant Synthesis* (POPL 2018). | recurrence-based loop summaries (§2, §6) | medium |
| 8 | N. P. Lopes, J. Lee, C.-K. Hur, Z. Liu, J. Regehr. *Alive2: Bounded Translation Validation for LLVM.* PLDI 2021. | the comparable tool; see `tv/doc/kb/alive2-loops.md` | high (title verified from the repo README) |
| 9 | D. Menendez, S. Nagarakatte, A. Gupta. *Alive-FP: Automated Verification of Floating Point Based Peephole Optimizations in LLVM.* SAS 2016. | prior work on FP semantics in a translation validator (§8) | medium |
| 10 | L. de Moura, N. Bjørner. *Z3: An Efficient SMT Solver.* TACAS 2008. | the solver; congruence closure, lambdas | high |
| 11 | Y. Ge, L. de Moura. *Complete Instantiation for Quantified Formulas in Satisfiability Modulo Theories.* CAV 2009. | how the recurrence axioms get instantiated (§2) | medium |
| 12 | A. Liu, G. L. Bernstein, A. Chlipala, J. Ragan-Kelley. *Verified Tensor-Program Optimization via High-Level Scheduling Rewrites.* POPL 2022. | verified tensor-program rewriting (related, different approach) | medium-low |
| 13 | Z. Jia, O. Padon, J. Thomas, T. Warszawski, M. Zaharia, A. Aiken. *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions.* SOSP 2019. | tensor graph rewrite verification (related, graph level) | medium |

**Gap to check when the network is available:** whether anyone has published
translation validation for a *tile-level* GPU tensor DSL (Triton/TTIR, TileLang,
Pallas). We are not aware of any — but that claim is unverified and is exactly
the kind of thing a reviewer will test.
