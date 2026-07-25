#include "semantics/Equivalence.h"

namespace tile_smt {

z3::check_result
checkEquivalence(const MemState &s1, const MemState &s2,
                 const std::vector<std::pair<MemId, MemId>> &pairing,
                 z3::solver &solver) {
  // Assert that at least one paired memory differs between the two programs,
  // checked POINTWISE at a single fresh symbolic witness address. UNSAT means
  // the heaps agree at every address → equivalent.
  //
  // We deliberately avoid `mem1.array != mem2.array` (array extensionality):
  // store() builds each heap as a z3::lambda, and extensionality between two
  // lambda arrays is undecidable for the default DPLL(T) solver (returns
  // `unknown`). select(lambda, witness) instead β-reduces to a quantifier-free
  // byte formula. This is equivalent in meaning — m1 = m2 ⇔ ∀a. m1[a] = m2[a],
  // and the fresh witness `a` quantifies the disagreement existentially.
  z3::context &ctx = solver.ctx();
  z3::expr witness = ctx.bv_const("__witness_addr", 64);
  z3::expr anyDiffers = ctx.bool_val(false);
  for (const auto &[srcId, tgtId] : pairing) {
    const Memory &mem1 = s1.mems.at(srcId);
    const Memory &mem2 = s2.mems.at(tgtId);
    anyDiffers = anyDiffers || (z3::select(mem1.array, witness) !=
                                z3::select(mem2.array, witness));
  }
  solver.add(anyDiffers);
  return solver.check();
}

} // namespace tile_smt
