#ifndef TILE_SMT_EQUIVALENCE_H
#define TILE_SMT_EQUIVALENCE_H

// tile-smt core — final-memory equivalence check.
// MLIR-free: depends only on Z3 and the tile-smt neutral types (MemState /
// MemId). The witness / lambda SMT encoding is preserved byte-for-byte from the
// pre-M0 Triton-coupled implementation, so verdicts/timing do not change.

#include "semantics/Memory.h"
#include "semantics/Types.h"

#include <utility>
#include <vector>
#include <z3++.h>

namespace tile_smt {

// Check semantic equivalence of two final memory states.
//
// `pairing` lists which memory of `s1` corresponds to which memory of `s2`
// (src MemId, tgt MemId). The two programs come from different functions, so a
// builder pairs their memories explicitly (e.g. positionally by kernel
// argument) rather than assuming a shared MemId assignment.
//
// Adds to `solver` the assertion that at least one paired memory differs,
// checked POINTWISE at a single fresh symbolic witness address. Then calls
// solver.check().
//
//   z3::unsat   — no difference exists at any address → programs are equivalent
//   z3::sat     — counterexample found; call solver.get_model() for the witness
//   z3::unknown — solver timed out or gave up
//
// (The core takes an explicit pairing so it never needs to see source IR; a
// std::vector is used instead of std::span because the core targets C++17.)
z3::check_result
checkEquivalence(const MemState &s1, const MemState &s2,
                 const std::vector<std::pair<MemId, MemId>> &pairing,
                 z3::solver &solver);

} // namespace tile_smt

#endif // TILE_SMT_EQUIVALENCE_H
