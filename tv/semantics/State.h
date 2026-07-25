#ifndef TRITON_TV_SEMANTICS_STATE_H
#define TRITON_TV_SEMANTICS_STATE_H

#include "semantics/mlir/AbstractFpShim.h"
#include "semantics/Env.h"
#include "semantics/Memory.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include <map>
#include <memory>
#include <z3++.h>

namespace Semantics {

using tile_smt::Memory;
using tile_smt::MemState;

// Complete symbolic program state at a point during execution.
//
// Memory model: one independent symbolic Memory per kernel pointer argument
// (Option B — separate-arrays model). This is sound because Triton kernels
// are required to pass non-aliasing pointer arguments; the compiler freely
// reorders accesses across distinct arguments. Each argument's Memory is an
// Array(BitVec(64), BitVec(8)) whose index is the absolute byte address.
//
// Per-argument memories live in `memState`, keyed by an opaque tile_smt::MemId.
// The adapter maps each pointer-argument mlir::Value to its MemId via
// `ptrArgToMem`; the core Memory / MemState never see mlir::Value.
//
// Non-pointer arguments (scalars, tensors passed by value) live only in Env.
//
// All interpret* methods are pure: they return an updated State and do not
// mutate *this. State is copy-constructible and move-constructible, but NOT
// copy/move-assignable (Memory holds z3::context by reference).
class State {
public:
  Env env;

  // One Memory per !tt.ptr<T> kernel argument, keyed by MemId.
  MemState memState;

  // Maps each pointer-argument mlir::Value to the MemId of its Memory.
  // The adapter owns this map (core never sees mlir::Value); handlers use it
  // only at function entry (initFromFunc) and equivalence pairing.
  std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem;

  // Future: Memory sharedMem;  // for TTGIR ttg.local_alloc / ttg.local_store

  // Shared Z3 context for all expressions in this state.
  z3::context &ctx;

  // FP encoding mode for this interpretation run.
  FPMode fpMode;

  // Registry of AbstractFp objects (uninterpreted FP function declarations).
  // Shared across all States derived from the same initFromFunc call so that
  // axiom emission is idempotent and function names are consistent.
  std::shared_ptr<AbstractFpRegistry> fpReg;

  State(Env env, MemState memState,
        std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem,
        z3::context &ctx, FPMode fpMode,
        std::shared_ptr<AbstractFpRegistry> fpReg);

  // Factory: build an initial State from a function's argument list.
  //
  // For each argument:
  //   - !tt.ptr<T>  → fresh Ptr in env (with base MemId set) +
  //                   fresh Memory in memState named "<prefix>_mem_argN"
  //   - tensor<...> → fresh Tensor in env
  //   - scalar      → fresh Scalar in env
  //
  // Call this once for each of the two programs being compared, using
  // distinct prefixes ("src", "tgt") so their Z3 symbols are distinguishable.
  // MemIds are minted in argument order, so the two programs' pointer arguments
  // pair up by position (src arg i ↔ tgt arg i get the same MemId).
  static State initFromFunc(mlir::ValueRange args, z3::context &ctx,
                            FPMode fpMode, const std::string &prefix = "arg");

  // Interpret a single MLIR operation and return the updated state.
  // Dispatches on dialect/op name. Unimplemented ops call llvm_unreachable.
  State interpretOp(mlir::Operation *op) const;

  // Interpret all ops in `block` sequentially.
  State interpretBlock(mlir::Block &block) const;

private:
  // Control flow handlers — framework stubs, concrete semantics deferred.

  // scf.if: execute both branches from *this; merge with z3::ite on condition.
  State interpretIf(mlir::Operation *op) const;

  // scf.for — static unrolling (deferred).
  State interpretFor(mlir::Operation *op) const;

  // scf.while — deferred.
  State interpretWhile(mlir::Operation *op) const;
};

// Check semantic equivalence of two final States.
//
// Adds to `solver` the assertion that the output memories differ in at least
// one pointer argument (disjunction over all memories, paired by MemId). Then
// calls solver.check().
//
//   z3::unsat   — no difference exists → programs are equivalent
//   z3::sat     — counterexample found; call solver.get_model() for witness
//   z3::unknown — solver timed out
//
// Both states must have been derived from the same initial symbolic inputs and
// the same MemId assignment (e.g. initFromFunc on the same argument list).
z3::check_result checkEquivalence(const State &s1, const State &s2,
                                  z3::solver &solver);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_STATE_H
