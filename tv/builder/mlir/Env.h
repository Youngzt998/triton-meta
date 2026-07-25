#ifndef TV_BUILDER_MLIR_ENV_H
#define TV_BUILDER_MLIR_ENV_H

#include "semantics/Context.h"
#include "semantics/Types.h"
#include "semantics/Value.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include <map>
#include <stdexcept>
#include <string>

namespace Semantics {

// Builder-side value model uses the tile-smt core value types directly.
using tile_smt::Context;
using tile_smt::DType;
using tile_smt::FPMode;
using tile_smt::MemId;
using tile_smt::Ptr;
using tile_smt::Scalar;
using tile_smt::Tensor;
using tile_smt::Value;

// Pointer comparator for mlir::Value so it can be used as a std::map key.
// mlir::Value is not default-constructible, so llvm::DenseMap is not usable
// directly with Value (z3::expr is not default-constructible either).
struct ValuePtrLess {
  bool operator()(mlir::Value a, mlir::Value b) const {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  }
};

// Abstract program state for symbolic execution.
//
// Maps each live MLIR SSA value to its Z3 encoding (Scalar | Tensor | Ptr).
// Values are looked up by pointer identity — MLIR SSA values are unique objects
// in the module, so pointer comparison is the correct equality criterion.
class Env {
public:
  // Bind `val` to `z3val`. Overwrites any previous binding.
  void bind(mlir::Value val, Value z3val);

  // Returns the Z3 encoding of `val`.
  // Throws std::out_of_range if `val` is not bound.
  const Value &lookup(mlir::Value val) const;

  // Returns true if `val` has a binding.
  bool contains(mlir::Value val) const;

  // Number of bound values.
  size_t size() const;

private:
  std::map<mlir::Value, Value, ValuePtrLess> bindings_;
};

// Create a fresh unconstrained symbolic Value for a given MLIR type by reading
// the type (adapter side) and calling the core Context builder (freshInput /
// freshPtr).
//
// Scalars  → Scalar with a fresh constant of the appropriate sort.
// Tensors  → Tensor with a fresh Array(BitVec(32), elem_sort) constant.
// Pointers → Ptr    with a fresh BitVec(64) constant (base = MemId{0}; callers
//            that own a real memory, e.g. State::initFromFunc, set it after).
//
// `name` is used as the Z3 symbol name; it should be unique per call site
// (e.g. derived from the SSA value's debug name or argument index).
Value makeSymbolicValue(mlir::Type type, Context &context,
                        const std::string &name);

// Populate `env` with fresh symbolic values for each argument of `args`.
// Names are generated as `<prefix>_arg0`, `<prefix>_arg1`, etc.
void initFuncArgs(Env &env, mlir::ValueRange args, Context &context,
                  const std::string &prefix = "arg");

} // namespace Semantics

#endif // TV_BUILDER_MLIR_ENV_H
