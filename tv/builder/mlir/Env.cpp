#include "builder/mlir/Env.h"

#include "builder/mlir/DTypeOf.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Casting.h"

#include <stdexcept>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Env
//===----------------------------------------------------------------------===//

void Env::bind(mlir::Value val, Value z3val) {
  bindings_.insert_or_assign(val, std::move(z3val));
}

const Value &Env::lookup(mlir::Value val) const {
  auto it = bindings_.find(val);
  if (it == bindings_.end())
    throw std::out_of_range("Env::lookup: value not bound");
  return it->second;
}

bool Env::contains(mlir::Value val) const { return bindings_.count(val) > 0; }

size_t Env::size() const { return bindings_.size(); }

//===----------------------------------------------------------------------===//
// makeSymbolicValue — read the MLIR type (adapter), build via core Context
//===----------------------------------------------------------------------===//

Value Semantics::makeSymbolicValue(mlir::Type type, Context &context,
                                   const std::string &name) {
  // Ranked tensor → Tensor: fresh Array(BitVec(32), elem_sort)
  if (auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    DType elem = dtypeOf(tensorTy.getElementType());
    tile_smt::Shape shape(tensorTy.getShape().begin(),
                          tensorTy.getShape().end());
    return context.freshInput(elem, shape, name);
  }

  // Triton pointer (scalar) → Ptr: fresh BitVec(64)
  // base is a placeholder (MemId{0}) here; callers that own the memory
  // (e.g. State::initFromFunc) set the real MemId after construction.
  if (auto ptrTy = llvm::dyn_cast<mlir::triton::PointerType>(type))
    return context.freshPtr(dtypeOf(ptrTy.getPointeeType()), MemId{0}, name);

  // Integer and float scalars → Scalar (empty shape → freshInput returns
  // Scalar)
  if (llvm::isa<mlir::IntegerType>(type) || llvm::isa<mlir::FloatType>(type))
    return context.freshInput(dtypeOf(type), tile_smt::Shape{}, name);

  // Fallback: treat as opaque 64-bit pointer (e.g. unrecognized dialect types).
  return context.freshPtr(DType::Ptr, MemId{0}, name);
}

//===----------------------------------------------------------------------===//
// initFuncArgs
//===----------------------------------------------------------------------===//

void Semantics::initFuncArgs(Env &env, mlir::ValueRange args, Context &context,
                             const std::string &prefix) {
  for (auto [idx, arg] : llvm::enumerate(args)) {
    std::string symName = prefix + "_arg" + std::to_string(idx);
    env.bind(arg, makeSymbolicValue(arg.getType(), context, symName));
  }
}
