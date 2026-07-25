#include "Env.h"

#include "semantics/mlir/AbstractFpShim.h"

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

bool Env::contains(mlir::Value val) const {
  return bindings_.count(val) > 0;
}

size_t Env::size() const { return bindings_.size(); }

//===----------------------------------------------------------------------===//
// makeSymbolicValue
//===----------------------------------------------------------------------===//

Value Semantics::makeSymbolicValue(mlir::Type type, z3::context &ctx,
                                   FPMode fpMode, const std::string &name) {
  // Ranked tensor → Tensor: fresh Array(BitVec(32), elem_sort)
  if (auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    mlir::Type elemTy = tensorTy.getElementType();
    DType    elem     = dtypeOf(elemTy);
    z3::sort elemSort = tile_smt::getElemSort(ctx, elem, fpMode);
    z3::sort arrSort  = ctx.array_sort(ctx.bv_sort(32), elemSort);
    z3::expr arr      = ctx.constant(name.c_str(), arrSort);
    tile_smt::Shape shape(tensorTy.getShape().begin(),
                          tensorTy.getShape().end());
    return Tensor{arr, std::move(shape), elem, std::nullopt};
  }

  // Triton pointer (scalar) → Ptr: fresh BitVec(64)
  // base is a placeholder (MemId{0}) here; callers that own the memory
  // (e.g. State::initFromFunc) set the real MemId after construction.
  if (auto ptrTy = llvm::dyn_cast<mlir::triton::PointerType>(type)) {
    z3::expr addr = ctx.bv_const(name.c_str(), 64);
    return Ptr{addr, dtypeOf(ptrTy.getPointeeType()), MemId{0}};
  }

  // Integer and float scalars → Scalar
  if (llvm::isa<mlir::IntegerType>(type) || llvm::isa<mlir::FloatType>(type)) {
    DType    ty   = dtypeOf(type);
    z3::sort sort = tile_smt::getElemSort(ctx, ty, fpMode);
    z3::expr val  = ctx.constant(name.c_str(), sort);
    return Scalar{val, ty};
  }

  // Fallback: treat as opaque 64-bit pointer (e.g. unrecognized dialect types).
  z3::expr addr = ctx.bv_const(name.c_str(), 64);
  return Ptr{addr, DType::Ptr, MemId{0}};
}

//===----------------------------------------------------------------------===//
// initFuncArgs
//===----------------------------------------------------------------------===//

void Semantics::initFuncArgs(Env &env, mlir::ValueRange args, z3::context &ctx,
                             FPMode fpMode, const std::string &prefix) {
  for (auto [idx, arg] : llvm::enumerate(args)) {
    std::string symName = prefix + "_arg" + std::to_string(idx);
    env.bind(arg, makeSymbolicValue(arg.getType(), ctx, fpMode, symName));
  }
}
