#include "semantics/mlir/AbstractFpShim.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace Semantics;

tile_smt::DType Semantics::dtypeOf(mlir::Type type) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
    switch (intTy.getWidth()) {
    case 1:  return tile_smt::DType::I1;
    case 8:  return tile_smt::DType::I8;
    case 16: return tile_smt::DType::I16;
    case 32: return tile_smt::DType::I32;
    case 64: return tile_smt::DType::I64;
    default: llvm_unreachable("dtypeOf: unsupported integer width");
    }
  }
  if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(type))
    return dtypeOf(floatTy);
  if (llvm::isa<mlir::triton::PointerType>(type))
    return tile_smt::DType::Ptr;
  llvm_unreachable("dtypeOf: unsupported mlir::Type");
}

tile_smt::DType Semantics::dtypeOf(mlir::FloatType type) {
  if (llvm::isa<mlir::Float16Type>(type))
    return tile_smt::DType::F16;
  if (llvm::isa<mlir::BFloat16Type>(type))
    return tile_smt::DType::BF16;
  if (llvm::isa<mlir::Float32Type>(type))
    return tile_smt::DType::F32;
  if (llvm::isa<mlir::Float64Type>(type))
    return tile_smt::DType::F64;
  llvm_unreachable("dtypeOf: unsupported mlir::FloatType");
}

tile_smt::AbstractFp &Semantics::getFp(tile_smt::AbstractFpRegistry &reg,
                                       mlir::FloatType type) {
  return reg.get(dtypeOf(type));
}
