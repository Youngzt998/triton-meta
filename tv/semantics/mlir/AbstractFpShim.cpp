#include "semantics/mlir/AbstractFpShim.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace Semantics;

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
