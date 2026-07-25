#ifndef TV_BUILDER_MLIR_DTYPEOF_H
#define TV_BUILDER_MLIR_DTYPEOF_H

// builder/mlir — map an MLIR scalar type to the neutral tile_smt::DType.
// Shared by all MLIR-based builders (needs MLIR, so it lives outside the core).

#include "semantics/Types.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace Semantics {

// Any scalar mlir::Type -> tile_smt::DType.
//   IntegerType(1/8/16/32/64)        -> I1/I8/I16/I32/I64
//   Float16/BFloat16/Float32/Float64 -> F16/BF16/F32/F64
//   triton::PointerType              -> Ptr
// (the pointee of a pointer is recovered separately via dtypeOf on the
// pointee.)
tile_smt::DType dtypeOf(mlir::Type type);

// mlir::FloatType -> tile_smt::DType.
tile_smt::DType dtypeOf(mlir::FloatType type);

} // namespace Semantics

#endif // TV_BUILDER_MLIR_DTYPEOF_H
