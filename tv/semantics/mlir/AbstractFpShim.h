#ifndef TRITON_TV_SEMANTICS_MLIR_ABSTRACTFPSHIM_H
#define TRITON_TV_SEMANTICS_MLIR_ABSTRACTFPSHIM_H

// Adapter shim for the MLIR-based builder.
//
// The Abstract FP model now lives in the MLIR-free core (tile_smt). This shim
// lets the Triton-coupled adapter keep using the old `Semantics::AbstractFp` /
// `Semantics::AbstractFpRegistry` names and look encodings up by
// `mlir::FloatType` (via `getFp`), while the registry itself is keyed by the
// neutral `tile_smt::DType`. Temporary: removed in M0 Step 5 when the adapter
// moves to builder/ and talks to the core directly.

#include "semantics/AbstractFp.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace Semantics {

// Keep the old adapter-facing names.
using AbstractFp = tile_smt::AbstractFp;
using AbstractFpRegistry = tile_smt::AbstractFpRegistry;

// Adapter-side classification: any scalar mlir::Type -> tile_smt::DType.
//   IntegerType(1/8/16/32/64)      -> I1/I8/I16/I32/I64
//   Float16/BFloat16/Float32/Float64 -> F16/BF16/F32/F64
//   triton::PointerType             -> Ptr
// (the pointee of a pointer is recovered separately via dtypeOf on the pointee.)
tile_smt::DType dtypeOf(mlir::Type type);

// Adapter-side classification: mlir::FloatType -> tile_smt::DType.
tile_smt::DType dtypeOf(mlir::FloatType type);

// Look up (or create) the AbstractFp for an mlir::FloatType, mapping via
// dtypeOf. Replaces the old `registry.get(mlir::FloatType)`.
tile_smt::AbstractFp &getFp(tile_smt::AbstractFpRegistry &reg,
                            mlir::FloatType type);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MLIR_ABSTRACTFPSHIM_H
