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

namespace Semantics {

// Keep the old adapter-facing names.
using AbstractFp = tile_smt::AbstractFp;
using AbstractFpRegistry = tile_smt::AbstractFpRegistry;

// Adapter-side classification: mlir::FloatType -> tile_smt::DType.
tile_smt::DType dtypeOf(mlir::FloatType type);

// Look up (or create) the AbstractFp for an mlir::FloatType, mapping via
// dtypeOf. Replaces the old `registry.get(mlir::FloatType)`.
tile_smt::AbstractFp &getFp(tile_smt::AbstractFpRegistry &reg,
                            mlir::FloatType type);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MLIR_ABSTRACTFPSHIM_H
