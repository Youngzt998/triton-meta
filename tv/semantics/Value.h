#ifndef TILE_SMT_VALUE_H
#define TILE_SMT_VALUE_H

// tile-smt core — value wrappers for encoded SSA values.
//
// A source value (scalar, tensor, or pointer) is encoded as one of these three
// Z3-backed structs. They are MLIR-free: types are the neutral tile_smt::DType,
// and pointer provenance is the opaque tile_smt::MemId (a builder maps its own
// source IR types / provenance onto these).
//
// These replace the old Semantics::Z3Scalar / Z3Tile / Z3Ptr, swapping the
// source-IR-typed fields for neutral ones:
//   Z3Scalar's source type    -> Scalar.ty     (DType)
//   Z3Tile's element type      -> Tensor.elem   (DType, = Ptr for ptr tiles)
//   Z3Tile's pointer base      -> Tensor.ptrBase (optional<MemId>)
//   Z3Ptr's pointee type       -> Ptr.pointee   (DType)
//   Z3Ptr's base argument      -> Ptr.base      (MemId)
// The per-value `fpMode` field is dropped: it is always the run's single mode,
// which lives on Memory / Context.

#include "semantics/Types.h"

#include <optional>
#include <variant>
#include <z3++.h>

namespace tile_smt {

// A scalar value encoded as a Z3 expression.
struct Scalar {
  z3::expr e;
  DType    ty;
};

// A tensor value encoded as a Z3 array: Array(BitVec(32), elem_sort).
// The index width is fixed at 32 bits. `shape` mirrors the source tensor dims.
//
// For pointer tiles (elem == DType::Ptr), `ptrBase` identifies which memory /
// address space all addresses in this tile derive from. An empty `ptrBase`
// means the provenance is unknown or the tile is not a pointer tile.
struct Tensor {
  z3::expr             e;        // Array(BitVec(32), elem_sort)
  Shape                shape;
  DType                elem;
  std::optional<MemId> ptrBase;  // set iff elem == DType::Ptr
};

// A scalar pointer value encoded as BitVec(64).
// Pointer tiles are represented as Tensor with elem == DType::Ptr.
//
// `base` identifies the memory / address space this pointer derives from.
struct Ptr {
  z3::expr e;        // BitVec(64)
  DType    pointee;
  MemId    base;
};

using Value = std::variant<Scalar, Tensor, Ptr>;

} // namespace tile_smt

#endif // TILE_SMT_VALUE_H
